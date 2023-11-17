"""
@name: generate_training.py                         
@description:                  
    Script for generating training data

@author: Christopher Brittin   
@email: "cabrittin"+ <at>+ "gmail"+ "."+ "com"
@date: 2019-12-05              
"""

import sys
import os
import argparse
from inspect import getmembers,isfunction
import cv2
import numpy as np
from tqdm import tqdm
import random 
import matplotlib.pyplot as plt
import pandas as pd
import json

from pycsvparser import read,write
from dataset.loader import Session
import dataset.loader as loader

def build(args):
    """"
    Pipes training data for staging

    args.ini file must be a json file

    """
    fin = open(args.ini)
    cfgs = json.load(fin)
    
    test_out = os.path.join(cfgs['root_dir_to'],cfgs['test_dir'])
    train_out = os.path.join(cfgs['root_dir_to'],cfgs['train_dir'])
    print(f"Writing train image data to: {train_out}") 
    print(f"Writing test image data to: {test_out}") 


    test_labels = []
    train_labels = []
    for ini in cfgs['ini']:
        ini = os.path.join(cfgs['root_dir_from'],ini)
        print(f"Piping training data from {ini}")
        stage_dataset(ini,test_out,train_out,
                test_split = cfgs['train_test_split'],
                test_labels = test_labels, train_labels = train_labels)
    
    
    test_labels_out = os.path.join(cfgs['root_dir_to'],cfgs['test_labels'])
    train_labels_out = os.path.join(cfgs['root_dir_to'],cfgs['train_labels'])
        
    write.from_list(test_labels_out,test_labels)
    write.from_list(train_labels_out,train_labels)

    print(f"Writing train labels to: {train_labels_out}") 
    print(f"Writing test labels to: {test_labels_out}") 

   

def stage_dataset(ini,test_out,train_out,test_split=0.2,test_labels=[],train_labels=[]):
    S = Session(ini)
    dx = 2*S.cfg.getint('roi','dx')
    dy = 2*S.cfg.getint('roi','dy')
    
    baseline_time_correction = 1.16
    num_frames = S.cfg.getfloat('analysis','num_frames')
    img_duration = S.cfg.getfloat('analysis','image_duration_hours')
    dt = img_duration*3600 / num_frames
    dt = dt / 60. * baseline_time_correction

    vdir = S.cfg['structure']['volumes']
    mdir = S.cfg['structure']['mask_volumes']

    df = pd.read_csv(S.cfg['structure']['process_log'])
    df = df[df['flag']==0] 
    df = df[df['trained']==1] 
    
    num_test = int(test_split * len(df))
    
    to_test = np.zeros(len(df),dtype=np.uint8)
    to_test[:num_test] = 1
    np.random.shuffle(to_test)

    
    test_kdx = len(test_labels)
    train_kdx = len(train_labels)
    for (idx,row) in tqdm(enumerate(df.to_dict(orient="records")),total=len(df),desc="Sequences clipped"):
        ipath = os.path.join(vdir,row['image_file'])
        mpath = os.path.join(mdir,row['mask_file'])
        Z = np.load(ipath)
        num_slices = len(Z)
        
        # If a highly sampled datasets then only take every 100th frame
        zloop = range(num_slices)
        if num_slices > 500: zloop = zloop[::50]

        ## Mask has to be loaded to use the data augmentation
        ## Masks are not actually applied
        M = np.load(mpath)
        

        for jdx in zloop:
            img = Z[jdx,:] 
            mask = M[jdx,:]
            
            #label_comma = get_label_comma(jdx,row)
            label_fold = get_label_fold(jdx,row)
            label_2fold = get_label_two_fold(jdx,row)
            label_all = get_all_labels(jdx,row)
            time_score = get_time_score(jdx,row,dt)

            if label_fold == -1: continue
            
            labels = [label_fold, label_2fold, label_all, time_score]

            if to_test[idx] == 1:
                img = img.reshape(dy,dx)
                mask = mask.reshape(dy,dx)
                image = loader.array_16bit_to_8bit(img)
                fout = test_out % test_kdx
                np.save(fout, image)
                test_labels.append([os.path.basename(fout)] + labels)
                test_kdx += 1

            else:
                img = img.reshape(dy,dx)
                mask = mask.reshape(dy,dx)
                image = loader.array_16bit_to_8bit(img)
                fout = train_out % train_kdx
                np.save(fout, image)
                train_labels.append([os.path.basename(fout)] + labels)
                train_kdx += 1
                
                """
                #Rotate
                image,mask = random_rotate(image,mask)
                fout = train_out % train_kdx
                np.save(fout, image)
                train_labels.append([os.path.basename(fout)] + labels)
                train_kdx += 1
 
                #Flip
                image,mask = random_flip(image,mask)
                fout = train_out % train_kdx
                np.save(fout, image)
                train_labels.append([os.path.basename(fout)] + labels)
                train_kdx += 1
                """

def get_label_comma(jdx,row):
    label = 0
    if jdx >= row['comma']: label = 1
    if row['hatch'] > -1 and jdx >= row['hatch']: label = -1
    return label

def get_label_fold(jdx,row):
    label = 0
    if jdx >= row['1.5-fold']: label = 1
    if row['hatch'] > -1 and jdx >= row['hatch']: label = -1
    return label

def get_label_two_fold(jdx,row):
    label = 0
    if jdx >= row['2-fold']: label = 1
    if row['hatch'] > -1 and jdx >= row['hatch']: label = -1
    return label

def get_all_labels(jdx,row):
    label = 0
    if jdx >= row['comma'] and jdx <row['1.5-fold']: label = 1
    if jdx >= row['1.5-fold'] and jdx <row['2-fold']: label = 2
    if jdx >= row['2-fold']: label = 3
    if row['hatch'] > -1 and jdx >= row['hatch']: label = -1

    return label

def get_time_score(jdx,row,dt,scale=400):
    ref_pt = row['1.5-fold']
    score = ((jdx - ref_pt) * dt) / scale
    score = max(-1,score)
    score = min(1,score)
    return score



def get_mean_and_std(args):
    fin = open(args.ini)
    cfgs = json.load(fin)
    
    train_in = os.path.join(cfgs['root_dir_to'],cfgs['train_dir'])
    train_labels = os.path.join(cfgs['root_dir_to'],cfgs['train_labels'])
    tpairs = read.into_list(train_labels,multi_dim=True)[::50]
    num_imgs = len(tpairs)
    
    Z = np.zeros((num_imgs,2))
    for idx in tqdm(range(num_imgs),desc="Images processed"):
        fin = train_in % idx
        img = np.load(fin) / 255.
        Z[idx,0] = np.mean(img)
        Z[idx,1] = np.std(img)

    print(Z.mean(0))


def mask_compare(img,mask_0,mask_1): 
    mu0 = img[mask_0].mean()
    mu1 = img[mask_1].mean()
    mudiff = np.log(mu0) - np.log(mu1)
    std = np.log(img[mask_1].std())
    return [mudiff,std] 

def keep_mask(mudiff,std):
    #return mudiff > 0.06 or (std > 5.25 and mudiff > 0)
    return mudiff > 0.1
    #return mudiff > 0.25

def convert_16_to_8_bit(img):
    img = img - np.amin(img)
    ratio = np.amax(img) / 256 
    img = (img / ratio)
    return img.astype('uint8')

def random_rotate(img,mask):
    rot_val = np.random.randint(low=0,high=4)
    img = np.rot90(img,rot_val)
    mask = np.rot90(mask,rot_val)
    return img,mask

def random_flip(img,mask):
    flip_val = np.random.randint(low=0,high=2)
    img = np.flip(img,flip_val)
    mask = np.flip(mask,flip_val)
    return img,mask

if __name__=="__main__":
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument('mode',
                        action = 'store',
                        choices = [t for (t,o) in getmembers(sys.modules[__name__]) if isfunction(o)],
                        help = 'Function call')
    
    parser.add_argument('ini',
                action = 'store',
                help = 'Path to .ini config file')
    
    parser.add_argument('--viz',
                dest = 'viz', 
                action = 'store_true',
                default = False,
                required = False,
                help = 'If True, then vizualize output')
    
    parser.add_argument('-n','--num_jobs',
            dest = 'num_jobs',
            action = 'store',
            default = 2,
            type=int,
            required = False,
            help = 'Number of parallel jobs')


    args = parser.parse_args()
    eval(args.mode + '(args)')

