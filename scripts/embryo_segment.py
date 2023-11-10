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
import pandas as pd
import json
import matplotlib.pyplot as plt
import glob
from natsort import natsorted


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
    
    test_x = os.path.join(cfgs['root_dir_to'],cfgs['test_x_dir'])
    test_y = os.path.join(cfgs['root_dir_to'],cfgs['test_y_dir'])
    train_x = os.path.join(cfgs['root_dir_to'],cfgs['train_x_dir'])
    train_y = os.path.join(cfgs['root_dir_to'],cfgs['train_y_dir'])
    
    print(f"Writing train_x image data to: {train_x}") 
    print(f"Writing train_y image data to: {train_y}") 
    print(f"Writing test_x image data to: {test_x}") 
    print(f"Writing test_y image data to: {test_y}") 

    for ini in cfgs['ini']:
        ini = os.path.join(cfgs['root_dir_from'],ini)
        print(f"Piping training data from {ini}")
        segment_dataset(ini,
                test_x,test_y,train_x,train_y,
                test_split = cfgs['train_test_split'],
                num_sample= cfgs['num_sample'])  

def check_masks(args):
    """"
    Pipes training data for staging

    args.ini file must be a json file

    """
    fin = open(args.ini)
    cfgs = json.load(fin)
    
    test_x = os.path.join(cfgs['root_dir_to'],cfgs['test_x_dir'])
    test_y = os.path.join(cfgs['root_dir_to'],cfgs['test_y_dir'])
    train_x = os.path.join(cfgs['root_dir_to'],cfgs['train_x_dir'])
    train_y = os.path.join(cfgs['root_dir_to'],cfgs['train_y_dir'])
    
    for ini in cfgs['ini']:
        ini = os.path.join(cfgs['root_dir_from'],ini)
        print(f"Piping training data from {ini}")
        check_dataset(ini,
                test_x,test_y,train_x,train_y,
                test_split = cfgs['train_test_split'],
                num_sample= cfgs['num_sample'])  


def preprocess(args):
    fin = open(args.ini)
    cfgs = json.load(fin)
    
    stats = []
    for ini in cfgs['ini']:
        ini = os.path.join(cfgs['root_dir_from'],ini)
        stats.append(get_segment_stats(ini))


def view_preprocess(args):
    fin = open(args.ini)
    cfgs = json.load(fin)
    
    for ini in cfgs['ini']:
        ini = os.path.join(cfgs['root_dir_from'],ini)
        view_discriminator(ini)



def get_iopiars(S): 
    vdir = S.cfg['structure']['volumes']
    mdir = S.cfg['structure']['mask_volumes']
    files = os.listdir(vdir)
    iopairs = []
    for f in files:
        vpath = os.path.join(vdir,f)
        if not os.path.isfile(vpath): continue
        mpath = os.path.join(mdir,f"mask_{f}")
        iopairs.append((vpath,mpath))
    return iopairs


def get_segment_stats(ini,num_sample=200):
    S = Session(ini)
    dx = 2*S.cfg.getint('roi','dx')
    dy = 2*S.cfg.getint('roi','dy')
    
    iopairs = get_iopiars(S)

    pdiff = np.zeros((len(iopairs)*num_sample,3)) 
    idx = 0
    for (jdx,(vol_path,mask_path)) in tqdm(enumerate(iopairs),total=len(iopairs),desc='I/O pairs'):
        
        Z = np.load(vol_path)
        M = np.load(mask_path)
        
        for i in range(num_sample):
            rdx = random.randint(0,Z.shape[0]-1)
            img = Z[rdx,:] 
            mask = M[rdx,:] 
            m0 = np.where(mask==0)[0]
            m1 = np.where(mask==1)[0] 
     
            pdiff[idx,:2] = mask_compare(img,mask) 
            pdiff[idx,2] = int(keep_mask(pdiff[idx,0],pdiff[idx,1]))
            
            
            idx += 1

    print(pdiff.mean(0))
    print(pdiff.std(0))

    fig,ax = plt.subplots(1,1,figsize=(5,5))
    ax.scatter(pdiff[:,0],pdiff[:,1],s=2,c=pdiff[:,2],cmap='bwr')
    plt.show()


def view_discriminator(ini,num_sample=200):
    S = Session(ini)
    dx = 2*S.cfg.getint('roi','dx')
    dy = 2*S.cfg.getint('roi','dy')
    
    iopairs = get_iopiars(S)
    
    wkeep = 'Keep'
    cv2.namedWindow(wkeep)
    cv2.moveWindow(wkeep,300,500)
    
    wrm = 'Remove'
    cv2.namedWindow(wrm)
    cv2.moveWindow(wrm,800,500)
 
    idx = 0
    for (jdx,(vol_path,mask_path)) in tqdm(enumerate(iopairs),total=len(iopairs),desc='I/O pairs'):
        
        Z = np.load(vol_path)
        M = np.load(mask_path)
        
        for i in range(num_sample):
            rdx = random.randint(0,Z.shape[0]-1)
            img = Z[rdx,:] 
            mask = M[rdx,:] 
            m0 = np.where(mask==0)[0]
            m1 = np.where(mask==1)[0] 
     
            norm_pdiff,msize = mask_compare(img,mask) 
            
            img = loader.array_16bit_to_8bit(img) 
            mask = M[rdx,:]
            
            img = img.reshape(dy,dx)
            mask = mask.reshape(dy,dx)

            image = (img * mask).astype('uint8')
            
            if keep_mask(norm_pdiff,msize):
                cv2.imshow(wkeep,image)
            else:
                cv2.imshow(wrm,image)
            
            if cv2.waitKey(1) & 0xFF == ord('q'): break


def segment_dataset(ini,test_x,test_y,train_x,train_y,test_split=0.2,num_sample=200):
    S = Session(ini)
    dx = 2*S.cfg.getint('roi','dx')
    dy = 2*S.cfg.getint('roi','dy')
    
    vdir = S.cfg['structure']['volumes']
    mdir = S.cfg['structure']['mask_volumes']
    files = os.listdir(vdir)
    iopairs = []
    for f in files:
        vpath = os.path.join(vdir,f)
        if not os.path.isfile(vpath): continue
        mpath = os.path.join(mdir,f"mask_{f}")
        iopairs.append((vpath,mpath))

    num_test = int(test_split * len(iopairs))
    to_test = np.zeros(len(iopairs),dtype=np.uint8)
    to_test[:num_test] = 1
    np.random.shuffle(to_test)

    _iout = [train_x,test_x]
    _mout = [train_y,test_y]
   
    idx = [0,0]
    for (jdx,(vol_path,mask_path)) in tqdm(enumerate(iopairs),total=len(iopairs),desc='I/O pairs'):
        iout = _iout[to_test[jdx]]
        mout = _mout[to_test[jdx]]
        
        Z = np.load(vol_path)
        M = np.load(mask_path)
        
        for i in range(num_sample):
            rdx = random.randint(0,Z.shape[0]-1)
            img = Z[rdx,:] 
            mask = M[rdx,:] 
            m0 = np.where(mask==0)[0]
            m1 = np.where(mask==1)[0] 
     
            norm_pdiff,msize = mask_compare(img,mask) 
 
            #img = loader.array_16bit_to_8bit(img) 
            img = img.reshape(dy,dx)
            mask = mask.reshape(dy,dx)
            img,mask = random_rotate(img,mask)
            img,mask = random_flip(img,mask)
            
            if keep_mask(norm_pdiff,msize):
                np.save(iout%idx[to_test[jdx]],img.astype('uint8'))
                np.save(mout%idx[to_test[jdx]],mask.astype('uint8'))

                idx[to_test[jdx]] += 1

def check_dataset(ini,test_x,test_y,train_x,train_y,test_split=0.2,num_sample=200):
    train_x = os.path.dirname(train_x)
    train_y = os.path.dirname(train_y)
    img_path = natsorted([f for f in glob.iglob(f'{train_x}/*')])
    mask_path = natsorted([f for f in glob.iglob(f'{train_y}/*')])
    
    wkeep = 'Check'
    cv2.namedWindow(wkeep)
    cv2.moveWindow(wkeep,300,500)
 
    for i in range(len(img_path)):
        img = np.load(img_path[i])
        mask = np.load(mask_path[i])

        image = (img * mask).astype('uint8')

        cv2.imshow(wkeep,image)

        if cv2.waitKey(1) & 0xFF == ord('q'): break

def mask_compare(img,mask):
    mask_0 = np.where(mask==0)[0]
    mask_1 = np.where(mask==1)[0] 
    
    mu0,mu1,std = 1,1,1
    if len(mask_0) > 0: mu0 = img[mask_0].mean()
    if len(mask_1) > 0: mu1 = img[mask_1].mean()
    #mudiff = np.log(mu1) - np.log(mu0)
    norm_pdiff = np.log(mu1) - np.log(mu0)
    #std = np.log(img[mask_1].std())
    msize = len(mask_1) 
    return [norm_pdiff,msize] 

def keep_mask(norm_pdiff,msize):
    #return mudiff > 0.06 or (std > 5.25 and mudiff > 0)
    #return mudiff > 0.1
    #return mudiff > 0.25
    c1 = norm_pdiff >= -1 and norm_pdiff <= -0.1
    c2 = msize >= 2000 and msize <= 4300
    return c1 and c2

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

