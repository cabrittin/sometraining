"""
@name: embryo_detection.py
@description:                  

@author: Christopher Brittin   
@email: "cabrittin"+ <at>+ "gmail"+ "."+ "com"
@date: 
"""

import sys
import os
import argparse
from inspect import getmembers,isfunction
import json
import cv2
import numpy as np

from pycsvparser import read

def array_16bit_to_8bit(a):
    # If already 8bit, return array
    if a.dtype == 'uint8': return a
    a = np.array(a,copy=True)
    display_min = np.amin(a)
    display_max = np.amax(a)
    a.clip(display_min, display_max, out=a)
    a -= display_min
    np.floor_divide(a, (display_max - display_min + 1) / 256,
                    out=a, casting='unsafe')
    return a.astype('uint8') 

def load_data_map(cfg):
    ddir = os.path.join(cfg['root'],cfg['data_dir']) 
    dmap = read.into_list(os.path.join(cfg['root'],cfg['data_map']),multi_dim=True)
    for (i,d) in enumerate(dmap):
        dmap[i][0] = os.path.join(ddir,d[0])
        dmap[i][1] = os.path.join(ddir,d[1])
    return dmap

def rois_from_file(fname):
    @read.parse_file(fname,multi_dim=True)
    def row_into_container(container,row=None,**kwargs):
        roi = [[int(row[1]),int(row[2])],[int(row[3]),int(row[4])]]
        container.append(roi)

    container = []
    row_into_container(container)
    return container

def add_roi_to_image(I,rois):
    I = cv2.cvtColor(I,cv2.COLOR_GRAY2RGB)
    for [r0,r1] in rois: cv2.rectangle(I,r0,r1,(255,0,0),2)
    return I

def build(args):
    """
    Builds augmented training data for embryo recognition
    
    Parameters:
    -----------
    args: Argsparse from commandline
    """
    fin = open(args.json)
    cfg = json.load(fin)
    
    test_x = os.path.join(cfg['root'],cfg['test_x'])
    test_y = os.path.join(cfg['root'],cfg['test_y'])
    train_x = os.path.join(cfg['root'],cfg['train_x'])
    train_y = os.path.join(cfg['root'],cfg['train_y'])
    dmap = load_data_map(cfg) 

    print(f"Writing train_x image data to: {train_x}") 
    print(f"Writing train_y image data to: {train_y}") 
    print(f"Writing test_x image data to: {test_x}") 
    print(f"Writing test_y image data to: {test_y}") 

def view_augmentation(args):
    fin = open(args.json)
    cfg = json.load(fin)
    dmap = load_data_map(cfg) 
    
    win = 'Augmentation'
    cv2.namedWindow(win)
    cv2.moveWindow(win,300,500)
    
    rois = rois_from_file(dmap[0][1])
    I = np.load(dmap[0][0])
    I = array_16bit_to_8bit(I)

    m,n = I.shape
    Z = np.zeros((2*m,3*n,3),dtype=np.uint8)

    Z[:m,:n] = add_roi_to_image(I,rois)
    
    #Flip Left/Right
    A,r = flip_lr(I,rois)
    Z[:m,n:2*n] = add_roi_to_image(A,r)

    #Flip Up/Down
    A,r = flip_ud(I,rois)
    Z[:m,2*n:] = add_roi_to_image(A,r)

    #Blank 
    A,r = random_blank_rois(I,rois,p_blank=0.5)
    Z[m:2*m,:n] = add_roi_to_image(A,r)
    
    #Rotate
    A,r = random_rotate_rois(I,rois)
    Z[m:2*m,n:2*n] = add_roi_to_image(A,r)
    
    #Random augment
    A,r = augment(I,rois)
    Z[m:2*m,2*n:] = add_roi_to_image(A,r)

    Z = cv2.resize(Z, (0,0), fx=0.35, fy=0.35)
    cv2.imshow(win,Z)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def augment(I,rois):
    p_blank = np.random.rand()
    A,r = random_blank_rois(I,rois,p_blank=p_blank)
    A,r = random_rotate_rois(A,r)
    if np.random.rand() < 0.5: 
        A,r = flip_lr(A,r)
    if np.random.rand() < 0.5:
        A,r = flip_ud(A,r)
    return A,r

def flip_lr(I,_rois):
    m,n = I.shape
    A = np.flip(I,axis=1)
    rois = [] 
    for r in _rois:
        tmp = [[n-r[0][0],r[0][1]],[n-r[1][0],r[1][1]]]
        rois.append(tmp)
    return A,rois

def flip_ud(I,_rois):
    _rois = _rois.copy()
    m,n = I.shape
    A = np.flip(I,axis=0)
    rois = [] 
    for r in _rois:
        tmp = [[r[0][0],m-r[0][1]],[r[1][0],m-r[1][1]]]
        rois.append(tmp)
    return A,rois

def random_blank_rois(I,_rois,p_blank=0.5):
    A = np.copy(I)
    r = _rois[0]
    dx = abs(r[1][1] - r[0][1])
    dy = abs(r[1][0] - r[0][0])
    m,n = I.shape

    B = I[:dy,n-dx:]
    
    rois = []
    make_blank = np.random.rand(len(_rois)) < p_blank
    for (i,r) in enumerate(_rois):
        if make_blank[i]: 
            A[r[0][1]:r[1][1],r[0][0]:r[1][0]] = B
        else: 
            rois.append(r)
    return A,rois

def random_rotate_rois(I,_rois):
    A = I[:,:] 
    rot_val = np.random.randint(size=len(_rois),low=0,high=4)
    for (i,r) in enumerate(_rois):
        a = A[r[0][1]:r[1][1],r[0][0]:r[1][0]] 
        a = np.rot90(a,rot_val[i]) 
        A[r[0][1]:r[1][1],r[0][0]:r[1][0]] = a
         
    return A,_rois 


if __name__=="__main__":
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument('mode',
                        action = 'store',
                        choices = [t for (t,o) in getmembers(sys.modules[__name__]) if isfunction(o)],
                        help = 'Function call')
    
    parser.add_argument('json',
                action = 'store',
                help = 'Path to .json config file')
    
    args = parser.parse_args()
    eval(args.mode + '(args)')


