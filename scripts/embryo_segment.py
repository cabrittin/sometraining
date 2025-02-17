"""
@name: embryo_segment.py 
@description:                  
   Generate segment data 

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
from tqdm import tqdm

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
        dmap[i][2] = os.path.join(ddir,d[2])
    return dmap

def load_data(dmap,idx):
    I = np.load(dmap[idx][0])
    I = array_16bit_to_8bit(I)
    return I,np.load(dmap[idx][1]),np.load(dmap[idx][2])

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

    I,rois,masks = load_data(dmap,0)
    
    tt_split = cfg['train_test_split']
    train_idx = 0
    test_idx = 0
    
    N = len(rois)

    for i in tqdm(range(N),desc="Samples processed"):
        [x0,y0,x1,y1] = rois[i,:]
        I0 = I[y0:y1,x0:x1]
        mask = masks[0,:,:]
        for j in range(3):
            A0 = np.copy(I0) 
            M0 = np.copy(mask)
            if j < 2: A0,M0 = flip(I0,mask,j)
            for k in range(4):
                A,M = rotate(A0,M0,k)
                if np.random.rand() < tt_split:
                    np.save(test_x%test_idx,A)
                    np.save(test_y%test_idx,M)
                    test_idx += 1
                else:
                    np.save(train_x%train_idx,A)
                    np.save(train_y%train_idx,M)
                    train_idx += 1 


def view_augmentation(args):
    fin = open(args.json)
    cfg = json.load(fin)
    dmap = load_data_map(cfg) 
 
    win1 = 'Augmentation-1'
    cv2.namedWindow(win1)
    cv2.moveWindow(win1,300,100)

    I,rois,masks = load_data(dmap,0)

    [x0,y0,x1,y1] = rois[0,:]
    I = I[y0:y1,x0:x1]
    mask = masks[0,:,:]

    m,n = I.shape
    Z = np.zeros((7*m,3*n),dtype=np.uint8)
    
    #Original mask
    Z[:m,:n] = I
    Z[:m,n:2*n] = 255*mask
    Z[:m,2*n:3*n] = np.multiply(I,mask)
    
    #Rotate 0
    A,M = rotate(I,mask,0)
    Z[m:2*m,:n] = A
    Z[m:2*m,n:2*n] = 255*M 
    Z[m:2*m,2*n:3*n] = np.multiply(A,M)
    
    #Rotate 1
    A,M = rotate(I,mask,1)
    Z[2*m:3*m,:n] = A
    Z[2*m:3*m,n:2*n] = 255*M 
    Z[2*m:3*m,2*n:3*n] = np.multiply(A,M)
    
    #Rotate 2
    A,M = rotate(I,mask,2)
    Z[3*m:4*m,:n] = A
    Z[3*m:4*m,n:2*n] = 255*M 
    Z[3*m:4*m,2*n:3*n] = np.multiply(A,M)
    
    #Rotate 3
    A,M = rotate(I,mask,3)
    Z[4*m:5*m,:n] = A
    Z[4*m:5*m,n:2*n] = 255*M 
    Z[4*m:5*m,2*n:3*n] = np.multiply(A,M)
    
    #Flip Up/Down
    A,M = flip(I,mask,0)
    Z[5*m:6*m,:n] = A
    Z[5*m:6*m,n:2*n] = 255*M 
    Z[5*m:6*m,2*n:3*n] = np.multiply(A,M)
    
    #Flip Left/Rigth
    A,M = flip(I,mask,1)
    Z[6*m:7*m,:n] = A
    Z[6*m:7*m,n:2*n] = 255*M 
    Z[6*m:7*m,2*n:3*n] = np.multiply(A,M)
 
    cv2.imshow(win1,Z)
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def rotate(I,mask,val=0):
    A = np.rot90(I,val) 
    M = np.rot90(mask,val) 
    return A,M 

def flip(I,mask,val=0):
    A = np.flip(I,axis=val)
    M = np.flip(mask,axis=val)
    return A,M


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


