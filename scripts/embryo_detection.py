"""
@name: embryo_detection.py
@description:                  

@author: Christopher Brittin   
@email: "cabrittin"+ <at>+ "gmail"+ "."+ "com"
@date: 

ROIs format [xmin,ymin,xmax,ymax] in numpy array

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

def add_roi_to_image(I,rois):
    I = cv2.cvtColor(I,cv2.COLOR_GRAY2RGB)
    for [r0,r1,r2,r3] in rois: cv2.rectangle(I,[r0,r1],[r2,r3],(255,0,0),2)
    return I

def build_mask_image(I,rois,masks):
    Z = np.zeros(I.shape,dtype=np.uint8)
    for (i,[r0,r1,r2,r3]) in enumerate(rois): 
        Z[r1:r3,r0:r2] = (i+1)*masks[i,:]
    #Z = cv2.cvtColor(Z,cv2.COLOR_GRAY2RGB)
    return Z

def load_data(dmap,idx):
    I = np.load(dmap[idx][0])
    I = array_16bit_to_8bit(I)
    return I,np.load(dmap[idx][1]),np.load(dmap[idx][2])

def sort_rois(_rois):
    rois = []
    for [x0,y0,x1,y1] in _rois:
        rois.append([min(x0,x1),min(y0,y1),max(x0,x1),max(y0,y1)])
    return rois

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
    for i in tqdm(range(cfg['num_sample']),desc="Samples generated"):
        #mdx = np.random.randint(low=0,high=M)
        
        r = []
        while len(r) == 0:
            #A,r,msk = augment(I[mdx],rois[mdx])
            A,r,msk = augment(I,rois,masks)
        
        r = sort_rois(r) 
        r = np.array(r).astype(np.uint16)
        msk = build_mask_image(A,rois,masks)  
        if np.random.rand() < tt_split:
            np.save(test_x%test_idx,A)
            np.savez(test_y%test_idx,roi=r,mask=msk)
            test_idx += 1
        else:
            np.save(train_x%train_idx,A)
            np.savez(train_y%train_idx,roi=r,mask=msk)
            train_idx += 1

def view_augmentation(args):
    fin = open(args.json)
    cfg = json.load(fin)
    dmap = load_data_map(cfg) 
    
    win1 = 'Augmentation-1'
    cv2.namedWindow(win1)
    cv2.moveWindow(win1,300,100)
    
    win2 = 'Augmentation-2'
    cv2.namedWindow(win2)
    cv2.moveWindow(win2,1500,100)
    
    I,rois,masks = load_data(dmap,0)

    m,n = I.shape
    #Z = np.zeros((2*m,3*n,3),dtype=np.uint8)
    Z = np.zeros((4*m,2*n,3),dtype=np.uint8)
    
    I0 = cv2.cvtColor(I,cv2.COLOR_GRAY2RGB)
    Z[:m,:n] = add_roi_to_image(I,rois)
    #Z[:m,:n] = I0
    Z[:m,n:2*n] = build_mask_image(I,rois,masks) 
    
    #Flip Left/Right
    A,r,msk = flip_lr(I,rois,masks)
    Z[m:2*m,:n] = add_roi_to_image(A,r)
    Z[m:2*m,n:2*n] = build_mask_image(I,r,msk) 
    
    #Flip Left/Right
    A,r,msk = flip_ud(I,rois,masks)
    Z[2*m:3*m,:n] = add_roi_to_image(A,r)
    Z[2*m:3*m,n:2*n] = build_mask_image(I,r,msk) 
    
    #Blank 
    A,r,msk = random_blank_rois(I,rois,masks,p_blank=0.5)
    Z[3*m:4*m,:n] = add_roi_to_image(A,r)
    Z[3*m:4*m,n:2*n] = build_mask_image(I,r,msk) 
    
    ZZ = np.zeros((3*m,2*n,3),dtype=np.uint8)
    
    #Rotate
    A,r,msk = random_rotate_rois(I,rois,masks)
    ZZ[:m,:n] = add_roi_to_image(A,r)
    ZZ[:m,n:2*n] = build_mask_image(I,r,msk) 
     
    #Random swap rois
    A,r,msk = random_swap_rois(I,rois,masks)
    ZZ[m:2*m,:n] = add_roi_to_image(A,r)
    ZZ[m:2*m,n:2*n] = build_mask_image(I,r,msk) 
    
    #Random augment
    A,r,msk = augment(I,rois,masks)
    ZZ[2*m:3*m,:n] = add_roi_to_image(A,r)
    ZZ[2*m:3*m,n:2*n] = build_mask_image(I,r,msk)
    
    Z = cv2.resize(Z, (0,0), fx=0.35, fy=0.35)
    cv2.imshow(win1,Z)
    
    ZZ = cv2.resize(ZZ, (0,0), fx=0.35, fy=0.35)
    cv2.imshow(win2,ZZ)
 
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def augment(I,rois,masks):
    p_blank = np.random.rand()
    A,r,msk = random_blank_rois(I,rois,masks,p_blank=p_blank)
    A,r,msk = random_swap_rois(A,r,msk)
    A,r,msk = random_rotate_rois(A,r,msk)
    if np.random.rand() < 0.5: 
        A,r,msk = flip_lr(A,r,msk)
    if np.random.rand() < 0.5:
        A,r,msk = flip_ud(A,r,msk)
    
    return A,r,msk

def flip_lr(I,_rois,_masks):
    m,n = I.shape
    A = np.flip(I,axis=1)
    rois = []
    masks = np.zeros(_masks.shape)
    for (i,r) in enumerate(_rois):
        tmp = [n-r[0],r[1],n-r[2],r[3]]
        rois.append(tmp)
        masks[i,:,:] = np.flip(_masks[i,:,:],axis=1)
    rois = sort_rois(rois) 
    return A,rois,masks

def flip_ud(I,_rois,_masks):
    m,n = I.shape
    A = np.flip(I,axis=0)
    rois = [] 
    masks = np.zeros(_masks.shape)
    for (i,r) in enumerate(_rois):
        tmp = [r[0],m-r[1],r[2],m-r[3]]
        rois.append(tmp)
        masks[i,:,:] = np.flip(_masks[i,:,:],axis=0)
    rois = sort_rois(rois) 
    return A,rois,masks

def random_blank_rois(I,_rois,_masks,p_blank=0.5):
    A = np.copy(I)
    r = _rois[0]
    dx = abs(r[3] - r[1])
    dy = abs(r[2] - r[0])
    m,n = I.shape

    B = I[:dy,n-dx:]
    
    rois = []
    masks = []
    make_blank = np.random.rand(len(_rois)) < p_blank
    for (i,r) in enumerate(_rois):
        if make_blank[i]: 
            A[r[1]:r[3],r[0]:r[2]] = B
        else: 
            rois.append(r)
            masks.append(_masks[i,:,:])
    return A,rois,np.array(masks)

def random_rotate_rois(I,_rois,_masks):
    A = np.copy(I)
    rot_val = np.random.randint(size=len(_rois),low=0,high=4)
    masks = np.zeros(_masks.shape) 
    for (i,r) in enumerate(_rois):
        a = A[r[1]:r[3],r[0]:r[2]] 
        a = np.rot90(a,rot_val[i]) 
        A[r[1]:r[3],r[0]:r[2]] = a
        masks[i,:,:] = np.rot90(_masks[i,:,:],rot_val[i]) 
    return A,_rois,masks 

def random_swap_rois(I,_rois,_masks):
    A = np.copy(I)
    B = np.copy(I)
    pdx = np.arange(len(_rois))
    np.random.shuffle(pdx)
    masks = np.zeros(_masks.shape) 
    for i in range(len(_rois)):
        r = _rois[pdx[i]]
        a = A[r[1]:r[3],r[0]:r[2]] 
        masks[i,:,:] = _masks[pdx[i],:,:] 
        r = _rois[i]
        B[r[1]:r[3],r[0]:r[2]] = a
    return B,_rois,masks 

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


