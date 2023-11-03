"""
@name:                         
@description:                  


@author: Christopher Brittin   
@email: "cabrittin"+ <at>+ "gmail"+ "."+ "com"
@date: 2019-12-05              
"""

import sys
import os
from configparser import ConfigParser,ExtendedInterpolation
import argparse
from inspect import getmembers,isfunction
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm

from tifffile import TiffFile

from pycsvparser import read


def _view_image_pair(func):
    def inner(args): 
        cfg = ConfigParser(interpolation=ExtendedInterpolation())
        cfg.read(args.ini)
        
        low,high = func(cfg)
        idx = 1000
        
        L = np.load(low%idx)
        H = np.load(high%idx)
        
        L = L / L.max()
        H = H / H.max()

        Z = np.concatenate((L,H),axis=1)

        fig,ax = plt.subplots(1,1,figsize=(10,10))
        ax.imshow(Z,cmap='gray')
        ax.set_xticks([])
        ax.set_yticks([])
        plt.show()
    
    return inner

@_view_image_pair
def view_train_pair(cfg):
    low = os.path.join(cfg['structure']['train_x'],'patch_%d.npy') 
    high = os.path.join(cfg['structure']['train_y'],'patch_%d.npy') 
    return low,high

@_view_image_pair
def view_test_pair(cfg):
    low = os.path.join(cfg['structure']['test_x'],'patch_%d.npy') 
    high = os.path.join(cfg['structure']['test_y'],'patch_%d.npy') 
    return low,high


def build(args):
    """ 
    Assuming that even and odd tiffpages are the low and high states, respectively
    """
    cfg = ConfigParser(interpolation=ExtendedInterpolation())
    cfg.read(args.ini)
    
    width = cfg.getint('image','width')
    dx = cfg.getint('build','patch_size')
    n_patches = width // dx

    pos = [os.path.join(cfg['structure']['data'],p) for p in read.into_list(cfg['files']['position_list'])]
    
    num_test = int(len(pos) * cfg.getfloat('build','train_test_split'))

    to_test = np.zeros(len(pos),dtype=np.uint8)
    to_test[:num_test] = 1
    np.random.shuffle(to_test)
    
    idx = [0,0]
    for (stack_idx,stack) in enumerate(pos):
        T = TiffFile(stack)
        
        is_test = to_test[stack_idx]
        dlabel = ['train','test'][is_test]
        low_out = os.path.join(cfg['structure'][f'{dlabel}_x'],'patch_%d.npy') 
        high_out = os.path.join(cfg['structure'][f'{dlabel}_y'],'patch_%d.npy') 
     
        n_pages = len(T.pages)
        for low_pdx in tqdm(range(0,n_pages,2),desc=f"Position {stack_idx}"):
            high_pdx = low_pdx + 1
        
            low_page = T.pages[low_pdx].asarray().astype(np.uint16)
            high_page = T.pages[high_pdx].asarray().astype(np.uint16)
            
            row0,col0 = 0,0
            for i in range(n_patches):
                col1 = col0 + dx
                for j in range(n_patches):
                    row1 = row0 + dx
                    Lp = low_page[row0:row1,col0:col1]
                    Hp = high_page[row0:row1,col0:col1]

                    np.save(low_out%idx[is_test],Lp)
                    np.save(high_out%idx[is_test],Hp)
                    
                    idx[is_test] += 1

                col0 = col1
                row0 = row1
            

if __name__=="__main__":
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument('mode',
                        action = 'store',
                        choices = [t for (t,o) in getmembers(sys.modules[__name__]) if isfunction(o)],
                        help = 'Function call')

    parser.add_argument('ini',
                action = 'store',
                help = 'Path to .ini file')

    params = parser.parse_args()
    eval(params.mode + '(params)')

