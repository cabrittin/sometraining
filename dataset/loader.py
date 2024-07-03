"""
@name: dataset.loader.py                       
@description:                  
Module for loading tiff stacks

@author: Christopher Brittin   
@email: "cabrittin"+ <at>+ "gmail"+ "."+ "com"
@date: 2019-12-05              
"""

from configparser import ConfigParser,ExtendedInterpolation
import os
import glob
import re
import numpy as np
import cv2

from tifffile import TiffFile

from pycsvparser.read import parse_file

class Session(object):
    """
    Assumes all stage positions have same number of stacks
    """
    def __init__(self,ini):
        self.cfg = ConfigParser(interpolation=ExtendedInterpolation())
        self.cfg.read(ini)
        self.positions = self.cfg['structure']['positions'].split(',')
        self.num_positions = len(positions)
        self.stacks = [[] for p in self.positions]
        for (idx,p) in enumerate(self.positions):
            img_dir = self.cfg[p]['images']
            glob_path = format_glob_path(img_dir,self.cfg['structure']['extension']) 
            self.stacks[idx] = stacks_from_path(glob_path)
    
    def __len__(self):
        return len(self.stacks[0])
    
    def get_position_stack(self,pdx,idx):
        return self.stacks[pdx][idx]
    
    def iter_position_stacks(self,pdx):
        for s in self.stacks[pdx]:
            yield s
    
    def iter_stacks(self):
        for i in range(self.__len__()):
            yield [p[i] for p in self.stacks]

    def chunk_position_stacks(self,pdx,nchunks):
        return split_n(self.stacks[pdx],nchunks)

def split_n(sequence, num_chunks):
    chunk_size, remaining = divmod(len(sequence), num_chunks)
    for i in range(num_chunks):
        begin = i * chunk_size + min(i, remaining)
        end = (i + 1) * chunk_size + min(i + 1, remaining)
        yield sequence[begin:end]



def rois_from_file(fname):
    @parse_file(fname,multi_dim=True)
    def row_into_container(container,row=None,**kwargs):
        roi = [(int(row[0]),int(row[1])),(int(row[2]),int(row[3]))]
        container.append(roi)
    
    container = []
    row_into_container(container)
    return container


def tif_from_stack(fname):
    return TiffFile(fname)

def image_to_rgb(image):
    ndims = len(image.shape)
    if ndims < 3:
        image = cv2.cvtColor(image,cv2.COLOR_GRAY2RGB)
    return image

def image_to_gray(image):
    ndims = len(image.shape)
    if ndims > 2:
        image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    return image


def array_from_stack(fname,zdx=0):
    return tif_from_stack(fname).pages[zdx].asarray()

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

def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    '''
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)
    '''
    return [ atoi(c) for c in re.split(r'(\d+)', text) ]

def format_path(L):
    ext = L.cfg['extension']
    path = os.sep.join([L.dname,'*'+ext])
    return path

def format_glob_path(directory,extension):
    return os.sep.join([directory,'*'+extension])
    
def stacks_from_path(path,order_stacks=True):
    stacks = glob.glob(path)
    if order_stacks: stacks.sort(key=natural_keys)
    return stacks

