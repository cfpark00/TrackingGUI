import os
import sys
sys.path.append(os.getcwd())
from src.Dataset import Dataset
from utilities import h5utils

import argparse
from tqdm import tqdm
import torch
import shutil
import numpy as np
import scipy.ndimage as sim
import cv2

parser = argparse.ArgumentParser(description='Adds points to a dataset')
parser.add_argument('file_path', help='file_path')
parser.add_argument('-channel',type=int, default=0,help='channel to compute cm from')
parser.add_argument('-thres',type=int, default=0,help='channel threshold')
parser.add_argument('-rotate',action='store_true',help='rotate moment')
parser.add_argument('-file_to', default="", help="destination file")
parser.add_argument('-inplace', action='store_true', help='inplace operation (default:False)')
parser.add_argument('-overwrite', action='store_true', help='overwrite new file (default:False)')

args=parser.parse_args()
file_path=args.file_path
fol,file_name=os.path.split(file_path)
name,suffix=file_name.split(".")
assert suffix=="h5","This only works for h5 files"
if args.file_to=="":
    file_to_path=os.path.join(fol,name+"_centered"+"."+suffix)
else:
    file_to_path=args.file_to
channel=args.channel
thres=args.thres
rotate=args.rotate
inplace=args.inplace
overwrite=args.overwrite


dataset=Dataset(file_path)
dataset.open()
data_info=dataset.get_data_info()

if not inplace:
    if os.path.exists(file_to_path):
        if overwrite:
            os.remove(file_to_path)
        else:
            print(file_to_path,"already present")
            sys.exit()
    shutil.copyfile(file_path,file_to_path)
    new_dataset=Dataset(file_to_path)
    new_dataset.open()
else:
    new_dataset=dataset

def apply_affine(im3d,almat):
    assert im3d.dtype==np.uint8, "only uint8 data"
    im3d=im3d.swapaxes(0,1)
    res=cv2.warpAffine(im3d, almat,(im3d.shape[1],im3d.shape[0])).swapaxes(0,1).astype(np.uint8)
    if len(res.shape)==2:
        return res[:,:,None]
    return res

center=np.array([data_info["W"],data_info["H"]]).astype(np.float32)/2-0.5
almat=np.array([[1,0,0],[0,1,0]]).astype(np.float32)

for t in tqdm(range(data_info["T"])):
    frame=dataset.get_frame(t)
    frcm=(frame[channel].max(2)>thres).astype(np.uint8)
    moments=cv2.moments(frcm.T)
    m=moments['m00']
    if m==0:
        new_dataset.set_frame(t,frame)
        continue
    if not rotate:
        cm=np.array([moments['m10']/m,moments['m01']/m])
        move=center-cm
        almat[:,2]=move
        frame[0]=apply_affine(frame[0],almat)
        frame[1]=apply_affine(frame[1],almat)
        new_dataset.set_frame(t,frame)
    else:
        cm=np.array([moments['m10']/m,moments['m01']/m])
        mat=np.array([[moments['nu20'],moments['nu11']],[moments['nu11'],moments['nu02']]])
        eigvals,eigvecs=np.linalg.eigh(mat)
        vec=eigvecs[:,np.argmax(eigvals)]
        theta_=-np.arctan2(vec[1],vec[0])
        rotmat=np.array([np.cos(theta_),-np.sin(theta_),np.sin(theta_),np.cos(theta_)]).reshape(2,2)
        almat[:,2]=-cm@rotmat.T+center
        almat[:,:2]=rotmat
        frame[0]=apply_affine(frame[0],almat)
        frame[1]=apply_affine(frame[1],almat)
        new_dataset.set_frame(t,frame)

if not inplace:
    new_dataset.close()
    dataset.close()
else:
    dataset.close()
