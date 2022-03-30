import os
import sys
sys.path.append(os.getcwd())
from src.Dataset import Dataset

import argparse
from tqdm import tqdm
import cv2
import numpy as np
import shutil

def apply_affine(im3d,almat):
    assert im3d.dtype==np.uint8, "only uint8 data"
    im3d=im3d.swapaxes(0,1)
    res=cv2.warpAffine(im3d, almat,(im3d.shape[1],im3d.shape[0])).swapaxes(0,1).astype(np.uint8)
    if len(res.shape)==2:
        return res[:,:,None]
    return res

parser = argparse.ArgumentParser(description='Adds points to a dataset')
parser.add_argument('file_path', help='file_path')
parser.add_argument('-almat_path', type=str,default="alignment.npy", help="path to alignment matrix")
parser.add_argument('-file_to', default="", help="destination file")

args=parser.parse_args()
file_path=args.file_path
fol,file_name=os.path.split(file_path)
name,suffix=file_name.split(".")
if args.file_to=="":
    file_to_path=os.path.join(fol,name+"_aligned"+"."+suffix)
else:
    file_to_path=args.file_to
almat_path=args.almat_path
almat=np.load(almat_path)
assert almat.shape[0]==2 and almat.shape[1]==2 and almat.shape[2]==3
ref_channel=np.nonzero(np.isnan(almat[:,0,0]))[0][0]
move_channel=1 if ref_channel==0 else 0
almat=almat[move_channel]

shutil.copyfile(file_path,file_to_path)

dataset=Dataset(file_to_path)
dataset.open()
data_info=dataset.get_data_info()

for i in tqdm(range(data_info["T"])):
    imal=dataset.get_frame(i)
    imal[move_channel]=apply_affine(imal[move_channel],almat)
    dataset.set_frame(i,imal,shape_change=False,compression="lzf")

print("Aligned data saved at",file_to_path)
dataset.close()
