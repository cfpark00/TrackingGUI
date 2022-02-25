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
device="cuda" if torch.cuda.is_available() else "cpu"

parser = argparse.ArgumentParser(description='Adds points to a dataset')
parser.add_argument('file_path', help='file_path')
parser.add_argument('-file_to', default="", help="destination file")
parser.add_argument('-inplace', action='store_true', help='inplace modifies file (default:False)')
parser.add_argument('-overwrite', action='store_true', help='overwrites destination file (default:False)')

parser.add_argument('-s1',type=int,default=0.5, help='inner Gaussian sigma')
parser.add_argument('-s2',type=int,default=3, help='outer Gaussian sigma')
parser.add_argument('-kernel_size',default=21,type=int, help='kernel_size in pixels(must be odd)')

args=parser.parse_args()
file_path=args.file_path
fol,file_name=os.path.split(file_path)
name,suffix=file_name.split(".")
if args.file_to=="":
    file_to_path=os.path.join(fol,name+"_dog"+"."+suffix)
else:
    file_to_path=args.file_to
kernel_size=args.kernel_size
assert kernel_size%2==1,"kernel_size must be odd"
s1=args.s1
s2=args.s2
inplace=args.inplace
overwrite=args.overwrite

padding=(kernel_size//2 if inplace else 0)

arr1d=torch.arange(kernel_size)-kernel_size//2
xarr,yarr=torch.meshgrid(arr1d,arr1d,indexing="ij")
rsq=xarr**2+yarr**2

g1=np.exp(-rsq/2/(s1**2))/(2*np.pi*s1**2)
g2=np.exp(-rsq/2/(s2**2))/(2*np.pi*s2**2)
ker=(g1-g2).to(device=device,dtype=torch.float32)
ker/=ker.sum()
ker=ker.T#xy swap for pytorch convention

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
dataset=Dataset(file_path)
dataset.open()

data_info=dataset.get_data_info()
ker=ker[None].repeat(data_info["C"],1,1,1)
for t in tqdm(range(1,data_info["T"]+1)):
    frame=dataset.get_frame(t-1)
    frame=torch.tensor(frame,device=device).permute(3,0,1,2)#D,C,W,H
    frame=(frame/255).to(dtype=torch.float32)
    res=torch.nn.functional.conv2d(frame,ker,groups=2,padding=padding).permute(1,2,3,0)
    res=(torch.clip(res,0,1)*255).to(device="cpu",dtype=torch.uint8).numpy()
    if not inplace:
        new_dataset.set_frame(t-1,res,shape_change=True)
    else:
        dataset.set_frame(t-1,res,shape_change=False)
dataset.close()
if not inplace:
    C,W,H,D=res.shape    
    new_data_info={"W":W,"H":H}
    new_dataset.update_data_info(new_data_info)
    new_dataset.close()
