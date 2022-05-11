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
print("using",device)

parser = argparse.ArgumentParser(description='Adds points to a dataset')
parser.add_argument('file_path', help='file_path')
parser.add_argument('-file_to', default="", help="destination file")
parser.add_argument('-inplace', action='store_true', help='inplace modifies file (default:False)')
parser.add_argument('-overwrite', action='store_true', help='overwrites destination file (default:False)')
parser.add_argument('-num_random_minmax',type=int,default=10, help='number of random frames to pool min max from (default:10)')

parser.add_argument('-s1',type=float,default=1., help='inner Gaussian sigma')
parser.add_argument('-s2',type=float,default=5., help='outer Gaussian sigma')
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
num_random_minmax=args.num_random_minmax
assert num_random_minmax>1

padding=(kernel_size//2 if inplace else 0)

arr1d=torch.arange(kernel_size)-kernel_size//2
xarr,yarr=torch.meshgrid(arr1d,arr1d,indexing="ij")
rsq=xarr**2+yarr**2

g1=np.exp(-rsq/2/(s1**2))/(2*np.pi*s1**2)
g2=np.exp(-rsq/2/(s2**2))/(2*np.pi*s2**2)
ker=(g1-g2).to(device=device,dtype=torch.float32)
ker/=ker.sum()
ker=ker.T#xy swap for pytorch convention, not that it matters

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

vol=data_info["W"]*data_info["H"]*data_info["D"]
t_minmaxs=np.random.choice(data_info["T"],min(data_info["T"],num_random_minmax),replace=False)
m,M=torch.full((data_info["C"],),np.inf,device=device),torch.full((data_info["C"],),-np.inf,device=device)
for t in t_minmaxs:
    frame=dataset.get_frame(t)
    frame=torch.tensor(frame,device=device).permute(3,0,1,2)#D,C,W,H
    frame=(frame/255).to(dtype=torch.float32)
    res=torch.nn.functional.conv2d(frame,ker,groups=2,padding=padding).permute(1,2,3,0)
    frmin=res.reshape(-1,np.prod(res.shape[1:])).min(dim=1)[0]
    frmax=res.reshape(-1,np.prod(res.shape[1:])).max(dim=1)[0]
    m=torch.minimum(m,frmin)
    M=torch.maximum(M,frmax)
#avoid saturation
m-=m*0.05
M+=M*0.05
sc=(M-m)[:,None,None,None]
m=m[:,None,None,None]


for t in tqdm(range(data_info["T"])):
    frame=dataset.get_frame(t)
    frame=torch.tensor(frame,device=device).permute(3,0,1,2)#D,C,W,H
    frame=(frame/255).to(dtype=torch.float32)
    res=torch.nn.functional.conv2d(frame,ker,groups=2,padding=padding).permute(1,2,3,0)
    res-=m
    res/=sc
    res=(torch.clip(res,0,1)*255).to(device="cpu",dtype=torch.uint8).numpy()
    if not inplace:
        new_dataset.set_frame(t,res,shape_change=True)
    else:
        dataset.set_frame(t,res,shape_change=False)

if not inplace:
    points=new_dataset.get_points()
    points[:,:,0]-=kernel_size//2
    points[:,:,1]-=kernel_size//2
    new_dataset.set_points(points)

dataset.close()
if not inplace:
    C,W,H,D=res.shape
    new_data_info={"W":W,"H":H}
    new_dataset.update_data_info(new_data_info)
    new_dataset.close()
