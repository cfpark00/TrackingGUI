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

parser = argparse.ArgumentParser(description='Adds points to a dataset')
parser.add_argument('file_path', help='file_path')
parser.add_argument('xi',type=int,default=0, help='start x(from 0)')
parser.add_argument('xf',type=int,default=-1, help='end x(not inclusive)')
parser.add_argument('yi',type=int,default=0, help='start y(from 0)')
parser.add_argument('yf',type=int,default=-1, help='end y(not inclusive)')
parser.add_argument('zi',type=int,default=0, help='start z(from 0)')
parser.add_argument('zf',type=int,default=-1, help='end z(not inclusive)')
parser.add_argument('-file_to', default="", help="destination file")
parser.add_argument('-inplace', action='store_true', help='inplace operation (default:False)')
parser.add_argument('-overwrite', action='store_true', help='overwrite new file (default:False)')

args=parser.parse_args()
file_path=args.file_path
fol,file_name=os.path.split(file_path)
name,suffix=file_name.split(".")
assert suffix=="h5","This only works for h5 files"
if args.file_to=="":
    file_to_path=os.path.join(fol,name+"_xyzsliced"+"."+suffix)
else:
    file_to_path=args.file_to
xi=args.xi
xf=args.xf
yi=args.yi
yf=args.yf
zi=args.zi
zf=args.zf
inplace=args.inplace
overwrite=args.overwrite

dataset=Dataset(file_path)
dataset.open()
data_info=dataset.get_data_info()
if xf==-1:
    xf=data_info["W"]
newW=xf-xi
if yf==-1:
    yf=data_info["H"]
newH=yf-yi
if zf==-1:
    zf=data_info["D"]
newD=zf-zi
assert xi<xf and 0<=xi and xf<=data_info["W"]
assert yi<yf and 0<=yi and yf<=data_info["H"]
assert zi<zf and 0<=zi and zf<=data_info["D"]

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

for t in tqdm(range(data_info["T"])):
    frame=dataset.get_frame(t)
    new_dataset.set_frame(t,frame[:,xi:xf,yi:yf,zi:zf],shape_change=True)

helper_names=dataset.get_helper_names()
for name in helper_names:
    helper=dataset.get_helper(name)
    helper-=np.array([xi,yi,zi])[None,None,:]
    if not inplace:
        new_dataset.set_helper(name,helper)
    else:
        dataset.set_helper(name,helper)

points=dataset.get_points()
points-=np.array([xi,yi,zi])[None,None,:]
if not inplace:
    new_dataset.set_points(points)
else:
    dataset.set_points(points)

new_data_info={"W":newW,"H":newH,"D":newD}
if not inplace:
    new_dataset.update_data_info(new_data_info)
    new_dataset.close()
    dataset.close()
else:
    dataset.update_data_info(new_data_info)
    dataset.close()
