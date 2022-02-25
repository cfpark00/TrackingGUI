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
parser.add_argument('ti',type=int,default=1, help='initial time')
parser.add_argument('tf',type=int,default=0, help='final time')
parser.add_argument('-file_to', default="", help="destination file")
parser.add_argument('-overwrite', action='store_true', help='overwrites file (default:False)')


args=parser.parse_args()
file_path=args.file_path
fol,file_name=os.path.split(file_path)
name,suffix=file_name.split(".")
assert suffix=="h5","This only works for h5 files"
if args.file_to=="":
    file_to_path=os.path.join(fol,name+"_timesliced"+"."+suffix)
else:
    file_to_path=args.file_to
ti=args.ti
tf=args.tf
overwrite=args.overwrite

dataset=Dataset(file_path)
dataset.open()
data_info=dataset.get_data_info()
if tf==0:
    tf=data_info["T"]
assert ti<=tf and 0<ti and tf<=data_info["T"]

if not overwrite:
    if os.path.exists(file_to_path):
        print(file_to_path,"already present")
        sys.exit()
    shutil.copyfile(file_path,file_to_path)
    new_dataset=Dataset(file_to_path)
    new_dataset.open()

for t in range(ti,tf+1):
    if overwrite:
        frame=dataset.get_frame(t-1)
        dataset.set_frame(t-ti,frame,shape_change=False)
    else:
        frame=dataset.get_frame(t-1)
        new_dataset.set_frame(t-ti,frame,shape_change=False)
for t in range(tf+1,data_info["T"]+1):
    if overwrite:
        dataset.remove(str(t-1)+"/frame")

T=tf-ti+1
new_data_info={"T":T}
if not overwrite:
    new_dataset.update_data_info(new_data_info)
    new_dataset.close()
    dataset.close()
else:
    dataset.update_data_info(new_data_info)
    dataset.close()