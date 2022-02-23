import argparse

parser = argparse.ArgumentParser(description='Adds points to a dataset')
parser.add_argument('file_path', help='file_path')
parser.add_argument('names',type=str, nargs='+',help="names")

args=parser.parse_args()
file_path=args.file_path
names=args.names
n=len(names)
assert n%2==0,"pair names and labels"

import h5py
import numpy as np

h5=h5py.File(file_path,"r+")
series_names=[]
series_labels=[]
for i in range(n//2):
    series_names.append(names[2*i])
    series_labels.append(names[2*i+1])
h5.attrs["series_names"]=series_names
h5.attrs["series_labels"]=series_labels
h5.close()



