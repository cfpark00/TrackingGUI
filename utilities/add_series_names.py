import argparse

parser = argparse.ArgumentParser(description='Adds points to a dataset')
parser.add_argument('file_path', help='file_path')
parser.add_argument('names',type=str, nargs='+',help="names")

args=parser.parse_args()
file_path=args.file_path
names=args.names

import h5py
import numpy as np

h5=h5py.File(file_path,"r+")
if "series_names" in h5.attrs.keys():
    series_names=h5.attrs["series_names"]
    for name in names:
        if name not in series_names:
            series_names.append(name)
    h5.attrs["series_names"]=series_names
else:
    h5.attrs["series_names"]=names
h5.close()



