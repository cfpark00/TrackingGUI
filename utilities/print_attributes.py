import argparse

parser = argparse.ArgumentParser(description='Adds points to a dataset')
parser.add_argument('file_path', help='file_path')
args=parser.parse_args()

file_path=args.file_path

import h5py


h5=h5py.File(file_path,"r")
print(h5.keys())
print(h5.attrs.keys())

h5.close()
