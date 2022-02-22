import argparse

parser = argparse.ArgumentParser(description='Adds points to a dataset')
parser.add_argument('file_path', help='file_path')

args=parser.parse_args()
file_path=args.file_path

import h5py
h5=h5py.File(file_path,"r+")
N_points=h5.attrs["N_neurons"]
h5.attrs["N_points"]=N_points
del h5.attrs["N_neurons"]

T=h5.attrs["T"]
if "pointdat" in h5.keys():
    points=np.array(h5["pointdat"]).astype(np.float32)
    del h5["pointdat"]
else:
    points=np.full((T,N_points+1,3),np.nan)
ds=h5.create_dataset("points",shape=points.shape,dtype=points.dtype)
ds[...]=points

h5.close()



