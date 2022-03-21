import h5py as h5
import numpy as np
#insert the metadata here
N_points,T,C,W,H,D=20,150,3,256,160,16
h5=h5.File("data/example.h5","w")
for i in range(T):
    #here is your uint8 image
    image=np.random.choice(256,(C,W,H,D),replace=True).astype(np.uint8)
    ds=h5.create_dataset(str(i)+"/frame",shape=image.shape,dtype=np.uint8,compression="lzf")
    ds[...]=image
h5.attrs["N_points"]=N_points
#h5.create_dataset("points",shape=(T,N_points+1,3,))
#missing
#annotatedi
#missing_index=
#ds[...]=np.random.choice(T256,(C,W,H,D),replace=True).astype(np.uint8)
h5.attrs["T"]=T
h5.attrs["C"]=C
h5.attrs["W"]=W
h5.attrs["H"]=H
h5.attrs["D"]=D
h5.close()
