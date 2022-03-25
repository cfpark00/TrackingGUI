import os
import sys
sys.path.append(os.getcwd())
from src.Dataset import Dataset

import argparse
from tqdm import tqdm
import cv2
import numpy as np

parser = argparse.ArgumentParser(description='Adds points to a dataset')
parser.add_argument('file_path', help='file_path')
parser.add_argument('-almat_path', type=str,default="almat_temp.npy", help="path to alignment matrix")
parser.add_argument('-file_to', default="", help="destination file")

args=parser.parse_args()
file_path=args.file_path
fol,file_name=os.path.split(file_path)
name,suffix=file_name.split(".")
if args.file_to=="":
    file_to_path=os.path.join(fol,name+"_dog"+"."+suffix)
else:
    file_to_path=args.file_to
almat_path=args.almat_path
almat=np.load(almat_path)
assert almat.shape[0]==2 and almat.shape[1]==2 and almat.shape[2]==3
ref_channel=np.nonzero(np.isnan(almat[:,0,0]))[0][0]
print("ref channel",ref_channel)
sys.exit()

dataset=Dataset(file_path)
dataset.open()
data_info=dataset.get_data_info()
ptsfrom=[]
ptsto=[]
while True:
    comm=input("Alignment[0,"+str(data_info["T"])+"]:")
    try:
        comm=int(comm)
    except:
        pass
    if type(comm)==int:
        if not (0<=comm<data_info["T"]):
            comm=np.random.randint(0,data_info["T"])
        print("Opening: "+str(comm))
        im=dataset.get_frame(comm).max(3)
        app=QApplication(sys.argv)
        diag = GetPtsWidget.GetPtsWidget("Get Points: "+str(comm+1)+"/"+str(data_info["T"]),im[1 if ref_channel==0 else 0],im[ref_channel],ptsfrom,ptsto)
        app.exec_()
        del app
        print("Current number of points: ",len(ptsfrom))
        continue
    if comm=="Done":
        break
    else:
        print("Alignment[0,"+str(data_info["T"])+"]:","Enter a number(-1 for random) or \"Done\"")
        continue
if len(ptsfrom)!=len(ptsto) or len(ptsto)<3:
    print("Aborting: at least 3 points needed.")
    dataset.close()
almat=get_opt_align(np.array(ptsfrom),np.array(ptsto))
dataset.close()
sys.exit()
"""
            ##Get affine transform
            almat=get_opt_align(np.array(ptsfrom),np.array(ptsto))
    else:
        almat=np.array([[1,0,0],[0,1,0]])

    if almat is None:
        print("Alignment Extraction Failed")
        h5DoG.close()
        return
    else:
        print("Alignment Extraction Successful")
    almat=almat.astype(np.float32)

    ##Check if Affine transform is good
    if not skip_validation:
        result=[False]
        app=QApplication(sys.argv)
        num=6
        shows=np.random.choice(h5DoG.attrs["T"],num,replace=True)
        ims={}
        for i in shows:
            imal=np.array(h5DoG[str(i)+"/frame"])
            imal[0]=apply_affine(imal[0],almat).astype(np.int16)
            ims[str(i)+"/"+str(h5DoG.attrs["T"])]=np.max(np.moveaxis(np.array([imal[0],imal[1],imal[0]]),0,3),axis=2)
        confirm=GetPtsWidget.ConfirmAlWidget("Confirm Alignment",ims,result)
        app.exec_()
        del app
        if not result[0]:
            print("Alignment Aborted")
            h5DoG.close()
            return

    #Get number of neurons
    if N_neurons is None:
        while True:
            N_neurons=input("Number of Neurons:")
            try:
                N_neurons=int(N_neurons)
                if not (0<N_neurons<400):
                    assert False
                break
            except:
                pass

    #apply affine transform
    h5Al=h5py.File(h5Alfn,"w")
    for key,val in h5DoG.attrs.items():
        h5Al.attrs[key]=val
    h5Al.attrs["N_neurons"]=N_neurons
    h5Al.attrs["almat"]=almat.astype(np.float32)

    if cuts is None:
        sh=(h5Al.attrs["C"],h5Al.attrs["W"],h5Al.attrs["H"],h5Al.attrs["D"])
        pads=get_valid_pad(sh[1:],minshape)
        sh=(sh[0],sh[1]+pads[0],sh[2]+pads[1],sh[3]+pads[2])
        pads=[(0,0),(0,pads[0]),(0,pads[1]),(0,pads[2])]
    else:
        sh=(h5Al.attrs["C"],cuts[0][1]-cuts[0][0],cuts[1][1]-cuts[1][0],cuts[2][1]-cuts[2][0])
        pads=get_valid_pad(sh[1:],minshape)
        sh=(sh[0],sh[1]+pads[0],sh[2]+pads[1],sh[3]+pads[2])
        pads=[(0,0),(0,pads[0]),(0,pads[1]),(0,pads[2])]
    h5Al.attrs["W"]=sh[1]
    h5Al.attrs["H"]=sh[2]
    h5Al.attrs["D"]=sh[3]
    means=np.zeros((h5Al.attrs["T"],2),dtype=np.float32)
    stds=np.zeros((h5Al.attrs["T"],2),dtype=np.float32)
    for i in range(h5Al.attrs["T"]):
        print("\r\t"+str(i+1)+"/"+str(h5Al.attrs["T"]),end="")
        dset=h5Al.create_dataset(str(i)+"/frame",sh,dtype="i2",compression="lzf")
        imal=np.array(h5DoG[str(i)+"/frame"])
        imal[0]=apply_affine(imal[0],almat).astype(np.int16)
        if cuts is not None:
            imal=imal[:,cuts[0][0]:cuts[0][1],cuts[1][0]:cuts[1][1],cuts[2][0]:cuts[2][1]]
        imal=np.pad(imal,pads)
        dset[...]=imal
        means[i]=np.mean(imal,axis=(1,2,3))
        stds[i]=np.std(imal,axis=(1,2,3))
    dset=h5Al.create_dataset("means",means.shape,dtype="f4")
    dset[...]=means
    dset=h5Al.create_dataset("stds",stds.shape,dtype="f4")
    dset[...]=stds
    h5Al.close()
    h5DoG.close()
"""
