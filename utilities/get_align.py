import os
import sys
sys.path.append(os.getcwd())
from src.Dataset import Dataset
from src.tools import GetPtsWidget
import scipy.optimize as sopt
from PyQt5.QtWidgets import *

import argparse
import numpy as np
import cv2

def get_opt_align(ptsfrom,ptsto):
    ptsfrom=ptsfrom.astype(np.float32)
    ptsto=ptsto.astype(np.float32)
    cmfrom=np.mean(ptsfrom,axis=0)
    cmto=np.mean(ptsto,axis=0)
    almat=np.array([[1,0,cmto[0]-cmfrom[0]],[0,1,cmto[1]-cmfrom[1]]])
    def energy(almat,ptsfrom,ptsto):
        almat=almat.reshape(2,3)
        return np.sum(np.square(np.matmul(almat[:,:2],ptsfrom.T).T+almat[:,2][None,:]-ptsto))
    res=sopt.minimize(energy,almat.flatten(),args=(ptsfrom,ptsto),method = 'SLSQP')
    if not res.success:
        return None
    almat=res.x.reshape(2,3)
    return almat

def apply_affine(im3d,almat):
    assert im3d.dtype==np.uint8, "only uint8 data"
    im3d=im3d.swapaxes(0,1)
    res=cv2.warpAffine(im3d, almat,(im3d.shape[1],im3d.shape[0])).swapaxes(0,1).astype(np.uint8)
    if len(res.shape)==2:
        return res[:,:,None]
    return res

parser = argparse.ArgumentParser(description='Adds points to a dataset')
parser.add_argument('file_path', help='file_path')
parser.add_argument('-save_path', type=str,default="almat_temp.npy", help="where to save almat")
parser.add_argument('-ref_channel', type=int,default=0, help="channel to not move")

args=parser.parse_args()
file_path=args.file_path
save_path=args.save_path
ref_channel=args.ref_channel
move_channel=1 if ref_channel==0 else 0

dataset=Dataset(file_path)
dataset.open()
data_info=dataset.get_data_info()
assert data_info["C"]==2, "only 2 channel supported"
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
        diag = GetPtsWidget.GetPtsWidget("Get Points: "+str(comm+1)+"/"+str(data_info["T"]),im[move_channel],im[ref_channel],ptsfrom,ptsto)
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
    sys.exit()
almat=get_opt_align(np.array(ptsfrom),np.array(ptsto))


result=[False]
app=QApplication(sys.argv)
num=6
shows=np.random.choice(data_info["T"],num,replace=True)
ims={}
for i in shows:
    imal=dataset.get_frame(i)
    print(move_channel)
    print(imal[move_channel].shape,apply_affine(imal[move_channel],almat).shape)
    imal[move_channel]=apply_affine(imal[move_channel],almat)
    ims[str(i)+"/"+str(data_info["T"])]=np.max(np.moveaxis(np.array([imal[move_channel],imal[ref_channel],imal[move_channel]]),0,3),axis=2)
confirm=GetPtsWidget.ConfirmAlWidget("Confirm Alignment",ims,result)
app.exec_()
del app
if not result[0]:
    print("Alignment Aborted")
    dataset.close()
    sys.exit()

almat_nan=np.full_like(almat,np.nan)
if ref_channel==0:
    almat=np.stack([almat_nan,almat],axis=0)
else:
    almat=np.stack([almat,almat_nan],axis=0)
np.save(save_path,almat)
dataset.close()
