import torch
import numpy as np
import scipy.spatial as sspat
import albumentations as A
from torch.utils.data import Dataset
import os
import glob

class TrainDataset(Dataset):
    def __init__(self,folpath,channels,shape):
        super().__init__()
        self.folpath=folpath
        self.channels=channels
        self.shape=shape
        self.framefolpath=os.path.join(self.folpath,"frames")
        self.maskfolpath=os.path.join(self.folpath,"masks")
        self.filelist=glob.glob(os.path.join(self.maskfolpath,"*"))
        self.indlist=[int(os.path.split(file)[1].split(".")[0]) for file in self.filelist]
        self.indlist=np.array(self.indlist)
        self.num_frames_tot=len(self.indlist)
        
        inf={}
        inf["mask"]="mask"

        self.comp=A.Compose([
            A.ShiftScaleRotate(shift_limit=0.12,scale_limit=0.12,rotate_limit=30,interpolation=1,border_mode=0,value=0,mask_value=0,p=1),
            A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.5),
            A.OpticalDistortion(border_mode=0,p=0.5),
            A.MotionBlur(blur_limit=8,p=0.5),
            ],
            additional_targets=inf
            )
        self.aff=A.Compose([
            A.ShiftScaleRotate(shift_limit=0.12,scale_limit=0.12,rotate_limit=30,interpolation=1,border_mode=0,value=0,mask_value=0,p=1),
            ],
            additional_targets=inf
            )
            
        self.query={}
        for i,dset_ind in enumerate(self.indlist):
            self.query[dset_ind]=i
        
        self.augment="comp_cut"
        self.grid2d=None

    def cut(self,feed,p=1):
        if np.random.random()<p:
            point=np.random.random(2)*np.array(self.shape[1:3])
            m=np.tan(np.pi*(np.random.random()-0.5))
            if np.isnan(m) or np.isinf(m):
                m=1
            top_or_bottom=np.random.random()>0.5
            if self.grid2d is None:
                self.grid2d=np.array(np.meshgrid(np.arange(self.shape[1]),np.arange(self.shape[2]),indexing="ij"))
            valid=(self.grid2d[1]-point[1])>(m*(self.grid2d[0]-point[0]))
            if valid.sum()<((self.shape[1]*self.shape[2])/2):
                valid=~valid
            feed["image"]*=valid[:,:,None]
            feed["mask"]*=valid[:,:,None]
        return feed

    def get_trf(self,fr,mask,mode):
        if mode=="none":
            return fr,mask
        fr=fr.numpy()
        mask=mask.numpy()
        feed={}
        feed["mask"]=mask #z is automatically channel
        
        if mode=="comp_cut":
            fr=self.CD_to_end(fr)
            res=self.comp(image=fr,**feed)
            res=self.cut(res,p=0.5)
            fr=self.CD_back(res["image"])
            mask=res["mask"]
            return torch.tensor(fr),torch.tensor(mask)
            
        elif mode=="aff_cut":
            fr,mask=self.CD_to_end(fr)
            res=self.aff(image=fr,**feed)
            res=self.cut(res,p=0.5)
            fr=self.CD_back(res["image"])
            mask=res["mask"]
            return torch.tensor(fr),torch.tensor(mask)
            
        elif mode=="comp":
            fr,mask=self.CD_to_end(fr)
            res=self.comp(image=fr,**feed)
            fr=self.CD_back(res["image"])
            mask=res["mask"]
            return torch.tensor(fr),torch.tensor(mask)
            
        elif mode=="aff":
            fr,mask=self.CD_to_end(fr)
            res=self.aff(image=fr,**feed)
            fr=self.CD_back(res["image"])
            mask=res["mask"]
            return torch.tensor(fr),torch.tensor(mask)
            
        else:
            assert False,"wrong trf(augmentation) mode"

    def CD_to_end(self,fr):
        return fr.transpose(1,2,0,3).reshape(self.shape[1],self.shape[2],-1)

    def CD_back(self,fr):
        return fr.reshape(self.shape[1],self.shape[2],self.shape[0],-1).transpose(2,0,1,3)

    def __getitem__(self,i):
        assert 0<=i<self.num_frames_tot
        ii=self.indlist[i]
        fr=(torch.load(os.path.join(self.framefolpath,str(ii)+".pt"))[self.channels]/255).to(torch.float32)
        mask=torch.load(os.path.join(self.maskfolpath,str(ii)+".pt"))
        fr,mask=self.get_trf(fr,mask,self.augment)
        return [fr,mask]

    def __len__(self):
        return self.num_frames_tot

    def change_augment(self,augment):
        self.augment=augment
        return self.augment

    def get_dset_inds(self,inds):
        dset_inds=[]
        for ind in inds:
            dset_inds.append(self.query[ind])
        return dset_inds

class EvalDataset(Dataset):
    def __init__(self,folpath,channels,T,maxz=False):
        super().__init__()
        self.maxz=maxz
        self.folpath=folpath
        self.channels=channels
        self.framefolpath=os.path.join(self.folpath,"frames")
        self.num_frames_tot=T

    def __getitem__(self,i):
        assert 0<=i<self.num_frames_tot
        fr=(torch.load(os.path.join(self.framefolpath,str(i)+".pt"))[self.channels]/255).to(torch.float32)
        if self.maxz:
            return fr.max(3)[0]
        else:
            return fr

    def __len__(self):
        return self.num_frames_tot


def get_mask(labels,coords,gridpts,radius=4):
    tree=sspat.cKDTree(coords)
    ds,iis=tree.query(gridpts,k=1)
    maskpts=labels[iis]
    return maskpts*(ds<=radius)
