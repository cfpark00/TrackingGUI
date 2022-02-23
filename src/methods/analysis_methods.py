import numpy as np
import os
import sys
import scipy.ndimage as sim
sys.path.append(os.getcwd())
from src.Dataset import *

class GaussianIntegral():
    def __init__(self,params):
        self.parameters={"sigma_x":1.5,"sigma_y":1.5,"sigma_z":0.4,"ker_x":7,"ker_y":7,"ker_z":1}
        arrs=[np.arange(self.parameters[key])-self.parameters[key]//2 for key in ["ker_x","ker_y","ker_z"]]
        x,y,z=np.meshgrid(*arrs,indexing="ij")
        kernel=np.exp(-0.5*((x/self.parameters["sigma_x"])**2+(y/self.parameters["sigma_y"])**2+(z/self.parameters["sigma_z"])**2))
        kernel/=kernel.sum()
        self.kernel=kernel
        self.coord_grid=np.stack([x,y,z],axis=0)
        
    def __call__(self,file_path):
        dataset=Dataset(file_path)
        dataset.open()
        try:
            points=dataset.get_points()
            self.data_info=dataset.get_data_info()
            self.C=self.data_info["C"]
            self.xm,self.xM=-0.5+self.parameters["ker_x"]//2,self.data_info["W"]-0.5-self.parameters["ker_x"]//2
            self.ym,self.yM=-0.5+self.parameters["ker_y"]//2,self.data_info["H"]-0.5-self.parameters["ker_y"]//2
            self.zm,self.zM=-0.5+self.parameters["ker_z"]//2,self.data_info["D"]-0.5-self.parameters["ker_z"]//2
            intensities=np.full((self.data_info["T"],self.data_info["N_points"]+1,self.C),np.nan)
            for time in range(1,self.data_info["T"]+1):
                print("\r"+str(time)+"/"+str(self.data_info["T"]),end="")
                image=dataset.get_frame(time-1)
                locs=points[time-1]
                intensities[time-1,:,:]=self.get_values(locs,image)
                print()
        except Exception as ex:
            print(ex)
        dataset.set_data("GaussianIntegral",intensities)
        dataset.set_data("signal_GaussianIntegral",intensities[:,:,1]/intensities[:,:,0])
        dataset.close()
        
    
    def get_valids(self,locs):
        nonan=~np.isnan(locs[:,0])
        locs=np.nan_to_num(locs)
        inx=(self.xm<locs[:,0])*(locs[:,0]<self.xM)
        iny=(self.ym<locs[:,1])*(locs[:,1]<self.yM)
        inz=(self.zm<locs[:,2])*(locs[:,2]<self.zM)
        return nonan*inx*iny*inz
    
    def get_values(self,locs,image):
        valids=self.get_valids(locs)
        result=np.full((len(locs),self.C),np.nan)
        inds=np.nonzero(valids)[0]
        for ind,pix in zip(inds,locs[valids]):
            coords=self.coord_grid+pix[:,None,None,None]
            vals=np.stack([sim.map_coordinates(image[c], coords, order=1) for c in range(self.C)],axis=0)
            vals=(vals*self.kernel[None]).sum(axis=(1,2,3))
            result[ind,:]=vals
        return result



methods={"Gaussian Integral":GaussianIntegral,}
