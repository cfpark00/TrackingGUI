import numpy as np
import os
import sys
sys.path.append(os.getcwd())
from src.Dataset import *

import numpy as np
import scipy.ndimage as sim
import threading

class GaussianIntegralClass():
    def __init__(self,params):
        self.params={"sigma_x":1.5,"sigma_y":1.5,"sigma_z":0.4,"ker_x":7,"ker_y":7,"ker_z":1}
        self.params.update(params)

        arrs=[np.arange(self.params[key])-self.params[key]//2 for key in ["ker_x","ker_y","ker_z"]]
        x,y,z=np.meshgrid(*arrs,indexing="ij")
        kernel=np.exp(-0.5*((x/self.params["sigma_x"])**2+(y/self.params["sigma_y"])**2+(z/self.params["sigma_z"])**2))
        kernel/=kernel.sum()
        self.kernel=kernel
        self.coord_grid=np.stack([x,y,z],axis=0)
        self.state=""
        self.cancel=False


    def run(self,file_path):
        dataset=Dataset(file_path)
        dataset.open()

        points=dataset.get_points()
        self.data_info=dataset.get_data_info()
        self.C=self.data_info["C"]
        self.xm,self.xM=-0.5+self.params["ker_x"]//2,self.data_info["W"]-0.5-self.params["ker_x"]//2
        self.ym,self.yM=-0.5+self.params["ker_y"]//2,self.data_info["H"]-0.5-self.params["ker_y"]//2
        self.zm,self.zM=-0.5+self.params["ker_z"]//2,self.data_info["D"]-0.5-self.params["ker_z"]//2
        intensities=np.full((self.data_info["T"],self.data_info["N_points"]+1,self.C),np.nan)
        self.state=["Integrating",0]
        for t in range(1,self.data_info["T"]+1):
            if self.cancel:
                return
            image=dataset.get_frame(t-1)
            locs=points[t-1]
            intensities[t-1,:,:]=self.get_values(locs,image)
            self.state=["Integrating",int(100*(t/self.data_info["T"]))]

        dataset.set_data("GaussianIntegral",intensities,overwrite=True)
        dataset.set_data("signal_GaussianIntegral",intensities[:,:,1]/intensities[:,:,0],overwrite=True)
        dataset.close()
        self.state="Done"

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

def GaussianIntegral(command_pipe_sub,file_path,params):
    method=GaussianIntegralClass(params)
    thread=threading.Thread(target=method.run,args=(file_path,))
    while True:
        command=command_pipe_sub.recv()
        if command=="run":
            thread.start()
        elif command=="report":
            command_pipe_sub.send(method.state)
        elif command=="cancel":
            method.cancel=True
            thread.join()
            break
        elif command=="close":
            thread.join()
            break

methods={"Gaussian Integral":GaussianIntegral,}
