import os
import sys
sys.path.append(os.getcwd())
from src.Dataset import *

import os
import shutil
import numpy as np
import scipy.ndimage as sim
import threading
import time

class SimpleNNClass():
    def __init__(self,params):
        self.state=""
        self.cancel=False
        
    def run(self,file_path):
        dataset=Dataset(file_path)
        _,file=os.path.split(file_path)
        folname=file.split(".")[0]
        self.folpath=os.path.join("data","data_temp",folname)
        if os.path.exists(self.folpath):
            shutil.rmtree(self.folpath)
        os.makedirs(self.folpath)
        dataset.open()
        
        self.data_info=dataset.get_data_info()
        os.makedirs(os.path.join(self.folpath,"frames"))
        os.makedirs(os.path.join(self.folpath,"masks"))
        self.state=["Making Files",0]
        for t in range(1,self.data_info["T"]+1):
            if self.cancel:
                self.quit()
                return
            image=dataset.get_frame(t-1)
            np.save(os.path.join(self.folpath,"frames",str(t)+".npy"),image)
            self.state[1]=int(100*t/self.data_info["T"])
        self.state=["Training Network",0]
        for t in range(1,self.data_info["T"]+1):
            if self.cancel:
                self.quit()
                return
            time.sleep(0.001)
            self.state[1]=int(100*t/self.data_info["T"])
        self.state=["Extracting Points",0]
        for t in range(1,self.data_info["T"]+1):
            if self.cancel:
                self.quit()
                return
            time.sleep(0.001)
            self.state[1]=int(100*t/self.data_info["T"])
        
        intensities=np.full((self.data_info["T"],self.data_info["N_points"]+1,3),np.nan)
        dataset.set_data("helper_SimpleNN",intensities,overwrite=True)
        dataset.close()
        shutil.rmtree(self.folpath)
        self.state="Done"
        
    def quit(self):
        shutil.rmtree(self.folpath)
        
def SimpleNN(command_pipe_sub,file_path,params):
    method=SimpleNNClass(params)
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
"""
def TargetNN(methodclass,command_pipe_sub,file_path,params):
    method=TargetNNClass(params)
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
"""
methods={"SimpleNN":SimpleNN,}

from src.methods.neural_network_tools import ThreeDCN