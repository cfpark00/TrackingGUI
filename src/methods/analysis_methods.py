import numpy as np
import os
import sys 
sys.path.append(os.getcwd())
from src import Dataset

class GaussianIntegral():
    def __init__(self,params):
        self.parameters={"sigma_pix_x":1.5,"sigma_pix_y":1.5,"sigma_pix_z":0.5,"sigma_ker_x":6,"sigma_ker_x":6,"sigma_ker_x":2}
    
    def __call__(self,file_path):
        dataset=Dataset(file_path)
        dataset.open()
        points=dataset.get_points()
        data_info=dataset.get_data_info()
        print(points.shape)
        dataset.close()
        

methods={"Gaussian Integral":GaussianIntegral,}
