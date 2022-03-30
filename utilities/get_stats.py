import os
import sys
sys.path.append(os.getcwd())
from src.Dataset import Dataset

import numpy as np
from tqdm import tqdm

dataset=Dataset(sys.argv[1])
dataset.open()
data_info=dataset.get_data_info()
mean=np.zeros((data_info["C"],data_info["W"],data_info["H"],data_info["D"]),dtype=np.float32)
var=mean.copy()
m=mean.copy()
m[...]=np.inf
M=mean.copy()
M[...]=-np.inf
for t in tqdm(range(data_info["T"])):
    fr=dataset.get_frame(t)/255
    mean+=fr
    var+=fr**2
    m=np.minimum(m,fr)
    M=np.maximum(M,fr)
mean/=data_info["T"]
var/=data_info["T"]
var-=mean**2
dataset.set_data("mean",mean,overwrite=True)
dataset.set_data("var",var,overwrite=True)
dataset.set_data("min",m,overwrite=True)
dataset.set_data("max",M,overwrite=True)
dataset.close()
