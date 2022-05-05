import os
import sys
sys.path.append(os.getcwd())
from src.Dataset import Dataset

import numpy as np
from tqdm import tqdm

dataset=Dataset(sys.argv[1])
dataset.open()
data_info=dataset.get_data_info()
t_mean=np.zeros((data_info["C"],data_info["W"],data_info["H"],data_info["D"]),dtype=np.float32)
t_var=t_mean.copy()
t_m=t_mean.copy()
t_m[...]=np.inf
t_M=t_mean.copy()
t_M[...]=-np.inf

s_mean=np.zeros((data_info["T"],data_info["C"]),dtype=np.float32)
s_var=s_mean.copy()
s_m=s_mean.copy()
s_m[...]=np.inf
s_M=s_mean.copy()
s_M[...]=-np.inf
for t in tqdm(range(data_info["T"])):
    fr=dataset.get_frame(t)/255
    t_mean+=fr
    t_var+=fr**2
    t_m=np.minimum(t_m,fr)
    t_M=np.maximum(t_M,fr)

    s_mean[t]=np.mean(fr,axis=(1,2,3))
    s_var[t]=np.var(fr,axis=(1,2,3))
    s_m[t]=np.min(fr,axis=(1,2,3))
    s_M[t]=np.max(fr,axis=(1,2,3))

t_mean/=data_info["T"]
t_var/=data_info["T"]
t_var-=t_mean**2
dataset.set_data("s_mean",s_mean,overwrite=True)
dataset.set_data("s_var",s_var,overwrite=True)
dataset.set_data("s_min",s_m,overwrite=True)
dataset.set_data("s_max",s_M,overwrite=True)
dataset.set_data("t_mean",t_mean,overwrite=True)
dataset.set_data("t_var",t_var,overwrite=True)
dataset.set_data("t_min",t_m,overwrite=True)
dataset.set_data("t_max",t_M,overwrite=True)
dataset.close()
