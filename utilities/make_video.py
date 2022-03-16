import os
import sys
sys.path.append(os.getcwd())
from src.Dataset import Dataset

import argparse
from tqdm import tqdm
import shutil
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

parser = argparse.ArgumentParser(description='Adds points to a dataset')
parser.add_argument('file_path', help='file_path')
#parser.add_argument('channel_colors', help='file_path')
#parser.add_argument('channel_gammas', help='file_path')
parser.add_argument('-file_to', default="", help="destination file")

args=parser.parse_args()
file_path=args.file_path
fol,file_name=os.path.split(file_path)
name,suffix=file_name.split(".")
if args.file_to=="":
    file_to_path=os.path.join(fol,name+".mp4")
else:
    file_to_path=args.file_to
#channel_colors=args.channel_colors
channel_colors=[[255,0,255],[0,255,0]]
channel_gammas=[0.7,0.7]
channel_colors=np.array(channel_colors)/255
channel_gammas=np.array(channel_gammas)

dataset=Dataset(file_path)
dataset.open()
data_info=dataset.get_data_info()

points=dataset.get_points()
points[:,0]=np.nan
existing=np.nonzero(np.any(~np.isnan(points[:,:,0]),axis=0))[0]
n_existing=len(existing)
ptscolors=np.random.random((n_existing,3))

fig = plt.figure(figsize=(10,8))
ax1=fig.add_subplot(111)
ax1.axis("off")
im1=ax1.imshow(np.zeros((data_info["W"],data_info["H"])).T)
scat1=ax1.scatter(x=np.zeros(n_existing),y=np.zeros(n_existing),edgecolors=ptscolors, facecolors='none')
Tstr=str(data_info["T"])
ax1.annotate("Playback Speed: 2X",[10,10],color="white")

def animate(i):
    print("\r Progress: "+str(i+1)+"/"+Tstr,end="")
    frame=dataset.get_frame(i)/255
    frame=(frame.max(3)**channel_gammas[:,None,None]).transpose(2,1,0)@channel_colors
    im1.set_array(frame)
    scat1.set_offsets(points[i,existing,:2])
    return im1,scat1

anim = FuncAnimation(fig, animate,frames=100, interval=100, blit=True)
anim.save(file_to_path)#data_info["T"]

dataset.close()
