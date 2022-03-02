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
import scipy.spatial as sspat

import matplotlib.pyplot as plt

class NNClass():
    def __init__(self,params):
        params_dict={}
        try:
            if params!="":
                for eq in params.split(";"):
                    key,val=eq.split("=")
                    params_dict[key]=eval(val)
        except:
            print("param parse failed")
        self.state=""
        self.cancel=False
        
        params_dict={}
        try:
            for txt in params.strip().split(";"):
                if txt=="":
                    continue
                key,val=txt.split("=")
                params_dict[key]=eval(val)
        except:
            assert False, "Parameter Parse Failure"   
        self.params={"min_points":1,"channels":None,"mask_radius":4,"2D":False,
        "lr":0.01,
        "n_steps":1000,"batch_size":3,"augment":{0:"comp_cut",500:"aff_cut",1000:"aff"},
        "Targeted":False,"n_epoch_posture":10,"batch_size_posture":16,"umap_dim":None,
        "weight_channel":None,
        }
        self.params.update(params_dict)
        
        if self.params["2D"]:
            assert not self.params["Targeted"], "Targeted Augementation only in 3D"
        assert self.params["min_points"]>0

    def run(self,file_path):
        from src.methods.neural_network_tools import nntools
        import torch
        from src.methods.neural_network_tools import Networks
        if self.params["Targeted"] and self.params["umap_dim"] is not None:
            import umap
        self.device= torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.state=["Preparing",0]
        self.dataset=Dataset(file_path)
        _,file=os.path.split(file_path)
        folname=file.split(".")[0]
        self.folpath=os.path.join("data","data_temp",folname)
        if os.path.exists(self.folpath):
            shutil.rmtree(self.folpath)
            pass
        os.makedirs(self.folpath)
        self.dataset.open()
        self.data_info=self.dataset.get_data_info()
        os.makedirs(os.path.join(self.folpath,"frames"))
        os.makedirs(os.path.join(self.folpath,"masks"))
        os.makedirs(os.path.join(self.folpath,"log"))
        
        if self.params["2D"]:
            assert self.data_info["D"]==1,"Input not 2D"
        
        self.state=["Making Files",0]
        T,N_points,C,W,H,D=self.data_info["T"],self.data_info["N_points"],self.data_info["C"],self.data_info["W"],self.data_info["H"],self.data_info["D"]
        points=self.dataset.get_points()
        min_points=self.params["min_points"]
        channels=self.params["channels"]
        if channels is None:
            channels=np.arange(C)
        n_channels=len(channels)
        grid=np.stack(np.meshgrid(np.arange(W),np.arange(H),np.arange(D),indexing="ij"),axis=0)
        gridpts=grid.reshape(3,-1).T

        #don't think about non existing points
        existing=np.any(~np.isnan(points[:,:,0]),axis=0)
        existing[0]=1#for background
        labels_to_inds=np.nonzero(existing)[0]
        N_labels=len(labels_to_inds)
        inds_to_labels=np.zeros(N_points+1)
        inds_to_labels[labels_to_inds]=np.arange(N_labels)
        for t in range(1,T+1):
            if self.cancel:
                self.quit()
                return
            image=self.dataset.get_frame(t-1)[channels]
            image=torch.tensor(image,dtype=torch.uint8)
            torch.save(image,os.path.join(self.folpath,"frames",str(t-1)+".pt"))
            pts=points[t-1]
            valid=~np.isnan(pts[:,0])
            if valid.sum()>=min_points:
                inds=np.nonzero(valid)[0]
                maskpts=nntools.get_mask(inds_to_labels[inds],pts[inds],gridpts,self.params["mask_radius"])
                mask=torch.tensor(maskpts.reshape(W,H,D),dtype=torch.uint8)
                torch.save(mask,os.path.join(self.folpath,"masks",str(t-1)+".pt"))
            self.state[1]=int(100*t/T)
        
        if self.params["Targeted"]:
            self.state=["Embedding Posture Space Training",0]
            self.encnet=Networks.AutoEnc2d(sh2d=(W,H),n_channels=n_channels,n_z=min(20,T//2))
            self.encnet.to(device=self.device,dtype=torch.float32)
            self.encnet.train()
            
            data=nntools.EvalDataset(folpath=self.folpath,channels=channels,T=T,maxz=True)
            loader=torch.utils.data.DataLoader(data, batch_size=self.params["batch_size_posture"],shuffle=True,num_workers=4,pin_memory=True)
            opt=torch.optim.Adam(self.encnet.parameters(),lr=self.params["lr"])
            n_epoch=self.params["n_epoch_posture"]
            f=open(os.path.join(self.folpath,"log","enc_loss.txt"),"w")
            for epoch in range(n_epoch):
                for i,ims in enumerate(loader):
                    if self.cancel:
                        self.quit()
                        return
                    ims=ims.to(device=self.device,dtype=torch.float32)
                    res,latent=self.encnet(ims)
                    loss=torch.nn.functional.mse_loss(res,ims)
                    opt.zero_grad()
                    loss.backward()
                    opt.step()
                    self.state[1]=int(100*(epoch*T+i+1)/(n_epoch*T) )
                    f.write(str(loss.item())+"\n")
            f.close()

            self.state=["Embedding Posture Space Evaluating",0]
            self.encnet.eval()
            vecs=[]
            with torch.no_grad():
                for i in range(T):
                    if self.cancel:
                        self.quit()
                        return
                    _,latent=self.encnet(data[i].unsqueeze(0).to(device=self.device,dtype=torch.float32))
                    vecs.append(latent[0].cpu().detach().numpy())
                    self.state[1]=int(100*((i+1)/T) )
            vecs=np.array(vecs).astype(np.float32)
            self.dataset.set_data("latent_vecs",vecs)
            def standardize(vecs):
                m=np.mean(vecs,axis=0)
                s=np.std(vecs,axis=0)
                return (vecs-m)/(s+1e-10)
            vecs=standardize(vecs)
            if self.params["umap_dim"] is not None:
                u_map=umap.UMAP(n_components=self.params["umap_dim"])
                vecs=u_map.fit_transform(vecs)
            distmat=sspat.distance_matrix(vecs,vecs).astype(np.float32)
            self.dataset.set_data("distmat",distmat)
            #plt.imshow(distmat)
            #plt.show()
        
        self.state=["Training Network",0]
        if self.params["2D"]:
            self.net=Networks.TwoDCN(n_channels=n_channels,num_classes=N_labels)
        else:
            self.net=Networks.ThreeDCN(n_channels=n_channels,num_classes=N_labels)
        self.net.to(device=self.device,dtype=torch.float32)

        train_data=nntools.TrainDataset(folpath=self.folpath,channels=channels,shape=(C,W,H,D))
        print("tdl",len(train_data))
        loader=torch.utils.data.DataLoader(train_data, batch_size=self.params["batch_size"],shuffle=True, num_workers=4,pin_memory=True)
        n_batches=len(loader)
        opt=torch.optim.Adam(self.net.parameters(),lr=self.params["lr"])
        n_steps=self.params["n_steps"]
        
        self.net.train()
        traindone=False
        stepcount=0
        while not traindone:
            if stepcount in self.params["augment"].keys():
                train_data.change_augment(self.params["augment"][stepcount])
            for i,(ims,masks) in enumerate(loader):
                if self.cancel:
                    self.quit()
                    return
                if self.params["2D"]:
                    ims=ims[:,:,:,:,0]
                    masks=masks[:,:,:,0]
                ims=ims.to(device=self.device,dtype=torch.float32)
                masks=masks.to(device=self.device,dtype=torch.long)
                res=self.net(ims)
                loss=nntools.selective_ce(res,masks)
                opt.zero_grad()
                loss.backward()
                opt.step()
                stepcount+=1
                self.state[1]=int(100*(stepcount/n_steps))
                if stepcount==n_steps:
                    traindone=True 
                    break
        
        if self.params["Targeted"]:
            self.net.eval()
            self.state=["Making Targeted Augmentation",0]
            for t in range(1,T+1):
                if self.cancel:
                    self.quit()
                    return
                time.sleep(0.001)
                self.state[1]=int(100*t/T)
            
            self.net.train()
            self.state=["Re Training Network",0]
            for t in range(1,T+1):
                if self.cancel:
                    self.quit()
                    return
                time.sleep(0.001)
                self.state[1]=int(100*t/T)

        self.state=["Extracting Points",0]
        ptss=np.full((T,N_points+1,3),np.nan)
        if self.params["2D"]:
            ptss[:,1:,2]=0
            grid=grid[:2,:,:,0]
            if self.params["weight_channel"] is None:
                weight=np.ones(grid.shape[1:])
        data=nntools.EvalDataset(folpath=self.folpath,channels=channels,T=T,maxz=False)
        self.net.eval()
        with torch.no_grad():
            for i in range(T):
                if self.cancel:
                    self.quit()
                    return
                im=data[i].to(device=self.device,dtype=torch.float32)
                if self.params["2D"]:
                    im=im[:,:,:,0]
                maskpred=torch.argmax(self.net(im.unsqueeze(0))[0],dim=0).cpu().detach().numpy()
                
                if self.params["weight_channel"] is None:
                    pts_dict=nntools.get_pts_dict(maskpred,grid,weight=weight)
                else:
                    im=im.cpu().detach().numpy()
                    pts_dict=nntools.get_pts_dict(maskpred,grid,weight=im[self.params["weight_channel"]])
                #print(pts_dict)
                if self.params["2D"]:
                    for label,coord in pts_dict.items():
                        ptss[i,label,:2]=coord
                else:
                    for label,coord in pts_dict.items():
                        ptss[i,label]=coord
                self.state[1]=int(100*((i+1)/T))
                if i==5:
                    break
        ptss[:,0,:]=np.nan
        
        self.dataset.set_data("helper_NN",ptss,overwrite=True)
        self.dataset.close()
        shutil.rmtree(self.folpath)
        self.state="Done"

    def quit(self):
        shutil.rmtree(self.folpath)
        

def NN(command_pipe_sub,file_path,params):
    method=NNClass(params)
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
methods={"NN":NN,}


if __name__=="__main__":
    import sys
    fp=sys.argv[1]
    method=NNClass({"Targeted":True,"n_epoch_posture":2,"batch_size_posture":1})
    method.run(fp)
