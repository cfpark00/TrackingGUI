import h5py
import numpy as np
supported_suffixes=["h5"]

class Dataset:
    def __init__(self,file_path):
        self.file_path=file_path
        self.suffix=self.file_path.split(".")[-1]
        assert self.suffix in supported_suffixes, suffix+" not supported"
        self.data=None

    def open(self):
        assert self.data is  None, "file already open"
        if self.suffix=="h5":
            self.data=h5py.File(self.file_path,"r+")

    def close(self):
        assert self.data is not None, "file not open"
        if self.suffix=="h5":
            self.data.close()
            self.data=None

    def get_frame(self,time):
        assert self.data is not None, "file not open"
        if self.suffix=="h5":
            return np.array(self.data[str(time-1)+"/frame"])
            
    def get_frame_z(self,time,z):
        assert self.data is not None, "file not open"
        if self.suffix=="h5":
            return np.array(self.data[str(time-1)+"/frame"][:,:,:,z])
            
    def get_data_info(self):
        assert self.data is not None, "file not open"
        if self.suffix=="h5":
            dict={"T":self.data.attrs["T"],
                "C":self.data.attrs["C"],
                "W":self.data.attrs["W"],
                "H":self.data.attrs["H"],
                "D":self.data.attrs["D"],
                "N_points":self.data.attrs["N_points"],
                }
            return dict
            
    def get_points(self):
        assert self.data is not None, "file not open"
        if self.suffix=="h5":
            return np.array(self.data["points"])
    
            
    def set_points(self,points):
        assert self.data is not None, "file not open"
        if self.suffix=="h5":
            self.data["points"][...]=points

    def get_helper(self,name):
        assert self.data is not None, "file not open"
        if self.suffix=="h5":
            key="helper_"+name
            if key not in self.data.keys():
                print("Helper not present, bug")
                return None
            else:
                return np.array(self.data[key])
                
        return retdict
        
    def get_helper_names(self):
        assert self.data is not None, "file not open"
        if self.suffix=="h5":
            ret=[]
            for key in self.data.keys():
                if key[:7]=="helper_":
                    ret.append(key[7:])
        else:
            ret=[]
        return ret

    def add_points(self,n_add):
        assert self.data is not None, "file not open"
        if self.suffix=="h5":
            if  "coords" not in self.data.keys():
                coords=np.full((self.data.attrs["T"],self.data.attrs["N_points"]+n_add+1,3),np.nan,dtype=np.float32)
            else:
                coords=np.array(self.data["coords"])
                del self.data["coords"]
                coords=np.concatenate([coords,np.full((coords.shape[0],n_add,3),np.nan)],axis=1)

            ds=self.data.create_dataset("coords",shape=coords.shape,dtype=coords.dtype)
            ds[...]=coords
            self.data.attrs["N_points"]=self.data.attrs["N_points"]+n_add
            
            
            
            
            
            
            
            
            
            
            
