import h5py

def repack(h5fn):
    h5=h5py.File(h5fn,"r")
    h5new=h5py.File(h5fn+"_temp","w")
    for key,val in h5.items():
        h5.copy(key,h5new)
    for key,val in h5.attrs.items():
        h5new.attrs[key]=val
    h5.close()
    h5new.close()
    os.remove(h5fn)
    os.rename(h5fn+"_temp",h5fn)
