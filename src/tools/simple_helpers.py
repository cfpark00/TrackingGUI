def load_settings(fn):
    dict={}
    with open(fn,"r") as f:
        for line in f.readlines():
            key,val=line.strip().split("=")
            dict[key]=val
    return dict
