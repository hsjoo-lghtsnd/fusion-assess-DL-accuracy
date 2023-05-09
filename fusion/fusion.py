import numpy as np

def my_sum(data):
    return data.sum(axis=1)
def my_avg(data):
    return my_sum(data)/3.

def R(data):
    return data[:,0,::]
def G(data):
    return data[:,1,::]
def B(data):
    return data[:,2,::]

def my_min(data):
    return data.min(axis=1)
def my_max(data):
    return data.max(axis=1)
def my_mid(data):
    return my_sum(data) - my_min(data) - my_max(data)

func = {0: my_avg,
        1: R,
        2: G,
        3: B,
        4: my_min,
        5: my_max,
        6: my_mid}

def baseline(data,opt):
    fuse = func[opt]
    return fuse(data)


def return_all(data, opt):
    N = data.shape[0]
    fuse_img = func[opt]
    


    field = np.zeros((N, data.shape[2], data.shape[3]))

    for i in range(N):
        field[i,::]=fuse_img(data[i,::])
    return field
