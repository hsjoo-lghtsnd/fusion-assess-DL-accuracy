import numpy as np
from fusionAssess import Qmet

def SD(fim):
    fim = fim * 255.
    return np.sqrt(Qmet.my_cov(fim,fim))

def SF(fim):
    fim = fim * 255.
    (N,M) = fim.shape

    dx = np.subtract(fim[:-1,:], fim[1:,:])
    dy = np.subtract(fim[:,:-1], fim[:,1:])

    sx = np.multiply(dx,dx)
    sy = np.multiply(dy,dy)

    SF = sx.sum()+sy.sum()

    return np.sqrt(SF/(N*M))

