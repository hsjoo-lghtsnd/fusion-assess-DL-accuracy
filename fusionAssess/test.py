import torch

org = torch.randint(0,256,(3,32,32))
A = org[0,:,:]
B = org[1,:,:]
C = org[2,:,:]

A = A/255.
B = B/255.
C = C/255.

D = (A+B+C)/3.

print(A.shape)
print(A.dtype)
print(D.shape)

A = A.numpy()
B = B.numpy()
C = C.numpy()
D = D.numpy()

from pytictoc import TicToc
from fusionAssess import Integrate

t = TicToc()

t.tic()
print(Integrate.assess(D,A,B,C))
t.toc()

