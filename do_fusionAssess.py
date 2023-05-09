import cifar10_loader
from pytictoc import TicToc
from fusionAssess import Integrate
from fusion import fusion
import numpy as np

def __main__():
    opt=int(input('input option[0-6]: '))
    if ((opt < 0) or (opt > 6)):
        print('invalid input')
        return

    t=TicToc()

    t.tic()
    (tr_im, tr_l, te_im, te_l) = cifar10_loader.cifar10('./data/cifar10')
    print('cifar10 load completed')
    t.toc()

    te = te_im.reshape(-1, 3, 32, 32)

    R = te[:,0,::]
    G = te[:,1,::]
    B = te[:,2,::]

    fim = fusion.baseline(te,opt)

    t.tic()
    data = Integrate.assess_all(fim, R,G,B)
    t.toc()

    filename = 'fusion'+str(opt)+'test.npy'
    np.save(filename, data)

    tr = tr_im.reshape(-1, 3, 32, 32)

    R = tr[:,0,::]
    G = tr[:,1,::]
    B = tr[:,2,::]

    fim = fusion.baseline(tr,opt)

    t.tic()
    data = Integrate.assess_all(fim, R,G,B)
    t.toc()

    filename = 'fusion'+str(opt)+'train.npy'
    np.save(filename, data)


__main__()
