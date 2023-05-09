import numpy as np

def my_normalize(img, NORMALIZE=False):
    # maps the img into [0, 1)
    # however, if the img is constant, maps into 0.

    if (NORMALIZE):
        m = 0.
        M = 1.
    else:
        m = 0.
        M = 255.

    img = img - m
    if (M-m != 0):
        img = img/(M-m)
    return img

def my_histogram(im1, im2, N=256, NORMALIZE=False):
    # N stands for histogram size

    i1 = np.reshape(im1,-1)
    i2 = np.reshape(im2,-1)

    L = np.size(i1)

    if (L != np.size(i2)):
        print('my_hist(): image size mismatch\n', L, np.size(i2))
        return
    
    i1 = np.floor(my_normalize(i1,NORMALIZE) * (N-1)).astype('int')
    i2 = np.floor(my_normalize(i2,NORMALIZE) * (N-1)).astype('int')

    h=np.zeros((N,N))
    h1=np.zeros((N,))
    h2=np.zeros((N,))

    for i in range(L):
        h[i1[i],i2[i]] = h[i1[i],i2[i]] + 1
        h1[i1[i]] = h1[i1[i]] + 1
        h2[i2[i]] = h2[i2[i]] + 1

    return (h, h1, h2)

def my_log2(x):
    # returns 0 if x==0
    # may replace by:
    # return 0 if x==0 else np.log2(x)

    t = (x==0)
    return np.log2(x+t)

def my_p(im1, im2, N=256):
    (hj, h1, h2) = my_histogram(im1, im2, N, True)
    L = np.sum(hj, axis=None)
    
    p = hj / L
    p1 = h1 / L
    p2 = h2 / L
    return (p, p1, p2)

def my_MI(im1, im2, N=256):
    (p, p1, p2) = my_p(im1, im2, N)
    MI = 0.
    for i in range(N):
        for j in range(N):
            MI = MI + p[i,j] * (my_log2(p[i,j]) - my_log2(p1[i]*p2[j]))

    return MI

def my_eps():
    # legacy code
    return 7./3 - 4./3 - 1

def my_TMI(im1, im2, q=1.85, N=256):
    (p, p1, p2) = my_p(im1, im2, N)
    TMI = 0.
    for i in range(N):
        for j in range(N):
            if (p[i,j] == 0):
                continue
            TMI = TMI + ((p[i,j]**q) / ((p1[i]*p2[j])**(q-1)))
    return (1-TMI)/(1-q)

def mutual_info(fim, im1, im2, im3):
    # this function returns the mutual information metric

    MI = my_MI(fim, im1) + my_MI(fim, im2) + my_MI(fim, im3)
    return MI

def tsallis_info(fim, im1, im2, im3, q=1.85):
    # this function returns the Tsallis mutual information metric
    # Cvejic 2006, q=1.85
    # Nava, 2007, q=0.43137
    
    TMI = my_TMI(fim, im1, q) + my_TMI(fim, im2, q) + my_TMI(fim, im3, q)
    return TMI


