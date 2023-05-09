import numpy as np

def my_mean(seq):
    L = seq.size
    return seq.sum()/L

def my_cov(s1, s2):
    # contract: s1.size == s2.size > 1
    
    s1 = s1.reshape(-1)
    s2 = s2.reshape(-1)

    m1 = my_mean(s1)
    m2 = my_mean(s2)

    N = s1.size
    if (N==1):
        print('error:my_cov() div/0')
        return

    cov = 0.
    for i in range(N):
        cov = cov + (s1[i] - m1)*(s2[i] - m2)
    return cov/(N-1)

def my_q0(im1, im2):
    if (im1.size != im2.size):
        print('error:my_q0() size mismatch')
        return
    m1 = my_mean(im1)
    m2 = my_mean(im2)

    c1 = my_cov(im1, im1) # note: it is already squared.
    c2 = my_cov(im2, im2)
    cxy = my_cov(im1, im2)

    return 4*cxy*m1*m2/((m1**2+m2**2)*(c1+c2))

def my_q(im1, im2, im3, fim):
    return (my_q0(im1,fim)+my_q0(im2,fim)+my_q0(im3,fim))/3.

def my_filt2(im, filt):
    matrix_container = np.lib.stride_tricks.as_strided(im,
            shape=tuple(np.add(np.subtract(im.shape,filt.shape),(1,1)))
                    +filt.shape,
            strides=im.strides*2)

    values = np.einsum('ij,klij->kl', filt, matrix_container)
    res = np.zeros(im.shape)

    res[1:-1,1:-1] = values

    return res


def my_filt2_old(im, filt):
    (N,M) = im.shape
    res = np.zeros((N,M))
    for n in range(1,N-1): # 4-for, but order=O(n^2)
        for m in range(1,M-1):
            for i in range(3):
                for j in range(3):
                    res[n,m] = res[n,m] + im[n+i-1,m+j-1]*filt[i,j]
    return res


def strength_fusion(im1,im2):
    return np.sqrt(np.multiply(im1,im1)+np.multiply(im2,im2))

def edge_image(im):
    filt_x = np.array([[1,1,1],[0,0,0],[-1,-1,-1]])
    filt_y = filt_x.transpose()

    ime_x = my_filt2(im, filt_x)
    ime_y = my_filt2(im, filt_y)
    ime = strength_fusion(ime_x,ime_y)
    return ime

def my_qE(im1, im2, im3, fim, alpha=1):
    # typically, alpha is nonnegative.
    im1e = edge_image(im1)
    im2e = edge_image(im2)
    im3e = edge_image(im3)
    fime = edge_image(fim)

    edgeQ = my_q(im1e,im2e,im3e,fime)
    edgeQ = edgeQ**alpha

    return my_q(im1,im2,im3,fim)*edgeQ

# Xydeas
def my_X_image(im):
    filt_y = np.array([[1,2,1],[0,0,0],[-1,-2,-1]])
    filt_x = filt_y.transpose()

    ime_x = my_filt2(im, filt_x)
    ime_y = my_filt2(im, filt_y)
    return (ime_x, ime_y)

def my_eps():
    # legacy code
    return 7./3 - 4./3 - 1

def my_div(num, den):
    den = (den==0)*my_eps() + den
    return np.divide(num,den)

def build_ga(im):
    (Tx, Ty) = my_X_image(im)
    g = strength_fusion(Tx,Ty)
    a = np.arctan(my_div(Ty,Tx))
    return (g, a)

def my_min(a,b):
    return (a>b)*b+(a<=b)*a

def build_GA_XF(im, fim):
    (gx, ax) = build_ga(im)
    (gf, af) = build_ga(fim)
    g1 = my_div(gx,gf)
    g2 = my_div(gf,gx)
    
    G_XF = my_min(g1,g2)

    A_XF = 1-my_div(np.abs(np.subtract(ax,af)),np.pi/2)
    return (G_XF, A_XF, gx, gf)

def my_sigmoid(x, gamma, kappa, b):
    return np.divide(gamma, 1+np.exp(kappa*(x-b)))

def my_qXy(im1, im2, im3, fim, L=1, GammaG=1, GammaA=1,
        kappaG=-10, kappaA=-20, bG=0.5, bA=0.75):
    # typically, L is nonnegative.
    (G_X1F, A_X1F, g1, gf) = build_GA_XF(im1, fim)
    (G_X2F, A_X2F, g2, gf) = build_GA_XF(im2, fim)
    (G_X3F, A_X3F, g3, gf) = build_GA_XF(im3, fim)

    QGX1F = my_sigmoid(G_X1F,GammaG,kappaG,bG)
    QGX2F = my_sigmoid(G_X2F,GammaG,kappaG,bG)
    QGX3F = my_sigmoid(G_X3F,GammaG,kappaG,bG)

    QAX1F = my_sigmoid(A_X1F,GammaA,kappaA,bA)
    QAX2F = my_sigmoid(A_X2F,GammaA,kappaA,bA)
    QAX3F = my_sigmoid(A_X3F,GammaA,kappaA,bA)

    QX1F = np.multiply(QGX1F,QAX1F)
    QX2F = np.multiply(QGX2F,QAX2F)
    QX3F = np.multiply(QGX3F,QAX3F)

    w1 = np.power(g1,L)
    w2 = np.power(g2,L)
    w3 = np.power(g3,L)
    
    W = w1.sum()+w2.sum()+w3.sum()

    Q1 = np.multiply(QX1F,w1)
    Q2 = np.multiply(QX2F,w2)
    Q3 = np.multiply(QX3F,w3)

    Q = Q1.sum()+Q2.sum()+Q3.sum()

    return Q/W
    



