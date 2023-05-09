from fusionAssess import MI
# mutual_info(fim, im1, im2, im3)
# tsallis_info(fim, im1, im2, im3, q=1.85)
# Nava, 2007: q=0.43137

from fusionAssess import Qmet
# my_q(im1, im2, im3, fim)
# my_qE(im1, im2, im3, fim, alpha=1)
#
# my_qXy(im1, im2, im3, fim, L=1, GG=1, GA=1, kG=-10, kA=-20, bG=0.5, bA=0.75)

from fusionAssess import SIA
# SD(fim)
# SF(fim)

import numpy as np

def assess(fim, im1, im2, im3):
    Q1 = MI.mutual_info(fim, im1, im2, im3)
    Q2 = MI.tsallis_info(fim, im1, im2, im3)
    Q3 = MI.tsallis_info(fim, im1, im2, im3, 0.43137)

    Q4 = Qmet.my_q(im1, im2, im3, fim)
    Q5 = Qmet.my_qE(im1, im2, im3, fim)
    
    Q6 = Qmet.my_qXy(im1, im2, im3, fim)

    Q7 = SIA.SD(fim)
    Q8 = SIA.SF(fim)

    return (Q1, Q2, Q3, Q4, Q5, Q6, Q7, Q8)

def assess_all(fim, im1, im2, im3):
    # assess for all
    N = fim.shape[0]

    field = np.zeros((N,8))
    for i in range(N):
        field[i,:] = assess(fim[i,::], im1[i,::], im2[i,::], im3[i,::])
    return field

