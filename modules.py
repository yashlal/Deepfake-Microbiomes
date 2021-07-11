import pandas as pd
from scipy.spatial import distance
import numpy as np
import random as rd

def JSD(p,q):
    m = np.add(p,q)/2
    left_entr = 0
    right_entr = 0

    for i in range(len(p)):
        if p[i] == 0:
            continue
        else:
            left_entr += p[i]*(np.log(p[i]/m[i]))
    for i in range(len(q)):
        if q[i] == 0:
            continue
        else:
            right_entr += q[i]*(np.log(q[i]/m[i]))
    dist = 0.5*(left_entr+right_entr)
    return dist**0.5

def get_LT(df):
    LT_ar = []
    for i in range(df.shape[0]):
        for j in range(i):
            LT_ar.append(df.iloc[i,j])
    return LT_ar

def regenerate_PWMatrix(LT_arr, dim):
    output_ar = []

    for i in range(dim):
        output_ar.append([])

        for j in range(dim):
            if j<i:
                v = int(((i/2)*(i-1))+j)
                output_ar[-1].append(LT_arr[v])
            if j==i:
                output_ar[-1].append(0)
            if j>i:
                v = int(((j/2)*(j-1))+i)
                output_ar[-1].append(1-LT_arr[v])

    return output_ar
    
def genrand(p):
    v = rd.random()
    if v <= p:
        return 1
    else:
        return 0
