import autograd.numpy as np
import pandas as pd
from scipy.spatial import distance
import autograd
import pandas as pd
import numpy as np
import random as rd
from generator import *
from predict_by_model import *
from GenerateLambdas import *

def JSD(p,q):
    m = [((p[i]+q[i])/2) for i in range(len(p))]
    m=np.array(m, dtype='float')
    left_entr = 0
    right_entr = 0

    for i in range(len(p)):
        if p[i] == 0:
            left_entr = left_entr + 0
        else:
            left_entr += p[i]*(np.log(np.array(p[i]/m[i])))
    for i in range(len(q)):
        if q[i] == 0:
            right_entr = right_entr + 0
        else:
            right_entr += q[i]*(np.log(np.array(q[i]/m[i])))
    dist = 0.5*(left_entr+right_entr)
    return dist**0.5

def get_LT(df):
    LT_ar = []
    for i in range(df.shape[0]):
        for j in range(i):
            LT_ar.append(df.iloc[i,j])
    return LT_ar
    
def regenerate_PWMatrix(LT_arr, dim):
    output_ar =[]

    for i in range(dim):
        output_ar.append([])

        for j in range(dim):
            if j<i:
                v = int(((i/2)*(i-1))+j)
                output_ar[-1].append(LT_arr[v])
            if j==i:
                output_ar[-1].append(1)
            if j>i:
                v = int(((j/2)*(j-1))+i)
                output_ar[-1].append(1-LT_arr[v])

    return output_ar
