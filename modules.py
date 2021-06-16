import autograd.numpy as np
import pandas as pd
from scipy.spatial import distance
import autograd
# from autograd.variable import Variable

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
