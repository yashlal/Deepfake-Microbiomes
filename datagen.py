from newsolver import predict_community_fullnp
import numpy as np
import pandas as pd
import random as rd
import time
from numba import njit
from numba.typed import List
import pickle

data = pd.read_excel('RealData.xlsx', index_col=0)
specs = data.columns.tolist()
trimmed_specs = []

for i in range(len(specs)):
    if data.iloc[:,i].astype(bool).sum() >= 85:
        trimmed_specs.append(specs[i])
dim1 = len(trimmed_specs)

typed_trimmed_specs = List()
[typed_trimmed_specs.append(x) for x in trimmed_specs]

@njit()
def get_LT(full_ar):
    ar = []
    for i in range(len(full_ar)):
        for j in range(i):
            ar.append(full_ar[i][j])
    return ar

@njit()
def generate_matrix(comm, tolerance):
    dim = len(comm)
    ar = np.zeros((dim,dim))

    for i in range(dim):
        for j in range(i+1):
            if i == j:
                ar[i][j] = 0
            else:
                r = rd.random()
                # m = mult[i*dim1+j]
                ar[i][j] = r
                ar[j][i] = (1-r)

    return ar

def datagen(n):
    datastorage = []
    epoch = 1
    saves = 107

    while epoch <= n:

        print(epoch)
        lm = generate_matrix(typed_trimmed_specs, 0)
        cm = predict_community_fullnp(lm, trimmed_specs, verb=False)

        if cm.shape[0]==0:
            continue
        else:
            data_object = [get_LT(lm), cm]
            datastorage.append(data_object)

            if (len(datastorage) % 1000)==0:
                PTH = 'Data/' + str(saves)
                outfile = open(PTH, 'wb')
                pickle.dump(datastorage, outfile)
                outfile.close()
                datastorage.clear()
                saves += 1

            epoch += 1

datagen(44000)
