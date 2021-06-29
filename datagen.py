from predict_by_model import *
import numpy as np
import pandas as pd
import random as rd
import tables
from multiprocessing import Pool
import time
from numba import njit
import timeit

specs = pd.read_excel('RealData.xlsx', index_col=0).columns.tolist()

@njit
def generate_matrix(comm, tolerance):
    dim = len(comm)
    ar = np.empty((dim,dim))

    for i in range(dim):
        for j in range(i+1):
            if i == j:
                ar[i][j] = 0
            else:
                r = rd.random()
                rat = r/1-r
                if rat<1:
                    ar[i][j] = r/(1-r)
                    ar[j][i] = 1
                else:
                    ar[i][j] = 1
                    ar[j][i] = 1/rat
    return ar

def save(data_):
    h5file = tables.open_file("Data/train.h5", mode='a')
    table = h5file.root.Group1.Train
    r = table.row

    for i in range(len(data_)):
        r['Lambda'] = data_[i][0]
        r['Community'] = data_[i][1]
        r.append()

    table.flush()
    h5file.close()

def datagen(n):
    datastorage = []
    epoch=1
    while epoch <= n:
        print(epoch)
        lm = generate_matrix(specs, 0)
        cm = predict_community_fullnp(lm, specs, verb=False)
        if cm==[]:
            continue
        else:
            data_object = [lm, cm]
            datastorage.append(data_object)
            if (len(datastorage)%1000)==0:
                save(datastorage)
                datastorage.clear()
            epoch += 1

datagen(100)
