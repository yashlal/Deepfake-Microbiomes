from newsolver import predict_community_fullnp
import numpy as np
import pandas as pd
import random as rd
import tables
import time
from numba import njit
from numba.typed import List
import matplotlib.pyplot as plt
import seaborn as sn

data = pd.read_excel('RealData.xlsx', index_col=0)
specs = data.columns.tolist()
trimmed_specs = []

for i in range(len(specs)):
    if data.iloc[:,i].astype(bool).sum() >= 85:
        trimmed_specs.append(specs[i])

typed_trimmed_specs = List()
[typed_trimmed_specs.append(x) for x in trimmed_specs]

@njit
def get_LT(full_ar):
    ar = List()
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
                ar[i][j] = r/(1-r)
                ar[j][i] = 1
    return ar

def save(data_):
    h5file = tables.open_file("E:/train.h5", mode='a')
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
        lm = generate_matrix(typed_trimmed_specs, 0)
        df = pd.DataFrame(lm, index=trimmed_specs, columns=trimmed_specs)
        rd.shuffle(trimmed_specs)
        newlm = df.loc[trimmed_specs, trimmed_specs].to_numpy()
        cm = predict_community_fullnp(newlm, trimmed_specs, verb=False)

        if cm==[]:
            continue
        else:
            data_object = [get_LT(lm), cm]
            datastorage.append(data_object)
            if (len(datastorage)%5000)==0:
                save(datastorage)
                datastorage.clear()
            epoch += 1

# sum = np.zeros(len(trimmed_specs))
# x = [d for d in range(len(trimmed_specs))]
#
# n=500
# for i in range(n):
#     print(i)
#     lm = generate_matrix(trimmed_specs, 0)
#     df = pd.DataFrame(lm, index=trimmed_specs, columns=trimmed_specs)
#     rd.shuffle(trimmed_specs)
#     newlm = df.loc[trimmed_specs, trimmed_specs].to_numpy()
#     cm = predict_community_fullnp(newlm, trimmed_specs)
#     sum += cm
#
# sum = sum/n
# plt.plot(x, sum)
# plt.axhline(1/len(trimmed_specs))
# plt.title('s=400 n=500')
# plt.xlabel('Species Index')
# plt.ylabel('Average Equilibrium RA')
# plt.show()

datagen(100)
