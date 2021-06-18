import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from predict_by_model import *
from generator import genrand
from classifier import *

def genLam():
    cr = pd.read_excel('Corr.xlsx', index_col=0)
    print(cr)
    arr = cr.to_numpy()

    for i in range(len(arr)):
        for j in range(len(arr)):
            if i==j:
                arr[i][j] = 0
            elif i>j:
                arr[i][j] = (arr[i][j]+0.2)*40
            elif j>i:
                arr[i][j] = 1
    cr.to_excel('Lam.xlsx')

def main():
    Lam = pd.read_excel('LamTest.xlsx', index_col=0)
    # com = predict_community(Lam, comm=Lam.index.to_list(), verb=True, vectorize=True)

    samples = pd.read_excel('RealData.xlsx', index_col=0)
    ordering = samples.columns.to_list()
    prob_distro = {}
    AllCommunityEquilibrium = {}


    for i in range(samples.columns.shape[0]):
        l = samples.iloc[:,i].to_numpy()
        prob_distro[samples.columns[i]] = np.count_nonzero(l) / l.shape[0]

    epoch = 1
    while epoch <= 100:
        print(epoch)
        AllCommunityEquilibrium[epoch] = {}
        spec_list = []
        for i in prob_distro:
            bool = genrand(prob_distro[i])
            if bool:
                spec_list.append(i)
            else:
                AllCommunityEquilibrium[epoch][i] = 0
        comm = predict_community(Lam, spec_list, verb=True)
        if comm=={}:
            continue
        for key,val in comm.items():
            AllCommunityEquilibrium[epoch][key] = val.round(3)
        spec_list.clear()
        epoch += 1

    df_output = pd.DataFrame(AllCommunityEquilibrium).transpose()[ordering]
    df_output.to_excel('GeneratorOutput/CU.xlsx')
