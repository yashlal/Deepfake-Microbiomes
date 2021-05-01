import pandas as pd
import numpy as np
from predict_by_model import *
import random as rd

def genrand(p):
    v = rd.random()
    if v <= p:
        return 1
    else:
        return 0

def generate_matrix(excel, sheetname, output_file_name):
    df = pd.read_excel(excel, sheet_name=sheetname, index_col=0)
    labels = df.index.to_list()
    ar = df.to_numpy()
    i = 0
    while i <= (len(ar)-1):
        for j in range(i+1):
            if i == j:
                ar[i][j] = 1
            else:
                r = rd.random()
                ar[i][j] = r
                ar[j][i] = 1-r
        i += 1
    df.to_excel(output_file_name)

def generator_fxn(workbook, sheetname, n, pairwise_file):
    print('Reading excel...')
    df = pd.read_excel(workbook, sheet_name = sheetname, index_col = 0)
    l = df.index.to_list()
    prob_dict = {}
    spec_list = []

    print('Making Probability Distribution...')
    for i in l:
        prob_dict[i] = 1

    for j in range(1,n+1):

        print("Trial " + str(j) + " in progress...")
        for el in prob_dict:
            bin = genrand(prob_dict[el])
            if bin == 1:
                spec_list.append(el)

        Equilibrium, FoundList = predict_community(spec_list, File = pairwise_file, lambdaVersion = "Equilibrium", verb = True)
        CommunityEquilibrium[j] = dict([(ky,val.round(3)) for ky,val in Equilibrium.items()])
    return CommunityEquilibrium

print(generator_fxn('NewPW.xlsx', 'Relative_Abundance', 1, 'NewPW.xlsx'))
