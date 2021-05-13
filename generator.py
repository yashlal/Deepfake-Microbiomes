import pandas as pd
import numpy as np
from predict_by_model import *
import random as rd

# Generates a random value in a Bernoulli style, used to determine whether or not a species will be included in the list
def genrand(p):
    v = rd.random()
    if v <= p:
        return 1
    else:
        return 0

# Generates a random interaction matrix with 1s along the diagonal and a_ij = 1-a_ji property
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

#Takes a workbook, gets all the species, generates a random prob distro, then uses that to generate a species list and uses predict_community to predict the equilibrium. For testing, all probabilities are set to 1
def generator_fxn(workbook, sheetname, n, pairwise_file):
    print('Reading excel...')
    df = pd.read_excel(workbook, sheet_name = sheetname, index_col = 0)
    l = df.index.to_list()
    prob_dict = {}
    spec_list = []
    CommunityEquilibrium = {}
    EditedCommunityEquilibrium = {} #hold non-zero commeq values

    print('Making Probability Distribution...')
    for i in l:
        prob_dict[i] = rd.random()

    for j in range(1,n+1):

        print("Trial " + str(j) + " in progress...")
        for el in prob_dict:
            bin = genrand(prob_dict[el])
            if bin == 1:
                spec_list.append(el)

        Equilibrium, FoundList = predict_community(spec_list, File = pairwise_file, lambdaVersion = "Equilibrium", verb = True)
        CommunityEquilibrium[j] = dict([(ky,val.round(3)) for ky,val in Equilibrium.items()])

        #new dict for non-zero values only
        EditedCommunityEquilibrium[j] = {}
        for k in CommunityEquilibrium[j]:
            if CommunityEquilibrium[j][k] != 0:
                EditedCommunityEquilibrium[j][k] = CommunityEquilibrium[j][k]
        spec_list.clear()
    return CommunityEquilibrium, EditedCommunityEquilibrium

#Run the generator
if __name__ == '__main__':
    CU, ECU = generator_fxn('NewPW.xlsx', 'Relative_Abundance', 100, 'NewPW.xlsx')
    print(CU)
    print("_____________________________________________________________________")
    print(ECU)
    df1 = pd.DataFrame(CU)
    df2 = pd.DataFrame(ECU)
    df1.to_excel('CU.xlsx')
    df2.to_excel('ECU.xlsx')
