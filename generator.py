import pandas as pd
import numpy as np
from predict_by_model import *
import random as rd
import cProfile
from torch import nn, optim
import torch.nn.functional as F

# Generates a random value in a Bernoulli style, used to determine whether or not a species will be included in the list
def genrand(p):
    v = rd.random()
    if v <= p:
        return 1
    else:
        return 0

# Generates a random interaction matrix with 1s along the diagonal and a_ij = 1-a_ji property
def generate_matrix(excel, sheetname, output_file_name, tolerance):
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
                #first two clauses force near zero to zero and near 1 to 1 preventing extraordinarily large lambdas and stiff ODEs
                if r<=tolerance:
                    ar[i][j] = 0
                    ar[j][i] = 1
                elif r>=(1-tolerance):
                    ar[i][j] = 1
                    ar[j][i] = 0
                else:
                    ar[i][j] = r
                    ar[j][i] = 1-r
        i += 1
    df.to_excel(output_file_name)

#Takes a workbook, gets all the species, generates a random prob distro, then uses that to generate a species list and uses predict_community to predict the equilibrium. For testing, all probabilities are set to 1
def generator_fxn(workbook, sheetname, n, pairwise_file):
    print('Reading excel...')
    df = pd.read_excel(workbook, sheet_name = sheetname, index_col = 0)
    l = df.index.to_list()
    #correct ordering for CU
    ordering = df.columns.to_list()
    prob_dict = {}
    spec_list = []
    CommunityEquilibrium = {}
    print('Generating All Lambdas...')
    Complete_Lambdas = GenerateLambdasFromRADF(RA_df = df)
    print('Making Probability Distribution...')
    for i in l:
        prob_dict[i] = rd.random()

    for j in range(1,n+1):
        CommunityEquilibrium[j] = {}
        #Loop through all the trials and each time generate species list
        print("Trial " + str(j) + " in progress...")
        for el in prob_dict:
            bin = genrand(prob_dict[el])
            if bin == 1:
                spec_list.append(el)
            # Make sure data has same dimension as real samples (sets RA of non-present species to zero)
            elif bin == 0:
                CommunityEquilibrium[j][el] = 0

        Equilibrium = predict_community(FullLambdaMat=Complete_Lambdas, comm=spec_list, verb=True)
        # For loop because we have to append the values to prevent overwriting preset zeros from bin == 0 condition
        for ky,val in Equilibrium.items():
            CommunityEquilibrium[j][ky] = val.round(3)
        #clear list for new trial/loop
        spec_list.clear()

    # Order data in same order as real samples
    df_output = pd.DataFrame(CommunityEquilibrium)
    df_output = df_output.transpose()
    df_output = df_output[ordering]
    return df_output


#Run the generator
def main(n=500):
    CU = generator_fxn('PWMatrix.xlsx', 'Relative_Abundance', n, 'PWMatrix.xlsx')
    CU.to_excel('GeneratorOutput/CU.xlsx')
    # generate_matrix('PWMatrix.xlsx', 'Relative_Abundance', 'PWMatrix.xlsx', 0)

main()
