import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import pickle
import seaborn as sb
from predict_by_model import *
import openpyxl
import random as rd


def GetStrn(strg):
    s1 = strg.split(";")[-1]
    s2 = "_".join(s1.split("__")[1:])
    return s2

def model_comm_eq(csv, _sep, dirhead, ObservationFile):

    miceData = pd.read_csv(csv, sep=_sep)
    Experiments = miceData.columns[np.where(['WK' in colnm for colnm in miceData.columns])]
    species = miceData.species.apply(GetStrn)

    CommunityEquilibrium = {}
    FoundLists = {}
    observed_communities = {}

    for ind, exp in enumerate(Experiments):

        #setup
        dirname = dirhead + str(ind+1)
        try:
            os.mkdir(dirname)
        except:
            pass

        #algorithm generation stuff
        by_proportion = miceData[exp]/sum(miceData[exp])
        spec_list = list(species[by_proportion>0.001])
        by_proportion.index = species
        Equilibrium,FoundList = predict_community(spec_list,File = ObservationFile,lambdaVersion = "Equilibrium", verb = True)


        CommunityEquilibrium[exp] = dict([(ky,val.round(3)) for ky,val in Equilibrium.items()])
        FoundLists[exp] = FoundList
        observed_communities[exp] = by_proportion[np.unique(FoundList)][~by_proportion[np.unique(FoundList)].index.duplicated()]

        fig,ax = plt.subplots(figsize = (10,10))

        keys = list(CommunityEquilibrium[exp].keys())
        # get values min the same order as keys, and parse percentage values
        vals = [float(CommunityEquilibrium[exp][k]) for k in keys]
        sb.barplot(x=keys, y=vals, ax = ax)
        ax.set_xticklabels(keys, rotation = 90)
        fig.savefig(dirname + "/commeq.png")

        fig,ax = plt.subplots(figsize = (10,10))

        keys = list(observed_communities[exp].index)

        # get values in the same order as keys, and parse percentage values
        vals = [observed_communities[exp].loc[k] for k in keys]
        data_dict = {}
        for i in range(len(keys)):
            data_dict[keys[i]] = vals[i]
        d = pd.DataFrame(list(data_dict.items()))
        d.to_excel(dirname + "/eq" + str(ind+1)+ ".xlsx")

        # trial_data.to_csv(dirname + "/eq.csv")
        sb.barplot(x=keys, y=vals, ax = ax)
        ax.set_xticklabels(keys, rotation = 90)
        fig.savefig(dirname + "/obscomm.png")
        
def edit_pairwise_data(file):
    wb = openpyxl.load_workbook(file)
    sheet = wb['Relative_Abundance']

    for i in range(2, sheet.max_row+1):
        for j in range(2, sheet.max_column+1):
            if i==j:
                sheet.cell(row=i,column=j).value = 1
            else:
                sheet.cell(row=i,column=j).value = rd.random()
    wb.save('rand.xlsx')

edit_pairwise_data('Pairwise_Chemostat.xlsx')
model_comm_eq(csv="Cdiff_mice_high_vs_low_risk.species.tsv", _sep='\t', dirhead='T', ObservationFile =  "Pairwise_Chemostat.xlsx")
