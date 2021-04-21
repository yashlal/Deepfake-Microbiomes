import numpy as np
import pandas as pd
import os
from predict_by_model import *


def GetStrn(strg):
    s1 = strg.split(";")[-1]
    s2 = "_".join(s1.split("__")[1:])
    return s2

if __name__ == "__main__":

    try:
        os.mkdir("MiceCDiffExperiments")
    except:
        pass


    miceData = pd.read_csv("Cdiff_mice_high_vs_low_risk.species.tsv",sep = '\t')
    Experiments = miceData.columns[np.where(['WK' in colnm for colnm in miceData.columns])]

    predictions = pd.DataFrame(index =  Experiments, columns = ["Invasion","CDiffDelta","Resistance","CDiffResistance","CDiffPromotion","InitialCDiffGrowth","SpeciesFound","SpeciesListed","ReadProportion"])

    species = miceData.species.apply(GetStrn)

    for exp in Experiments:

        by_proportion = miceData[exp]/sum(miceData[exp])
        spec_list = list(species[by_proportion>0.001])
        by_proportion.index = species


    #     ### Without strain, function will select the first one (CD196)
    #     # ''Clostridium_difficile_CD196.mat'',
    #     # ''Clostridium_difficile_NAP07.mat'',
    #     # ''Clostridium_difficile_NAP08.mat'',
    #     # ''Clostridium_difficile_R20291.mat''

        experiment = {"Community":spec_list,"Invader":"Clostridium_difficile"}
        invName,invSuc,zt,t,Qt,InvaderQ,r0,foundlist = predict_by_model(experiment,lambdaVersion = "Equilibrium", verb = True)

        if np.sum(zt[-1]).round(3) != 1:
            print("###################################################################################")
            print("BUG -> zi do not sum to 1",np.sum(zt[-1]))
            print("###################################################################################")
        elif np.min(zt[-1]).round(3) < 0:
            print("###################################################################################")
            print("BUG -> zi<0",np.min(zt[-1]))
            print("###################################################################################")
        elif np.max(zt[-1]).round(3) >1:
            print("###################################################################################")
            print("BUG -> zi > 1",np.max(zt[-1]))
            print("###################################################################################")


        cddelta = (zt.T[-1][-1] - zt.T[-1][0])/zt.T[-1][0]
        found = len(foundlist)
        total = len(spec_list)
        biomass = sum(by_proportion[np.unique(foundlist)])
        predictions.loc[exp] = [invSuc,cddelta,Qt,InvaderQ,r0 + InvaderQ,r0,found,total,biomass]

    predictions.to_csv("MiceCDiffExperiments/micePredictionEqFit.csv")
