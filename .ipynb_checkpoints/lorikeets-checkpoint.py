import numpy as np
import pandas as pd
import os
from predict_by_model import *


def GetStrn(strg):
    s1 = strg.split(";")[-1]
    s2 = "_".join(s1.split("__")[1:])
    s3 = s2.split(".")[0]
    return s3

if __name__ == "__main__":

    try:
        os.mkdir("LorikeetExperiments")
    except:
        pass


    birdData = pd.read_excel("LorikeetData.xlsx")
    Experiments = birdData.iloc[2:].rename({"Taxonomy":"SampleID",'Unnamed: 1':"LorikeetID",'Unnamed: 2':"Date",'Unnamed: 3':"Age",'Unnamed: 4':"Sex",'Unnamed: 5':"Species",'Unnamed: 6':"Enteritis"},axis  = 1)

    predictions = pd.DataFrame(index =  Experiments["SampleID"], columns = ["LorikeetID","Date","Age","Sex","Species","Enteritis","Invasion","CperfDelta","Resistance","CperfResistance","CperfPromotion","InitialCperfGrowth","SpeciesFound","SpeciesListed","ReadProportion"])

    species = pd.Series(Experiments.columns[7:]).apply(GetStrn)

    for exp in Experiments.index:

        by_proportion = Experiments.loc[exp][7:]/sum(Experiments.loc[exp][7:])
        spec_list = list(by_proportion[by_proportion>0.001].index)
        spec_list = [GetStrn(s) for s in spec_list]
        by_proportion.index = species


        Experiment = {"Community":spec_list,"Invader":"Clostridium_perfringens"}
        invName,invSuc,zt,t,Qt,InvaderQ,r0,foundlist = predict_by_model(Experiment,lambdaVersion = "Equilibrium", verb = True)
        cddelta = (zt.T[-1][-1] - zt.T[-1][0])/zt.T[-1][0]
        found = len(foundlist)
        total = len(spec_list)
        biomass = sum(by_proportion[np.unique(foundlist)])
        predictions.loc[Experiments.loc[exp,"SampleID"]] = [Experiments.loc[exp,"LorikeetID"],Experiments.loc[exp,"Date"],Experiments.loc[exp,"Age"],Experiments.loc[exp,"Sex"],Experiments.loc[exp,"Species"],Experiments.loc[exp,"Enteritis"],invSuc,cddelta,Qt,InvaderQ,r0+InvaderQ,r0,found,total,biomass]

    predictions.to_csv("LorikeetExperiments/LorikeetExperimentsEqFit.csv")
