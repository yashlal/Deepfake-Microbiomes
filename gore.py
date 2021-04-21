import numpy as np
import pandas as pd
import os
from predict_by_model import *
import matplotlib.pyplot as plt
import pickle


def generateOnSimplex(n):
    initialRands = np.append(np.random.rand(n-1),[0,1])
    li = np.sort(initialRands)
    return [li[i] - li[i-1] for i in range(1,len(li))]

def MSerrors(actual,prediction):
    MSqError = ((actual - pd.Series(prediction))**2).sum()/len(actual)
    MSqErrorTotal = MSqError.mean()
    return MSqError,MSqErrorTotal

def GenerateScale(actual,N = 10000):
    comm = actual.columns
    errTot = pd.Series(dict([(c,0) for c in comm]))
    errAllTot = 0
    for i in range(N):
        rando = generateOnSimplex(len(comm))
        randpred = pd.Series(dict([(comm[i],rando[i]) for i in range(len(comm))]))
        err,errAll = MSerrors(actual,randpred)
        errTot = err + errTot
        errAllTot += errAll
    return errTot/N,errAllTot/N

if __name__ == "__main__":

    N = 1000

    dir = "gore_data/"


    trioExperiments = pd.read_excel(dir+"trio_lastTransfer.xlsx",sheet_name = None)
    groupExperiments = pd.read_excel(dir+"7and8Species_lastTransfer.xlsx", sheet_name = None)

    allLambdas = GenerateLambdasFromExcelAllPairs(dir+"PairEq.xlsx")

    AllResultsDF = pd.DataFrame(columns = ["Experiment","Experiment Size","Species","Predicted","Actual","Error","Survival Preddiction","Survival Actual","Survival Correct"])

    for expKey,expDF in trioExperiments.items():

        community = list(expDF.iloc[:,1:].columns)
        subLam = allLambdas.loc[community,community]
        community_prediction,fulltraj = predict_justComm(subLam)

        community_actual = expDF[expDF["Unnamed: 0"] == 5][community]
        CA_normed = community_actual.div(community_actual.sum(axis = 1), axis = 0)
        mnCA_normed = CA_normed.mean()

        MSqError,MSqErrorTotal = MSerrors(CA_normed,community_prediction)

        errorScale,totErrorScale = GenerateScale(CA_normed,N = N)

        for mic in community:
            survCor = bool(community_prediction[mic].round(5)) == bool(mnCA_normed[mic].round(5))
            resultRow = [expKey,len(community),mic,community_prediction[mic],mnCA_normed[mic],MSqError[mic]/errorScale[mic],bool(community_prediction[mic].round(5)),bool(mnCA_normed[mic].round(5)),survCor]
            AllResultsDF.loc[expKey + "X" + mic] = resultRow

        resultRowT = [expKey,len(community),"Total",1,1,MSqErrorTotal/totErrorScale,sum([bool(community_prediction[mic]) for  mic in community]),sum([bool(mnCA_normed[mic]) for  mic in community]),all([AllResultsDF.loc[expKey + "X" + mic,"Survival Correct"] for mic in community])]
        AllResultsDF.loc[expKey + "XTotal"] = resultRowT

    for expKey,expDF in groupExperiments.items():


        community = list(expDF.iloc[:,1:].columns)
        subLam = allLambdas.loc[community,community]
        community_prediction,fulltraj = predict_justComm(subLam)

        community_actual = expDF[expDF["Unnamed: 0"] == 5][community]
        CA_normed = community_actual.div(community_actual.sum(axis = 1), axis = 0)
        mnCA_normed = CA_normed.mean()

        MSqError,MSqErrorTotal = MSerrors(CA_normed,community_prediction)

        errorScale,totErrorScale = GenerateScale(CA_normed,N = N)

        for mic in community:
            survCor = bool(community_prediction[mic].round(5)) == bool(mnCA_normed[mic].round(5))
            resultRow = [expKey,len(community),mic,community_prediction[mic],mnCA_normed[mic],MSqError[mic]/errorScale[mic],bool(community_prediction[mic].round(5)),bool(mnCA_normed[mic].round(5)),survCor]
            AllResultsDF.loc[expKey + "X" + mic] = resultRow

        resultRowT = [expKey,len(community),"Total",1,1,MSqErrorTotal/totErrorScale,sum([bool(community_prediction[mic]) for  mic in community]),sum([bool(mnCA_normed[mic]) for  mic in community]),all([AllResultsDF.loc[expKey + "X" + mic,"Survival Correct"] for mic in community])]
        AllResultsDF.loc[expKey + "XTotal"] = resultRowT

    AllResultsDF.to_csv(dir + "PredictionVsActual.csv")
