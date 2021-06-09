import pandas as pd
import numpy as np
from generator import *
# data and metric imports
import sklearn.model_selection
import sklearn.datasets
import sklearn.metrics
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier
from skbio.stats.distance import permanova
from scipy.spatial import distance
from skbio import DistanceMatrix
import matplotlib.pyplot as plt

#generates useable lists from the excel to feed into autoPyTorch
#Trimmed291 has been trimmed only for species with at least 1 sample where they have 5% RA (sheet has only 291 parameters/rows)
def gen_data_lists(datafile):
    df1=pd.read_excel(datafile, sheet_name="Sheet1")
    np1=list(df1.to_numpy())
    # df2=pd.read_excel("YData.xlsx", sheet_name="Sheet1")
    # np2=list(df2.to_numpy())
    X = [list(i[1:]) for i in np1]
    # y = [int(j[1]) for j in np2]
    return np.array(X)

#Build discriminator
#X1xlsx is the excel wkbk with the real samples
#X2xlsx is the excek ekbk with the fake samples
#Uses Jensen Shannon Distance Metric with a PERMANOVA test
def discriminator(X1xlsx, X2xlsx):
    X1_df = pd.read_excel(X1xlsx, index_col=0)
    X1 = X1_df.to_numpy()
    y1 = np.ones(X1.shape[0])
    X2 = gen_data_lists(X2xlsx)
    y2 = np.zeros(X2.shape[0])
    X = np.concatenate((X1,X2), axis=0)
    y = np.concatenate((y1,y2))


    dm = np.zeros((X.shape[0],X.shape[0]))

    for i in range(X.shape[0]):
        for j in range(i+1):
            dm[i][j] = dm[j][i] = distance.jensenshannon(X[i], X[j])
    dm_from_np = DistanceMatrix(dm)
    results = permanova(dm_from_np, y, permutations=10000)
    return results
