import pandas as pd
import numpy as np
import torch
from torch import nn
from autoPyTorch import AutoNetClassification
from autoPyTorch import HyperparameterSearchSpaceUpdates
from generator import *
# data and metric imports
import sklearn.model_selection
import sklearn.datasets
import sklearn.metrics
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_selection import RFE, chi2
from sklearn.svm import LinearSVC
from sklearn.datasets import load_iris
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier

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
#LSVC enabled controls whether or not LSVC is added and c is the value for LSVC
def discriminator(X1xlsx, X2xlsx, lsvc_enabled=False, c=3.3, train_size=0.75):
    X1_df = pd.read_excel(X1xlsx, index_col=0)
    X1 = X1_df.to_numpy()
    y1 = np.ones(X1.shape[0])
    X2 = gen_data_lists(X2xlsx)
    y2 = np.zeros(X2.shape[0])
    X = np.concatenate((X1,X2), axis=0)
    y = np.concatenate((y1,y2))

    if lsvc_enabled:
        lsvc = LinearSVC(C=c, penalty="l1", dual=False, max_iter=10000).fit(X, y)
        model = SelectFromModel(lsvc, prefit=True)
        X_new = model.transform(X)
        print(X_new.shape)

        X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X_new, y, train_size=train_size, random_state=42)


    else:
        X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, train_size=train_size, random_state=42)

    clf = RandomForestClassifier(random_state=0)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print("Test Accuracy Score", sklearn.metrics.accuracy_score(y_test, y_pred))
