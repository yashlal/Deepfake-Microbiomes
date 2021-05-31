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
    df2=pd.read_excel("YData.xlsx", sheet_name="Sheet1")
    np2=list(df2.to_numpy())
    X = [list(i[1:]) for i in np1]
    y = [int(j[1]) for j in np2]
    return np.array(X), np.array(y)

#Run APT's AutoNetClassification
if __name__ == '__main__':
    # c=0.2
    X_data, y_data = gen_data_lists('XData.xlsx')
    print(X_data.shape)
    # while c<=20.1:
    #     lsvc = LinearSVC(C=c, penalty="l1", dual=False, max_iter=10000).fit(X_data, y_data)
    #     model = SelectFromModel(lsvc, prefit=True)
    #     X_new = model.transform(X_data)
    #     print(X_new.shape)
    #
    #     X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X_new, y_data, random_state=42)
    #
    #     clf = RandomForestClassifier(random_state=0)
    #     clf.fit(X_train, y_train)
    #     y_pred = clf.predict(X_test)
    #     # print('C value is', c)
    #     print("Accuracy Score", sklearn.metrics.accuracy_score(y_test, y_pred))
    #     print("C value is", c)
    #     c += 0.1

    # autoPyTorch = AutoNetClassification("full_cs", max_runtime=12000, min_budget=1200, max_budget=3600, log_level='info')
    # autoPyTorch.fit(X_train, y_train, validation_split=0.3)
    # y_pred = autoPyTorch.predict(X_test)
    # print("Accuracy Score", sklearn.metrics.accuracy_score(y_test, y_pred))
    # pytorch_model = autoPyTorch.get_pytorch_model()
    # print(pytorch_model)
    # pytorch_model(X_tens)

    lsvc = LinearSVC(C=3.3, penalty="l1", dual=False, max_iter=10000).fit(X_data, y_data)
    model = SelectFromModel(lsvc, prefit=True)
    X_new = model.transform(X_data)
    print(X_new.shape)

    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X_new, y_data, train_size=0.95, random_state=42)

    clf = RandomForestClassifier(random_state=0)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    train_pred = clf.predict(X_train)
    full_pred = clf.predict(X_new)
    # print('C value is', c)
    print("Test Accuracy Score", sklearn.metrics.accuracy_score(y_test, y_pred))
    print("Train Accuracy Score", sklearn.metrics.accuracy_score(y_train, train_pred))
    print("All Data Accuracy Score", sklearn.metrics.accuracy_score(y_data, full_pred))
