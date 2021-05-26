import pandas as pd
import numpy as np
import torch
from torch import nn
from autoPyTorch import AutoNetClassification
from autoPyTorch import HyperparameterSearchSpaceUpdates

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
    matrix = pd.read_excel(datafile, sheet_name='Sheet1')
    CT = pd.read_excel('ClassificationTable.xlsx', sheet_name='Supplementary Table S6', header=19)
    old_ar = matrix.to_numpy()
    new_ar = np.transpose(old_ar)
    ar_CT = CT.to_numpy()

    specs_in_matrix = list(old_ar[0][1:])
    specs_in_CT = ar_CT[:,0]


    X = []
    str_y = []

    for s in specs_in_CT:
        ind = specs_in_matrix.index(s)
        X.append(list(new_ar[ind+1][1:]))
        str_y.append(ar_CT[ind][14])

    y = [1 if x=='CRC' else 0 for x in str_y]

    return np.array(X),np.array(y)

#Run APT's AutoNetClassification
if __name__ == '__main__':
    X, y = gen_data_lists('Matrix.xlsx')
    print(X.shape)

    # lsvc = LinearSVC(C=10, penalty="l1", dual=False).fit(X, y)
    # model = SelectFromModel(lsvc, prefit=True)
    # X_new = model.transform(X)
    # X_tens = torch.from_numpy(X_new)
    # print(X_new.shape)
    # print(X_tens.size())

    # X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X_new, y, train_size = 0.95, random_state=42)
    # clf = RandomForestClassifier(random_state=0)
    # clf.fit(X_train, y_train)
    # y_pred = clf.predict(X_test)

    # autoPyTorch = AutoNetClassification("full_cs", max_runtime=300, min_budget=30, max_budget=90, log_level='info')
    # autoPyTorch.fit(X_train, y_train, validation_split=0.3)
    #
    # y_pred = autoPyTorch.predict(X_test)
    # print("Accuracy Score", sklearn.metrics.accuracy_score(y_test, y_pred))
    # pytorch_model = autoPyTorch.get_pytorch_model()
    # print(pytorch_model)
    # pytorch_model(X_tens)
