import pandas as pd
import numpy as np
from autoPyTorch import AutoNetClassification

# data and metric imports
import sklearn.model_selection
import sklearn.datasets
import sklearn.metrics

matrix = pd.read_excel('Matrix.xlsx', sheet_name='Supplementary Data 1')
CT = pd.read_excel('ClassificationTable.xlsx', sheet_name='Supplementary Table S6', header=19)
old_ar = matrix.to_numpy()
new_ar = np.transpose(old_ar)
ar_CT = CT.to_numpy()

specs_in_matrix = list(old_ar[0][1:])
specs_in_CT = ar_CT[:,0]

# print(specs_in_matrix)
# print("_______________________________________________________________________________________________________________________")
# print(specs_in_CT)
X = []
str_y = []

for s in specs_in_CT:
    ind = specs_in_matrix.index(s)
    X.append(list(new_ar[ind+1][1:]))
    str_y.append(ar_CT[ind][14])

y = [1 if x=='CRC' else 0 for x in str_y]
X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, random_state=1)
# running Auto-PyTorch
autoPyTorch = AutoNetClassification("medium_cs",  # config preset
                                    log_level='info',
                                    max_runtime=1200,
                                    min_budget=50,
                                    max_budget=150)

autoPyTorch.fit(X_train, y_train, validation_split=0.3)
y_pred = autoPyTorch.predict(X_test)

print('Prediction', y_pred)
print("Accuracy score", sklearn.metrics.accuracy_score(y_test, y_pred))
