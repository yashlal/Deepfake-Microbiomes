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


X = []
str_y = []

for s in specs_in_CT:
    ind = specs_in_matrix.index(s)
    X.append(list(new_ar[ind+1][1:]))
    str_y.append(ar_CT[ind][14])

y = [1 if x=='CRC' else 0 for x in str_y]
X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, random_state=42)
# running Auto-PyTorch
autoPyTorch = AutoNetClassification("full_cs",  # config preset
                                    log_level='info',
                                    max_runtime=252000)

autoPyTorch.fit(X_train, y_train, validation_split=0.3)
y_pred = autoPyTorch.predict(X_test)

print("Accuracy score", sklearn.metrics.accuracy_score(y_test, y_pred))
# autonet = AutoNetClassification("full_cs", budget_type='epochs', min_budget=1, max_budget=9, num_iterations=1, log_level='debug', use_pynisher=False)
# autonet.fit(X_train, y_train, validation_split=0.3)
# y_pred = autonet.predict(X_test)

# print("Accuracy score", sklearn.metrics.accuracy_score(y_test, y_pred))
