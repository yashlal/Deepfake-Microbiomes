from newsolver import predict_community_fullnp
import numpy as np
import csv
import pandas as pd
import random as rd
from numba import njit
from numba.typed import List
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
import pickle
import torch.optim as optim
import time
from math import sqrt
import matplotlib.pyplot as plt
from tqdm import tqdm

train_size, test_size = 3000, 50

data = pd.read_excel('RealData.xlsx', index_col=0, engine='openpyxl')
specs = data.columns.tolist()
trimmed_specs = []

for i in range(len(specs)):
    if data.iloc[:,i].astype(bool).sum() >= 85:
        trimmed_specs.append(specs[i])
dim1 = len(trimmed_specs)

typed_trimmed_specs = List()
[typed_trimmed_specs.append(x) for x in trimmed_specs]

@njit()
def get_LT(full_ar):
    ar = []
    for i in range(len(full_ar)):
        for j in range(i):
            ar.append(full_ar[i][j])
    return ar

@njit()
def generate_matrix(comm, tolerance):
    dim = len(comm)
    ar = np.zeros((dim,dim))

    for i in range(dim):
        for j in range(i+1):
            if i == j:
                ar[i][j] = 0
            else:
                r = rd.random()
                # m = mult[i*dim1+j]
                ar[i][j] = r
                ar[j][i] = (1-r)

    return ar

def datagen():
    lm = generate_matrix(typed_trimmed_specs, 0)
    cm = predict_community_fullnp(lm, trimmed_specs, verb=False)
    return (cm, get_LT(lm))

# select CUDA if available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
if str(device) == 'cuda:0':
	print('CUDA device selected!')
elif str(device) == 'cpu':
	print('CUDA device not available. CPU selected')


class MyNet(nn.Module):
    def __init__(self, hyperparam):
        super(MyNet, self).__init__()
        self.fc1 = nn.Linear(462, hyperparam)
        self.fc2 = nn.Linear(hyperparam, 231*461)
    def forward(self,x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

mytest_x = []
mytest_y = []
mytrain_x = []
mytrain_y = []

pbar1=tqdm(range(test_size))
pbar1.set_description('Generating Test Set')
for i in pbar1:
      x, y = datagen()
      mytest_x.append(torch.from_numpy(x).float().to(device))
      mytest_y.append(torch.FloatTensor(y).to(device))

pbar2=tqdm(range(train_size))
pbar2.set_description('Generating Train Set')
for i in pbar2:
      x, y = datagen()
      mytrain_x.append(torch.from_numpy(x).float().to(device))
      mytrain_y.append(torch.FloatTensor(y).to(device))

def test_net(model, test_x, test_y):
    test_loss = 0
    for i in range(len(test_x)):
      input, true_y = test_x[i], test_y[i]

      output = model(input).to(device)
      loss = criterion(output, true_y).to(device)
      test_loss += sqrt((loss.item())/(231*461))

    return test_loss/(len(test_x))

def testconfig(model):
  s_arr = []
  pbar3=tqdm(range(train_size))
  pbar3.set_description('Training Neural Net')
  for i in pbar3:
      optimizer.zero_grad()
      input, true_y = mytrain_x[i], mytrain_y[i]

      output = model(input).to(device)

      loss = criterion(output, true_y).to(device)
      s = sqrt((loss.item())/(231*461))
      s_arr.append(s)
      loss.backward()

      optimizer.step()
  acc = test_net(model, mytest_x, mytest_y)
  return acc

pbar4=tqdm(range(1530, 4630, 10))
pbar4.set_description('HP Tuning Progress')

with open('output.csv', 'a+', newline='') as f: # a+ CREATES IF NOT EXISTS
  for it in pbar4:
    hyperparam = it
    net = MyNet(hyperparam).to(device)

    # MULTI-GPU CONFIGURATION
    if torch.cuda.device_count() > 1:
        print(f'Using {torch.cuda.device_count()} GPUs')
        net = nn.DataParallel(net)
    elif torch.cuda.device_count() == 1:
        print(f'Using {torch.cuda.device_count()} GPU')
        pass

    criterion = nn.MSELoss(reduction='sum')
    optimizer = optim.Adam(net.parameters(), lr=1e-4)

    acc = testconfig(net)
    print(f'Hidden Layer n={it}')
    print(f'Test Acc: {acc}')
    
    # WRITE AS CSV
    writer = csv.writer(f)
    writer.writerow([it, acc])