from newsolver import predict_community_fullnp
import numpy as np
import pandas as pd
import random as rd
from numba import njit
from numba.typed import List
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle
import torch.optim as optim
import time
from math import sqrt
import matplotlib.pyplot as plt

data = pd.read_excel('RealData.xlsx', index_col=0)
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
    def __init__(self):
        super(MyNet, self).__init__()
        self.fc1 = nn.Linear(462, 462*5)
        self.fc2 = nn.Linear(462*5, 231*461)
    def forward(self,x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

net = MyNet().to(device)
criterion = nn.MSELoss(reduction='mean')
optimizer = optim.Adam(net.parameters(), lr=1e-4)

loss_v = []

for i in range(10000):
    optimizer.zero_grad()
    x, y = datagen()
    input = torch.from_numpy(x).float().to(device)
    true_y = torch.FloatTensor(y).to(device)

    output = net(input).to(device)

    loss = criterion(output, true_y).to(device)
    loss_v.append(sqrt(loss.item()))
    loss.backward()

    optimizer.step()

    print(f'Epoch {i}: {sqrt(loss.item())}')

for i in range(100):
  print(f'Test Epoch {i}')
  test_loss = 0
  x, y = datagen()
  test_x = torch.from_numpy(x).float().to(device)
  test_y = torch.FloatTensor(y).to(device)

  output = net(input).to(device)

  loss = criterion(output, true_y).to(device)
  test_loss += sqrt(loss.item())
  print(f'Average Test RMS: {test_loss/100}')

PATH = 'model.pth'
torch.save(net.state_dict(), PATH)

plt.plot(loss_v)
plt.savefig('Loss.png')
plt.show()
