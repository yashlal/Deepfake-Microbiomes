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
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.optim as optim
import time
from math import sqrt
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from scipy.stats import wasserstein_distance as WD
import modules
import importlib
importlib.reload(modules)

data = pd.read_excel('RealData.xlsx', index_col=0)
specs = data.columns.tolist()
trimmed_specs = []
typed_trimmed_specs = List()

for i in range(len(specs)):
    if data.iloc[:,i].astype(bool).sum() >= 85:
        trimmed_specs.append(specs[i])
        typed_trimmed_specs.append(specs[i])

dim1 = 3
dim2 = 3

trimmed_specs = trimmed_specs[0:dim1]
typed_trimmed_specs = typed_trimmed_specs[0:dim1]
# select CUDA if available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
if str(device) == 'cuda:0':
    print('CUDA device selected!')
elif str(device) == 'cpu':
	print('CUDA device not available. CPU selected')

def datagen():
    lm = modules.generate_matrix(typed_trimmed_specs)
    cm = modules.predict_community_fullnp(lm, trimmed_specs, verb=False)
    return (cm, modules.get_LT(lm))

class MyNet(nn.Module):
    def __init__(self, hyperparam1):
        super(MyNet, self).__init__()
        self.fc1 = nn.Linear(6, hyperparam1)
        self.fc2 = nn.Linear(hyperparam1, 6)
    def forward(self,x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def train_net(model, train_size):
    lv = []
    for traj in range(train_size):
        traj, lams = datagen()

        for t in range(traj.shape[0]-1):
            optimizer.zero_grad()

            x_ = np.concatenate((traj[i], lams))
            y_ = np.concatenate((traj[i+1], lams))

            input = torch.from_numpy(x_).float().to(device)
            true_y = torch.from_numpy(y_).float().to(device)

            output = model(input).to(device)
            loss = criterion(output, true_y).to(device)
            lv.append(loss.detach().item())

            loss.backward()
            optimizer.step()

            print(f'Trajectory {epoch}: Time {t+1}/{traj.shape[0]-1}: Loss is {lv[-1]}')

def test_net(model, test_size):
    pass

if __name__=='__main__':
    train_size, param1 = 3, 6
    path = 'model.pth'

    net = MyNet(param1).to(device)

    #Multi GPU Support
    if torch.cuda.device_count() > 1:
          print(f'Using {torch.cuda.device_count()} GPUs')
          net = nn.DataParallel(net)
    elif torch.cuda.device_count() == 1:
        print(f'Using {torch.cuda.device_count()} GPU')

    criterion = nn.MSELoss()
    optimizer = optim.Adam(net.parameters(), lr=1e-4)

    lv = train_net(net, train_size=train_size)

    plt.plot(lv)
    plt.xlabel('Epoch Number')
    plt.ylabel('Loss')
    plt.title(f'{len(trimmed_specs)} Species: Loss over Time')
    plt.savefig('3SpecLoss')
    plt.show()
