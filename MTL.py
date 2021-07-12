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
from modules import *
from scipy.stats import wasserstein_distance as WD

# select CUDA if available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
if str(device) == 'cuda:0':
    print('CUDA device selected!')
elif str(device) == 'cpu':
	print('CUDA device not available. CPU selected')

data = pd.read_excel('RealData.xlsx', index_col=0)
specs = data.columns.tolist()
trimmed_specs = []
typed_trimmed_specs = List()

for i in range(len(specs)):
    if data.iloc[:,i].astype(bool).sum() >= 85:
        trimmed_specs.append(specs[i])
        typed_trimmed_specs.append(specs[i])

class MyNet(nn.Module):
    def __init__(self, hyperparam):
        super(MyNet, self).__init__()
        self.fc1 = nn.Linear(462, hyperparam)
        self.fc2 = nn.Linear(hyperparam, 231*461)
    def forward(self,x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def train_net(model, train_size, n_branches):
    true_outputs = []
    loss_values = []
    for i in range(n_branches):
        true_outputs.append(generate_matrix(typed_trimmed_specs))
    pbar=tqdm(range(train_size))
    pbar.set_description('Training Neural Net with 3 Branches')
    for epoch in pbar:
        total_loss = 0
        s = 0
        optimizer.zero_grad()
        loss_values.append([])
        for branch in range(n_branches):
            full_m = pd.DataFrame(true_outputs[branch], index=trimmed_specs, columns=trimmed_specs)
            npcm = np.zeros(len(trimmed_specs))
            size = rd.randint(25, 235)
            subset = rd.sample(trimmed_specs, size)
            subset_lam = (full_m.loc[subset, subset]).to_numpy()
            cm = predict_community_fullnp(subset_lam, subset, verb=False)
            for j in range(len(cm)):
                npcm[trimmed_specs.index(subset[j])] = cm[j]

            input = torch.from_numpy(npcm).float().to(device)
            true_y = torch.FloatTensor(get_LT(true_outputs[branch])).to(device)
            output = model(input).to(device)
            loss = criterion(output, true_y).to(device)
            loss_values[-1].append(loss.item())
            total_loss += loss
            s += sqrt(loss.item()/(231*461))

        s = (s / n_branches)
        print(f'Epoch {epoch}: Loss {s}')
        total_loss.backward()
        optimizer.step()

    return loss_values


if __name__=='__main__':
    train_size, test_size, param, n_branches = 15000, 25, 2500, 1
    path = 'model.pth'

    net = MyNet(param).to(device)

    #Multi GPU Support
    if torch.cuda.device_count() > 1:
          print(f'Using {torch.cuda.device_count()} GPUs')
          net = nn.DataParallel(net)
    elif torch.cuda.device_count() == 1:
        print(f'Using {torch.cuda.device_count()} GPU')

    criterion = nn.MSELoss(reduction='sum')
    optimizer = optim.Adam(net.parameters(), lr=1e-4)

    lv = train_net(net, train_size=train_size, n_branches=n_branches)

    plt.plot(lv)
    plt.savefig('Loss')
    plt.show()
