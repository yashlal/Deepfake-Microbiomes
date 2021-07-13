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

data = pd.read_excel('RealData.xlsx', index_col=0)
specs = data.columns.tolist()
trimmed_specs = []
typed_trimmed_specs = List()

for i in range(len(specs)):
    if data.iloc[:,i].astype(bool).sum() >= 85:
        trimmed_specs.append(specs[i])
        typed_trimmed_specs.append(specs[i])

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

def train_net(model, train_size, super_train_size):
    loss_values = []
    lengths = []
    pbar2=tqdm(range(super_train_size))
    pbar2.set_description(f'Training Neural Net on {super_train_size} SuperEpochs')
    for super_epoch in pbar2:
        full_m = pd.DataFrame(generate_matrix(typed_trimmed_specs), index=trimmed_specs, columns=trimmed_specs)
        train_y = get_LT(full_m.to_numpy())
        pbartrain=tqdm(range(train_size))
        pbartrain.set_description(f'Training Neural Net on {train_size} Epochs')
        for epoch in pbartrain:
            y = pd.DataFrame(np.zeros((len(trimmed_specs), len(trimmed_specs))), index=trimmed_specs, columns=trimmed_specs)
            # npcm = np.zeros(len(trimmed_specs))
            size = rd.randint(5, 15)
            # size = 2
            subset = rd.sample(trimmed_specs, size)
            subset_lam = (full_m.loc[subset, subset]).to_numpy()
            y.loc[subset, subset] = subset_lam
            cm = predict_community_fullnp(y.to_numpy(), trimmed_specs, verb=False)

            # for i in range(len(cm)):
            #   npcm[trimmed_specs.index(subset[i])] = cm[i]


            optimizer.zero_grad()

            x = cm
            y = get_LT(y.to_numpy())

            input = torch.from_numpy(x).float().to(device)
            true_y = torch.FloatTensor(y).to(device)
            output = model(input).to(device)
            loss = criterion(output, true_y).to(device)
            s = sqrt(loss.item()/(231*461))
            print(f'SuperEpoch {super_epoch}: Epoch {epoch}: Loss {s} (Subset Size {size})')
            loss_values.append(s)
            lengths.append(size)
            loss.backward()
            optimizer.step()
            # if s<=0.002:
            #     break

    return loss_values, lengths

if __name__=='__main__':
    super_train_size, train_size, test_size, param, = 1, 3000, 25, 2500
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

    lv, lg = train_net(net, train_size=train_size, super_train_size=super_train_size)

    plt.plot(lv)
    plt.xlabel('Epoch Number')
    plt.ylabel('Loss')
    plt.title('Loss over Time')
    plt.savefig('Loss')
    plt.show()
