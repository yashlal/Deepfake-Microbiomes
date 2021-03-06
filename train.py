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
    def __init__(self, hyperparam1, hyperparam2):
        super(MyNet, self).__init__()
        self.fc1 = nn.Linear(dim1, 3)
        self.fc2 = nn.Linear(hyperparam1,  hyperparam2)
        self.fc3 = nn.Linear(hyperparam2, dim2)
        self.sigmoid = nn.Sigmoid()
    def forward(self,x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return self.sigmoid(x)

def train_net(model, train_size):
    loss_values = []

    pbartrain=tqdm(range(train_size))
    pbartrain.set_description(f'Training Neural Net on {train_size} Epochs')
    for epoch in pbartrain:

        optimizer.zero_grad()

        x, y = datagen()

        input = torch.from_numpy(x).float().to(device)
        true_y = torch.FloatTensor(y).to(device)
        output = model(input).to(device)
        if (epoch%100)==0:
            print(output.detach().tolist(), y)
        loss = criterion(output, true_y).to(device)
        s = sqrt(loss.item()/(dim2))
        print(f'Epoch {epoch}: Loss {s}')
        loss_values.append(s)
        loss.backward()
        optimizer.step()


    return loss_values

def test_net(model, test_size):
    s = 0
    for epoch in range(test_size):
        x, y = datagen()

        input = torch.from_numpy(x).float().to(device)
        true_y = torch.FloatTensor(y).to(device)
        output = model(input).to(device)
        pred_lam = np.array(modules.regenerate_PWMatrix(output.tolist(), dim1))
        pred_cm = predict_community_fullnp(pred_lam, trimmed_specs, verb=False)
        s += WD(pred_cm,x)
        print(WD(pred_cm, x))
    return s/test_size

if __name__=='__main__':
    train_size, test_size, param1, param2 = 5000, 25, 3, 3
    path = 'model.pth'

    net = MyNet(param1, param2).to(device)

    #Multi GPU Support
    if torch.cuda.device_count() > 1:
          print(f'Using {torch.cuda.device_count()} GPUs')
          net = nn.DataParallel(net)
    elif torch.cuda.device_count() == 1:
        print(f'Using {torch.cuda.device_count()} GPU')

    criterion = nn.MSELoss(reduction='sum')
    optimizer = optim.Adam(net.parameters(), lr=1e-4)

    lv = train_net(net, train_size=train_size)
    test_perf = test_net(net, test_size)

    plt.plot(lv)
    plt.xlabel('Epoch Number')
    plt.ylabel('Loss')
    plt.title(f'{len(trimmed_specs)} Species: Loss over Time')
    plt.savefig('3SpecLoss')
    plt.show()

    print(test_perf)
