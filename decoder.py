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
from sklearn.decomposition import PCA
import seaborn as sns

data = pd.read_excel('RealData.xlsx', index_col=0)
specs = data.columns.tolist()
trimmed_specs = []
typed_trimmed_specs = List()
pca = PCA(n_components=2)

for i in range(len(specs)):
    if data.iloc[:,i].astype(bool).sum() >= 85:
        trimmed_specs.append(specs[i])
        typed_trimmed_specs.append(specs[i])

def datagen():
    lm = generate_matrix(typed_trimmed_specs)
    cm = predict_community_fullnp(lm, trimmed_specs, verb=False)
    return get_LT(lm), cm

# select CUDA if available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
if str(device) == 'cuda:0':
    print('CUDA device selected!')
elif str(device) == 'cpu':
	print('CUDA device not available. CPU selected')

class MyNet(nn.Module):
    def __init__(self, hyperparam):
        super(MyNet, self).__init__()
        self.fc1 = nn.Linear(231*461, hyperparam)
        self.fc2 = nn.Linear(hyperparam, 462)
    def forward(self,x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def train_net(model, train_size):
    loss_values = []
    pbar2 = tqdm(range(train_size))
    pca_training_communities = []
    pbar2.set_description(f'Training Neural Net on {train_size} Epochs')
    for epoch in pbar2:
        optimizer.zero_grad()
        x, y = datagen()

        input = torch.FloatTensor(x).to(device)
        true_y = torch.from_numpy(y).float().to(device)
        output = model(input).to(device)
        loss = criterion(output, true_y).to(device)
        s = WD(y.tolist(), output.detach().tolist())
        print(f'Epoch {epoch}: Loss {s}')
        loss_values.append(s)
        loss.backward()
        optimizer.step()
        pca_training_communities.append(y.tolist())
        pca_training_communities.append(output.detach().tolist())
    pca.fit(pca_training_communities)
    return loss_values

def test_net(model, test_size):
    pca_testing_communities = []
    labels = []
    pbar3 = tqdm(range(test_size))
    pbar3.set_description(f'Testing Neural Net on {test_size} Epochs')
    for epoch in pbar3:
        x, y = datagen()
        input = torch.FloatTensor(x).to(device)
        true_y = torch.from_numpy(y).float().to(device)
        output = model(input).to(device)
        s = WD(y.tolist(), output.detach().tolist())
        print(f'Epoch {epoch}: Loss {s}')
        pca_testing_communities.append(y.tolist())
        labels.append('Real')
        pca_testing_communities.append(output.detach().tolist())
        labels.append('Fake')
    transformed_data = pca.fit_transform(pca_testing_communities)
    return transformed_data, labels

if __name__=='__main__':
    train_size, test_size, param, = 3000, 25, 2500
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

    lv = train_net(net, train_size=train_size)
    transformed_data, labels = test_net(model=net, test_size=test_size)
    df = pd.DataFrame(transformed_data, columns=['PCA1', 'PCA2'])

    ax0 = sns.scatterplot(data=df, x='PCA1', y='PCA2', hue=labels)
    ax0.set(title='Ordination Plot', xlabel='PCA1', ylabel='PCA2')

    plt.savefig('Ordination Plot')
    plt.show()
