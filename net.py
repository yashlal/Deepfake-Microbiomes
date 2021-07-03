import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle
import numpy as np
import torch.optim as optim
import time
from math import sqrt
from rich.console import Console; c = Console()

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
        self.fc2 = nn.Linear(462*5, 462*10)
        self.fc3 = nn.Linear(462*10, 462*20)
        self.fc4 = nn.Linear(462*20, 231*461)
        self.dropout = nn.Dropout(0.1)
    def forward(self,x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
       	x = F.relu(self.fc3(x))
        x = self.dropout(x)
        x = self.fc4(x)
        return x

net = MyNet().to(device)


# for n in range(1,151):
#     PATH = 'Data/Train1-150/' + str(n)
f = open('Data/Train1-150/1', 'rb')
data = pickle.load(f)

criterion = nn.MSELoss(reduction='mean')
optimizer = optim.Adam(net.parameters(), lr=1e-4)

with c.status('Training...'):
    for i in range(1000):
        optimizer.zero_grad()
        start = time.time()

        t1=time.time()
        input = torch.from_numpy(data[i][1]).float().to(device)
        true_y = torch.FloatTensor(data[i][0]).to(device)

        t2=time.time()
        output = net(input).to(device)

        t3=time.time()
        loss = criterion(output, true_y).to(device)

        t4=time.time()
        loss.backward()

        t5=time.time()
        optimizer.step()

        print(f'Epoch {i}: {sqrt(loss.item())}')

# PATH = 'model.pth'
# torch.save(net.state_dict(), PATH)
