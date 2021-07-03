import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle
import numpy as np
import torch.optim as optim
import time

device = torch.device("cuda" if use_cuda else "cpu")

class MyNet(nn.Module):
    def __init__(self):
        super(MyNet, self).__init__()
        self.fc1 = nn.Linear(462, 462)
        self.fc2 = nn.Linear(462, 462*5)
        self.fc3 = nn.Linear(462*5, 231*461)
    def forward(self,x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

net = MyNet().to(device)

# for n in range(1,151):
#     PATH = 'Data/Train1-150/' + str(n)
f = open('Data/Train1-150/1', 'rb')
data = pickle.load(f)

criterion = nn.MSELoss(reduction='sum')
optimizer = optim.SGD(net.parameters(), lr=0.01)

for i in range(100):
    start = time.time()
    print(i)

    optimizer.zero_grad()

    t1=time.time()
    input = torch.from_numpy(data[i][1]).float().to(device)
    true_y = torch.FloatTensor(data[i][0])
    print(time.time()-t1)

    t2=time.time()
    output = net(input)
    print(time.time()-t2)

    t3=time.time()
    loss = criterion(output, true_y)
    print(loss)
    print(time.time()-t3)

    t4=time.time()
    loss.backward()
    print(time.time()-t4)

    t5=time.time()
    optimizer.step()
    print(time.time()-t5)

    print(time.time()-start)

# PATH = 'model.pth'
# torch.save(net.state_dict(), PATH)
