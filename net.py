import torch
import torch.nn as nn
import torch.nn.functional as F

class MyNet(nn.Module):
    def __init__(self):
        super(MyNet, self).__init__()
        self.fc1 = nn.Linear(462, 462*5)
        self.fc2 = nn.Linear(462*5, 462*5)
        self.fc2 = nn.Linear(462*5, 462*50)
        self.fc3 = nn.Linear(462*50, 462*200)
        self.fc5 = nn.Linear(462*200, 231*461)
    def forward(self,x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = self.fc5(x)
net = MyNet()
print(net)
