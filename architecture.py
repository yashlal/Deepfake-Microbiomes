import pandas as pd
import numpy as np
import random as rd
from generator import *
from predict_by_model import *
from GenerateLambdas import *
import torch.optim as optim
from torch import nn
import torch.nn.functional as F
import torch
from torch.autograd import Variable


def get_LT(df):
    LT_ar = []
    for i in range(df.shape[0]):
        for j in range(i):
            LT_ar.append(df.iloc[i,j])
    return LT_ar

def regenerate_PWMatrix(LT_arr, dim):
    output_ar =[]

    for i in range(dim):
        output_ar.append([])

        for j in range(dim):
            if j<i:
                v = int(((i/2)*(i-1))+j)
                output_ar[-1].append(LT_arr[v])
            if j==i:
                output_ar[-1].append(1)
            if j>i:
                v = int(((j/2)*(j-1))+i)
                output_ar[-1].append(1-LT_arr[v])

    return output_ar
