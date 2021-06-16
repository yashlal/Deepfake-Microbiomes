import pandas as pd
import autograd.numpy as np
import random as rd
from generator import *
from predict_by_model import predict_community
from GenerateLambdas import *
import autograd
from autograd.builtins import tuple
from scipy.integrate import odeint as BlackBox
import matplotlib.pyplot as plt
from architecture import get_LT, regenerate_PWMatrix
from scipy.spatial import distance
from modules import JSD
from datetime import datetime

dim1 = 280
dim2 = 39060

#system must be put into two functions because of incompatible input type weirdness
def f1(y,t,Lambda):
    mat = regenerate_PWMatrix(Lambda, dim1)
    term1 = np.dot(mat, y)
    return y*term1

def f2(y,t,Lambda):
    mat = regenerate_PWMatrix(Lambda, dim1)
    term2 = -np.dot(y.T, np.dot(mat, y))
    return y*term2

#Jacobians for two functions
J1 = autograd.jacobian(f1, argnum=0)
J2 = autograd.jacobian(f2, argnum=0)

#gfl means gradient wrt Lambda
gfl1 = autograd.jacobian(f1, argnum=2)
gfl2 = autograd.jacobian(f2, argnum=2)

def ODESYS(Y, t, Lambda):

    sensitivity_matrix = np.reshape(Y[-dim1*dim2:], (dim1, dim2))
    dydt = f1(Y[0:dim1], t, Lambda)+f2(Y[0:dim1], t, Lambda)
    #this line resolves the aforementioned "input type weirdness"
    Jac1 = [[j._value for j in i] for i in J1(Y[0:dim1], t, Lambda)]
    Jac = Jac1 + J2(Y[0:dim1], t, Lambda)
    GFL = gfl1(Y[0:dim1], t, Lambda)+gfl2(Y[0:dim1], t, Lambda)
    GYL = (Jac@sensitivity_matrix + GFL)
    #flatten
    GYL = np.reshape(GYL, (dim1*dim2,))
    return np.concatenate([dydt, GYL])

#COST = modules.JSD

#Setting up
target = (pd.read_excel('XData.xlsx').iloc[0,1:]).to_list()
target_vector = np.array([x for x in target if x!=0]).astype('float')
df1 = pd.read_excel('Sample1.xlsx', index_col=0)
init_lambda = GenerateLambdasFromRADF(df1).to_numpy()

init_y = ((np.ones(dim1))/dim1).astype('float')
init_grad = np.zeros((dim2*dim1,))
Y0 = np.concatenate([init_y, init_grad])

time = np.linspace(0, 50, num=500)

grad_C = autograd.grad(JSD)

maxiter = 100
learning_rate = 1 #Big steps
for i in range(maxiter):
    print(i, datetime.now().time())
    sol = BlackBox(ODESYS, y0 = Y0, t = time, args = tuple([init_lambda]))

    Y = sol[-1, :dim1]

    print(JSD(Y, target_vector), datetime.now().time())

    matr = np.reshape(sol[-1, -dim2*dim1:], ((dim1, dim2)))
    grad = ((grad_C(Y, target_vector)[:,None].T)@matr)[0]
    init_lambda = np.array([init_lambda[i] - grad[i] for i in len(range(grad))])

final_lambda = regenerate_PWMatrix(init_lambda, dim1)
df2 = pd.DataFrame(final_lambda, index = df1.index.tolist(), columns = df1.index.tolist())
df2.to_excel('Final.xlsx')
