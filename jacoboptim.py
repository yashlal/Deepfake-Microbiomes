import autograd
import autograd.numpy as anp
import pandas as pd
from modules import *
from predict_by_model import *
from GenerateLambdas import *
from autograd.builtins import tuple
from scipy.integrate import odeint as BlackBox
import matplotlib.pyplot as plt
import math

epoch = 1
dim1 = 3
dim2 = 27
RADF = pd.read_excel('JacobTest.xlsx', index_col=0)
LamMat = GenerateLambdasFromRADF(RADF)
lam_vec = anp.array(LamMat.values.tolist())
# target_vector = anp.array(list((predict_community(LamMat, comm = RADF.index.tolist(), verb=True).values())))
# print(target_vector)

def f(z, t, Lambda):
    term1 = anp.dot(Lambda,z)
    term2 = anp.dot(z,term1)
    term3 = term1-term2
    dzdt =  anp.multiply(z, term3)
    return dzdt

J = autograd.jacobian(f, argnum=0)
grad_f_theta = autograd.jacobian(f, argnum=2)

def odeSys(z, t, Lambda):
    z_spec = z[0:dim1]
    sensitivity = anp.reshape(z[-dim2:], (3,3,3))
    dzdt = f(z_spec,t,Lambda)
    sensitivity = J(z_spec,t,Lambda)@sensitivity + grad_f_theta(z_spec,t,Lambda)
    return anp.concatenate([dzdt, anp.reshape(sensitivity, (dim2,))])

def Cost(y_obs):
    def cost(Y):
        '''Squared Error Loss'''
        n = y_obs.shape[0]
        er=0
        for i in range(n):
            er = er + JSD(Y[i], y_obs[i])

        return er/n

    return cost

init_z = (anp.ones(3)/3)
init_sens = anp.zeros(27)
init_lam = anp.array([[0, 1, 1], [6, 0, 1], [3.5, 1.1, 0]])
time = np.linspace(0,50,500)
Z0 = anp.concatenate([init_z, init_sens])

loss_values = []
learning_rate = 1

true_sol = BlackBox(odeSys, y0 = Z0, t = time, args = tuple([lam_vec]))
target = true_sol[:, :3]
cost = Cost(target)
C_grad = autograd.grad(cost)

# sol = BlackBox(odeSys, y0 = Z0, t = time, args = tuple([init_lam]))
# comm = sol[:, :3]
# print(anp.nan_to_num(C_grad(comm)))

learning_rate = 0.1
while epoch<=10000:
    sol = BlackBox(odeSys, y0 = Z0, t = time, args = tuple([init_lam]))
    comm = sol[:, :3]
    sensitivity = anp.array([anp.reshape(x, (3,3,3)) for x in sol[:, -27:]])

    dist = cost(comm)
    costgrad = anp.nan_to_num(C_grad(comm))
    print('Epoch Number ' + str(epoch) + ':' + str(dist))
    loss_values.append(dist)
    step = []
    for i in range(sensitivity.shape[0]):
        step.append(sensitivity[i]@costgrad[i])
    step = anp.array(step).sum(0)

    init_lam -= learning_rate*step

    epoch += 1

plt.plot(loss_values)
plt.show()
