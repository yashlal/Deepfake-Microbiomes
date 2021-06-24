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
target_vector = anp.array(list((predict_community(LamMat, comm = RADF.index.tolist(), verb=True).values())))
print(target_vector)

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

cost = JSD
C_grad = autograd.grad(cost)

init_z = (anp.ones(3)/3)
init_sens = anp.zeros(27)
init_lam = anp.array([[0, 1, 1], [6, 0, 1], [3.5, 1.1, 0]])
time = np.linspace(0,50,500)
Z0 = anp.concatenate([init_z, init_sens])

loss_values = []
learning_rate = 1


while epoch<=1000:
    sol = BlackBox(odeSys, y0 = Z0, t = time, args = tuple([init_lam]))
    comm = anp.array([x.round(7) for x in sol[-1][0:3]])
    sensitivity = anp.reshape(sol[-1][-27:], (3,3,3))

    dist_grad = C_grad(comm, target_vector)
    step = sensitivity@dist_grad

    dist = cost(comm, target_vector)
    loss_values.append(dist)
    print('Epoch Number ' + str(epoch) + ':' + str(dist))
    # print(comm)

    init_lam -= learning_rate*step

    epoch += 1

# plt.plot(loss_values)
plt.plot(loss_values)
plt.show()
