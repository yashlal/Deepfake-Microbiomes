import autograd
from autograd.builtins import tuple
import autograd.numpy as np

#Import ode solver and rename as BlackBox for consistency with blog
from scipy.integrate import odeint as BlackBox
import matplotlib.pyplot as plt

def f(y,t,theta):
    '''Function describing dynamics of the system'''
    S,I = y
    ds = -theta*S*I
    di = theta*S*I - I

    return np.array([ds,di])

#Jacobian wrt y
J = autograd.jacobian(f,argnum=0)
#Gradient wrt theta
grad_f_theta = autograd.jacobian(f,argnum=2)

def ODESYS(Y,t,theta):

    #Y will be length 4.
    #Y[0], Y[1] are the ODEs
    #Y[2], Y[3] are the sensitivities

    #ODE
    dy_dt = f(Y[0:2],t,theta)
    #Sensitivities
    grad_y_theta = J(Y[:2],t,theta)@Y[-2::] + grad_f_theta(Y[:2],t,theta)

    return np.concatenate([dy_dt,grad_y_theta])

def Cost(y_obs):
    def cost(Y):
        '''Squared Error Loss'''
        n = y_obs.shape[0]
        err = np.linalg.norm(y_obs - Y, 2, axis = 1)

        return np.sum(err)/n

    return cost

np.random.seed(19920908)
## Generate Data
#Initial Condition
Y0 = np.array([0.99,0.01, 0.0, 0.0])
#Space to compute solutions
t = np.linspace(0,5,101)
#True param value
theta = 5.5

sol = BlackBox(ODESYS, y0 = Y0, t = t, args = tuple([theta]))

#Corupt the observations with noise
y_obs = sol[:,:2] + np.random.normal(0,0.05,size = sol[:,:2].shape)

plt.scatter(t,y_obs[:,0], marker = '.', alpha = 0.5, label = 'S')
plt.scatter(t,y_obs[:,1], marker = '.', alpha = 0.5, label = 'I')


plt.legend()

theta_iter = 1.5
cost = Cost(y_obs[:,:2])
grad_C = autograd.grad(cost)

maxiter = 100
learning_rate = 1 #Big steps
for i in range(maxiter):

    sol = BlackBox(ODESYS,y0 = Y0, t = t, args = tuple([theta_iter]))

    Y = sol[:,:2]

    theta_iter -=learning_rate*(grad_C(Y)*sol[:,-2:]).sum()

    if i%10==0:
        print(theta_iter)

print('YYYEAAAAHHH!')

sol = BlackBox(ODESYS, y0 = Y0, t = t, args = tuple([theta_iter]))
true_sol = BlackBox(ODESYS, y0 = Y0, t = t, args = tuple([theta]))


plt.plot(t,sol[:,0], label = 'S', color = 'C0', linewidth = 5)
plt.plot(t,sol[:,1], label = 'I', color = 'C1', linewidth = 5)

plt.scatter(t,y_obs[:,0], marker = '.', alpha = 0.5)
plt.scatter(t,y_obs[:,1], marker = '.', alpha = 0.5)


plt.plot(t,true_sol[:,0], label = 'Estimated ', color = 'k')
plt.plot(t,true_sol[:,1], color = 'k')

plt.legend()
plt.show()
