import numpy as np
import pandas as pd
from scipy.integrate import ode
import json
from scipy.integrate import odeint
from numba import njit
import time

@njit()
def odeSys(t, zeta, Lambda):
    z = np.exp(zeta)
    term1 = np.dot(Lambda,z)
    term2 = np.dot(z.T,term1)
    dzetadt = term1-term2
    return dzetadt

def wrapper(LambdaMat):
    return lambda t,zeta: odeSys(t, zeta, LambdaMat)

def predict_community_fullnp(LambdaMat, comm, verb=False):

    #this wrapper allows NJIT with odesys
    f = wrapper(LambdaMat)

    numComm = len(comm)
    zeta0 = np.log(np.ones(numComm)/numComm)
    t0 = 0

    community = ode(f).set_integrator('lsoda')
    community.set_initial_value(zeta0,t0)

    t = [t0]
    dt = 0.1
    Z = [np.zeros_like(zeta0), np.zeros_like(zeta0), np.exp(zeta0)]

    deriv_estimate = 1

    if verb:
        print("Computing Community Equilibrium...")

    while deriv_estimate>10**(-7) and t[-1]<500:
        Z += [np.exp(community.integrate(community.t + dt))]
        t += [community.t]
        centered_differences = (Z[-1]-Z[-3])/(3*dt)
        deriv_estimate = np.linalg.norm(centered_differences)
        #deriv_estimate = np.sqrt(np.dot(centered_differences,centered_differences))


    cm = Z[2:]

    if np.sum(cm).round(3) != 1:
        print("Error: zi do not sum to 1", np.sum(cm))
        # print(cm)
        # df = pd.DataFrame(Z, columns=comm)
        # df.to_excel('ZProblem.xlsx')
        # df2 = pd.DataFrame(LambdaMat, index=comm, columns=comm)
        # df2.to_excel('LamProblem.xlsx')
        cm = np.array([])
        return cm

    elif np.min(cm).round(3) < 0:
        print("Error: exists zi<0", np.min(cm).round(3))
        # print(cm)
        # df = pd.DataFrame(Z, columns=comm)
        # df.to_excel('ZProblem.xlsx')
        # df2 = pd.DataFrame(LambdaMat, index=comm, columns=comm)
        # df2.to_excel('LamProblem.xlsx')
        cm = np.array([])
        return cm

    return cm
