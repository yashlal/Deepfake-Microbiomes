import numpy as np
from scipy.integrate import ode
import matplotlib.pyplot as plt
import json
from GenerateLambdas import *
from scipy.integrate import odeint
import matplotlib.pyplot as plt

def odeSys(t,z,Lambda):
    time = t
    term1 = np.dot(Lambda,z)
    term2 = np.dot(z.T,term1)
    term3 = term1-term2
    dzdt =  np.multiply(z, term3)
    return dzdt

def predict_community(FullLambdaMat, comm, verb=False, full=False):

    if full == False:
        LambdaMat = SelectLambdas(comm, FullLambdaMat=FullLambdaMat)
    else:
        LambdaMat = FullLambdaMat

    if LambdaMat.size == 0:
        print("Empty Community")

    if verb:
        print("Found "+str(LambdaMat.shape[0]) + "/" + str(len(comm)) + " community members.")

    numComm = len(LambdaMat)
    z0 = np.ones(numComm)/numComm
    t0 = 0

    community = ode(odeSys).set_integrator('lsoda')
    community.set_initial_value(z0,t0).set_f_params(LambdaMat.values)

    t = [t0]
    dt = 0.1
    Z = [np.zeros_like(z0),np.zeros_like(z0),z0]

    deriv_estimate = 1

    if verb:
        print("Computing Community Equilibrium...")

    while deriv_estimate>10**(-7) and t[-1]<500:
        Z += [community.integrate(community.t + dt)]
        t += [community.t]
        centered_differences = (Z[-1]-Z[-3])/(3*dt)
        deriv_estimate = np.sqrt(np.dot(centered_differences,centered_differences))

    Z =  np.array(Z[2:])
    CommunityEquilibrium = dict([(LambdaMat.index[i],Z[-1][i]) for i in range(numComm)])

    if np.sum(list(CommunityEquilibrium.values())).round(3) != 1:
        print("Error: zi do not sum to 1", np.sum(list(CommunityEquilibrium.values())))
        CommunityEquilibrium = {}
        df = pd.DataFrame(Z, columns=comm)
        df.to_excel('ZProblem.xlsx')
        print(len(term1list))
    elif np.min(list(CommunityEquilibrium.values())).round(3) < 0:
        print("Error: exists zi<0", np.min(list(CommunityEquilibrium.values())).round(3))
        CommunityEquilibrium = {}

    return CommunityEquilibrium

def predict_community_fullnp(LambdaMat, comm, verb=False):
    numComm = len(comm)
    z0 = np.ones(numComm)/numComm
    t0 = 0

    community = ode(odeSys).set_integrator('lsoda')
    community.set_initial_value(z0,t0).set_f_params(LambdaMat)

    t = [t0]
    dt = 0.1
    Z = [np.zeros_like(z0), np.zeros_like(z0), z0]

    deriv_estimate = 1

    if verb:
        print("Computing Community Equilibrium...")

    while deriv_estimate>10**(-7) and t[-1]<500:
        Z += [community.integrate(community.t + dt)]
        t += [community.t]
        centered_differences = (Z[-1]-Z[-3])/(3*dt)
        deriv_estimate = np.sqrt(np.dot(centered_differences,centered_differences))

    CommunityEquilibrium = dict([(comm[i],Z[-1][i]) for i in range(numComm)])

    if np.sum(list(CommunityEquilibrium.values())).round(3) != 1:
        print("Error: zi do not sum to 1", np.sum(list(CommunityEquilibrium.values())))
        CommunityEquilibrium = {}
    elif np.min(list(CommunityEquilibrium.values())).round(3) < 0:
        print("Error: exists zi<0", np.min(list(CommunityEquilibrium.values())).round(3))
        CommunityEquilibrium = {}

    return list(CommunityEquilibrium.values())
