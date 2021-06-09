import numpy as np
from scipy.integrate import ode
import matplotlib.pyplot as plt
import json
from GenerateLambdas import *


def odeSys(t,z,params):
    theta,Lambda = params
    term1 = np.dot(Lambda,z)
    term2 = np.dot(z.T,term1)
    return z*(term1-term2)

def ComputeQ(z,Lambda):
    term1 = np.dot(Lambda,z)
    term2 = np.dot(z.T,term1)
    return term2

class Experiment:
    '''
        Class to define an invasion experiment.
    '''
    def __init__(self):
        self.Community = []
        self.Invader = None


def predict_community(FullLambdaMat, comm, verb=False):

    if isinstance(comm,str):
        with open(comm) as fl:
            commMembers = json.load(fl)
            experiment =  Experiment(commMembers,None)
    elif isinstance(comm,dict):
        experiment =  Experiment(comm["Community"],None)
    elif isinstance(comm,list):
        experiment = Experiment()
        experiment.Community = comm
        experiment.Invader = None
    elif not isinstance(comm,Experiment):
        print("Must provide experiment as list, dict, Experiemnt, or .json file name")
        return None


    LambdaMat = SelectLambdas(comm, FullLambdaMat=FullLambdaMat)

    if LambdaMat.size == 0:
        print("Empty Community")
        return experiment.Invader,False,None,None,None,None,None,None

    if verb:
        print("Found "+str(LambdaMat.shape[0]) + "/" + str(len(experiment.Community)) + " community members.")

    initialComm = 1

    Theta = 1
    numComm = len(LambdaMat)
    z0 = np.ones(numComm)/numComm
    t0 = 0

    community = ode(odeSys).set_integrator('vode')
    community.set_initial_value(z0,t0).set_f_params([Theta,LambdaMat.values])

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
        print("###################################################################################")
        print("Error: zi do not sum to 1", np.sum(list(CommunityEquilibrium.values())))
        CommunityEquilibrium = {}
        print("###################################################################################")
    elif np.min(list(CommunityEquilibrium.values())).round(3) < 0:
        print("###################################################################################")
        print("Error: exists zi<0",np.min(zt[-1]))
        print("###################################################################################")
    elif np.max(list(CommunityEquilibrium.values())).round(3) >1:
        print("###################################################################################")
        print("Error: exists zi > 1",np.max(zt[-1]))
        print("###################################################################################")

    return CommunityEquilibrium
