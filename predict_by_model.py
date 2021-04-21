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


def predict_by_model(experiment,lambdaVersion = "Equilibrium",verb=False,generateLambdas = GenerateLambdasFromExcel,File = "Pairwise_Chemostat.xlsx"):

    if isinstance(experiment,str):
        with open(experiment) as fl:
            ExpDict = json.load(fl)
            experiment =  Experiment(ExpDict["Community"],ExpDict["Invader"])
    elif isinstance(experiment,dict):
        experiment =  Experiment(experiment["Community"],experiment["Invader"])
    elif not isinstance(experiment,Experiment):
        print("Must provide experiment as dict, Experiemnt, or .json file name")
        return None

    if verb:
        print("Generating Lambdas")

    LambdaMat,LambdaInvaderComm,LambdaCommInvader,foundList = generateLambdas(experiment,version = lambdaVersion,File = File)

    if LambdaMat.size == 0:
        print("Empty Community")
        return experiment.Invader,False,None,None,None,None,None,None

    if verb:
        print("Found "+str(LambdaMat.shape[0]) + "/" + str(len(experiment.Community)) + " community members.")


    invader0 = 0.1/LambdaMat.shape[0]
    initialComm = 1
    Theta = 1


    CommunityEquilibriumDict,_ = predict_justComm(LambdaMat,verb  = verb)
    CommunityEquilibrium = np.array([CommunityEquilibriumDict[mic] for mic in LambdaMat.index])

    # print("Community Stability: ", Qt)
    CommStability = ComputeQ(CommunityEquilibrium,LambdaMat.values)

    newLambda = np.append(LambdaMat.values,[LambdaInvaderComm],axis = 0)
    newLambda = np.append(newLambda,np.array([np.append(LambdaCommInvader,0)]).T,axis = 1)


    totalMass = initialComm + invader0
    initialinvader = invader0/totalMass

    z0 = np.append([initialComm*CommunityEquilibrium/totalMass],[initialinvader])

    InvaderQ = ComputeQ(z0,newLambda)

    # print("Community Resistance to "+ ExpDict["Invader"] + ": ", InvaderQ)

    r0 = np.dot(newLambda[-1],z0) - InvaderQ
    # print(ExpDict["Invader"] + " Initial Invasion Rate: ",r0)

    t0 = 0
    invasion = ode(odeSys).set_integrator('lsoda')
    invasion.set_initial_value(z0,t0).set_f_params([Theta,newLambda])
    t = [t0]
    dt = 0.1
    zt = [z0]
    invader = [z0[-1]]

    Qt = []

    deriv_estimate = 1

    if verb:
        print("Simulating Invasion")

    while deriv_estimate>10**(-7) and t[-1]<500:
        sol = invasion.integrate(t[-1] + dt)
        zt += [sol]
        t += [invasion.t]
        Qt += [ComputeQ(sol,newLambda)]
        if len(zt)>2:
            centered_differences = (zt[-1]-zt[-3])/(3*dt)
            deriv_estimate = np.sqrt(np.dot(centered_differences,centered_differences))

    if np.sum(np.array(zt)[-1]).round(3) != 1:
        print("###################################################################################")
        print("Error: zi do not sum to 1",np.sum(zt[-1]))
        print("###################################################################################")
    elif np.min(np.array(zt)[-1]).round(3) < 0:
        print("###################################################################################")
        print("Error: exists zi<0",np.min(zt[-1]))
        print("###################################################################################")
    elif np.max(np.array(zt)[-1]).round(3) >1:
        print("###################################################################################")
        print("Error: exists zi > 1",np.max(zt[-1]))
        print("###################################################################################")

    return experiment.Invader,zt[-1][-1]>initialinvader,np.array(zt),t,CommStability,InvaderQ,r0,foundList

def predict_community(comm,lambdaVersion = "Equilibrium",verb=False,generateLambdas = GenerateLambdasFromExcel,File = "Pairwise_Chemostat.xlsx"):

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

    if verb:
        print("Generating Lambdas")

    LambdaMat,foundList = generateLambdas(experiment,version = lambdaVersion,File = File)

    if LambdaMat.size == 0:
        print("Empty Community")
        return experiment.Invader,False,None,None,None,None,None,None

    if verb:
        print("Found "+str(LambdaMat.shape[0]) + "/" + str(len(experiment.Community)) + " community members.")

    CommunityEquilibrium,fullSim = predict_justComm(LambdaMat,verb = verb)

    return CommunityEquilibrium,foundList



def predict_justComm(LambdaMat,verb=False):

    initialComm = 1

    Theta = 1
    numComm = len(LambdaMat)
    z0 = np.ones(numComm)/numComm
    t0 = 0

    community = ode(odeSys).set_integrator('lsoda')
    community.set_initial_value(z0,t0).set_f_params([Theta,LambdaMat.values])

    t = [t0]
    dt = 0.1
    Z = [np.zeros_like(z0),np.zeros_like(z0),z0]

    deriv_estimate = 1

    if verb:
        print("Computing Community Equilibrium")

    while deriv_estimate>10**(-7) and t[-1]<500:
        Z += [community.integrate(community.t + dt)]
        t += [community.t]
        centered_differences = (Z[-1]-Z[-3])/(3*dt)
        deriv_estimate = np.sqrt(np.dot(centered_differences,centered_differences))


    Z =  np.array(Z[2:])
    CommunityEquilibrium = dict([(LambdaMat.index[i],Z[-1][i]) for i in range(numComm)])
    fullSim = dict([(LambdaMat.index[i],Z.T[i]) for i in range(numComm)])

    if np.sum(list(CommunityEquilibrium.values())).round(3) != 1:
        print("###################################################################################")
        print("Error: zi do not sum to 1",np.sum(zt[-1]))
        print("###################################################################################")
    elif np.min(list(CommunityEquilibrium.values())).round(3) < 0:
        print("###################################################################################")
        print("Error: exists zi<0",np.min(zt[-1]))
        print("###################################################################################")
    elif np.max(list(CommunityEquilibrium.values())).round(3) >1:
        print("###################################################################################")
        print("Error: exists zi > 1",np.max(zt[-1]))
        print("###################################################################################")

    return CommunityEquilibrium,fullSim



if __name__ == "__main__":

    invName,invSuc,zt,t,Qt,InvaderQ,r0 = predict_by_model("Experiment.json")

    print("Community Stability: ", Qt)

    print("Community Resistance to "+ invName + ": ", InvaderQ)

    print(invName + " Initial Invasion Rate: ",r0)

    print("Final Invader:", zt[-1][-1])

    if invSuc:
        print(invName + " Invasion Successful")
    else:
        print(invName + " Invasion Resisted")

    invader = zt.T[-1]

    fig,ax = plt.subplots()
    ax.plot(t,invader)
    ax.set_title("Invasion of " + invName)
    plt.show()
