import numpy as np
import pandas as pd


def GenerateLambdasFromExcel(Experiment,File = "Pairwise_Chemostat.xlsx",version = "Equilibrium",mu = 1):

    Community = Experiment.Community
    Invader = Experiment.Invader

    RelativeAbundanceAll = pd.read_excel(File,sheet_name = "Relative_Abundance",index_col = 0)

    CommunityIndex = []
    foundList = []
    for CM in Community:
        if len(CM):
            try:
                CommunityIndex += [RelativeAbundanceAll.index[np.where(RelativeAbundanceAll.index.str.find(CM)>=0)][0]]
                foundList += [CM]
            except IndexError:
                print(CM + " not found")


    if Invader != None:
        try:
            InvaderIndex = RelativeAbundanceAll.index[np.where(RelativeAbundanceAll.index.str.find(Invader)>=0)][0]
        except IndexError:
            print("Invader Not Found")
            return None

        CommunityIndex = [cm for cm in np.unique(CommunityIndex) if cm != InvaderIndex]


        #print(RelativeAbundanceAll.index[np.where(RelativeAbundanceAll.index.str.find("intestini")>0)])

        ###Get a list of community members.
        All = list(CommunityIndex) + [InvaderIndex]

    else:
        CommunityIndex = np.unique(CommunityIndex)

        #print(RelativeAbundanceAll.index[np.where(RelativeAbundanceAll.index.str.find("intestini")>0)])

        ###Get a list of community members.
        All = list(CommunityIndex)


    RelativeAbundance = RelativeAbundanceAll.loc[All,All]

    ## \lambda_i^j  = \alpha_{ji} - \alpha_{jj} -  \mu*(\alpha_{ij} - \alpha_{ji})
    ### \mu = 1/[(R_0 - 1)k]
    #### k -> mean interaction value. No idea how to estimate.
    #### R_0 Basic reproduction number. Similarly no idea, except that R_0>1

    mean_interaction = mu

    if version == "Equilibrium":
        AllLambdas = pd.DataFrame(columns = RelativeAbundance.columns, index = RelativeAbundance.index)
        for i in range(len(AllLambdas.index)):
            for j in range(i+1):
                if i==j:
                    AllLambdas.iloc[i,j] = 0
                elif RelativeAbundance.iloc[i,j] == 0:
                    AllLambdas.iloc[i,j] = -mean_interaction
                    AllLambdas.iloc[j,i] = mean_interaction
                elif RelativeAbundance.iloc[j,i] == 0:
                    AllLambdas.iloc[i,j] = mean_interaction
                    AllLambdas.iloc[j,i] = -mean_interaction
                else:
                    AllLambdas.iloc[i,j] = mean_interaction*RelativeAbundance.iloc[i,j]/(1-RelativeAbundance.iloc[i,j])
                    AllLambdas.iloc[j,i] = mean_interaction

    elif version == "LogRatio":
        TotalMassAll = pd.read_excel(File,sheet_name = "Total_Biomass",index_col = 0)

        TotalMass = TotalMassAll.loc[All,All]
        GrowthMass = RelativeAbundance*TotalMass
        GrowthMass.replace(to_replace = 0, value = 0.001, inplace = True)
        Alphas = np.log((GrowthMass.T/np.diag(GrowthMass)).T)
        AllLambdas = Alphas.T - np.diag(Alphas) - mu*(Alphas  - Alphas.T)

    elif version == "Difference":
        TotalMassAll = pd.read_excel(File,sheet_name = "Total_Biomass",index_col = 0)

        TotalMass = TotalMassAll.loc[All,All]
        GrowthMass = RelativeAbundance*TotalMass
        Alphas = (GrowthMass.T - np.diag(GrowthMass)).T
        AllLambdas = Alphas.T - np.diag(Alphas) - mu*(Alphas  - Alphas.T)




    CommLambdas = AllLambdas.loc[CommunityIndex,CommunityIndex]
    if Invader != None:
        LambdaInvaderComm = AllLambdas.loc[InvaderIndex].values[:-1]
        LambdaCommInvader = AllLambdas.loc[:,InvaderIndex].values[:-1]

        return CommLambdas,LambdaInvaderComm,LambdaCommInvader,foundList
    else:
        return CommLambdas,foundList


def GenerateLambdasFromExcelAllPairs(File,version = "Equilibrium",mu = 1):

    RelativeAbundanceAll = pd.read_excel(File,sheet_name = "Relative_Abundance",index_col = 0)
    All = RelativeAbundanceAll.index

    ## \lambda_i^j  = \alpha_{ji} - \alpha_{jj} -  \mu*(\alpha_{ij} - \alpha_{ji})
    ### \mu = 1/[(R_0 - 1)k]
    #### k -> mean interaction value. No idea how to estimate.
    #### R_0 Basic reproduction number. Similarly no idea, except that R_0>1

    mean_interaction = mu

    if version == "Equilibrium":
        AllLambdas = pd.DataFrame(columns = RelativeAbundanceAll.columns, index = RelativeAbundanceAll.index)
        for i in range(len(AllLambdas.index)):
            for j in range(i+1):
                if i==j:
                    AllLambdas.iloc[i,j] = 0
                elif RelativeAbundanceAll.iloc[i,j] == 0:
                    AllLambdas.iloc[i,j] = -mean_interaction
                    AllLambdas.iloc[j,i] = mean_interaction
                elif RelativeAbundanceAll.iloc[j,i] == 0:
                    AllLambdas.iloc[i,j] = mean_interaction
                    AllLambdas.iloc[j,i] = -mean_interaction
                else:
                    AllLambdas.iloc[i,j] = mean_interaction*RelativeAbundanceAll.iloc[i,j]/(1-RelativeAbundanceAll.iloc[i,j])
                    AllLambdas.iloc[j,i] = mean_interaction

    elif version == "LogRatio":

        TotalMassAll = pd.read_excel(File,sheet_name = "Total_Biomass",index_col = 0)
        GrowthMass = RelativeAbundanceAll*TotalMassAll
        GrowthMass.replace(to_replace = 0, value = 0.001, inplace = True)
        Alphas = np.log((GrowthMass.T/np.diag(GrowthMass)).T)
        AllLambdas = Alphas.T - np.diag(Alphas) - mu*(Alphas  - Alphas.T)

    elif version == "Difference":
        TotalMassAll = pd.read_excel(File,sheet_name = "Total_Biomass",index_col = 0)
        GrowthMass = RelativeAbundanceAll*TotalMassAll
        Alphas = (GrowthMass.T - np.diag(GrowthMass)).T
        AllLambdas = Alphas.T - np.diag(Alphas) - mu*(Alphas  - Alphas.T)

    return AllLambdas
