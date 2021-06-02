import numpy as np
import pandas as pd


def GenerateLambdasFromRADF(RA_df, mu = 1):

    ## \lambda_i^j  = \alpha_{ji} - \alpha_{jj} -  \mu*(\alpha_{ij} - \alpha_{ji})
    ### \mu = 1/[(R_0 - 1)k]
    #### k -> mean interaction value. No idea how to estimate.
    #### R_0 Basic reproduction number. Similarly no idea, except that R_0>1

    mean_interaction = mu

    AllLambdas = pd.DataFrame(columns = RA_df.columns, index = RA_df.index)
    for i in range(len(AllLambdas.index)):
        for j in range(i+1):
            if i==j:
                AllLambdas.iloc[i,j] = 0
            elif RA_df.iloc[i,j] == 0:
                AllLambdas.iloc[i,j] = -mean_interaction
                AllLambdas.iloc[j,i] = mean_interaction
            elif RA_df.iloc[j,i] == 0:
                AllLambdas.iloc[i,j] = mean_interaction
                AllLambdas.iloc[j,i] = -mean_interaction
            else:
                AllLambdas.iloc[i,j] = mean_interaction*RA_df.iloc[i,j]/(1-RA_df.iloc[i,j])
                AllLambdas.iloc[j,i] = mean_interaction

    return AllLambdas

def SelectLambdas(species, FullLambdaMat):
    return FullLambdaMat.loc[species, species]
