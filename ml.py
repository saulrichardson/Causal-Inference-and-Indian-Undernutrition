import statsmodels.api as sm
import pandas as pd
import numpy as np
from scipy.special import expit
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
import math


def backdoor_adjustment(Y, A, Z, data):
    """
    Compute the average causal effect E[Y(A=1)] - E[Y(A=0)] via backdoor adjustment

    Inputs
    ------
    Y: string corresponding variable name of the outcome
    A: string corresponding variable name
    Z: list of variable names to adjust for
    data: pandas dataframe

    Return
    ------
    ACE: float corresponding to the causal effect
    """

    formula = Y + "~" + A
    if len(Z) > 0:
        formula += " + " + "+".join(Z)

    model = sm.GLM.from_formula(formula=formula, data=data, family=sm.families.Gaussian()).fit()
    data_A0 = data.copy()
    data_A1 = data.copy()
    data_A0[A] = 0
    data_A1[A] = 1
    return(np.mean(model.predict(data_A1)) - np.mean(model.predict(data_A0)))

def get_numpy_matrix(data, variables):
    """
    Takes a pandas dataframe and a list of variable names, and returns
    just the raw matrix for those specific variables
    """


    matrix = data[variables].to_numpy()

    # if there's only one variable, ensure we return a matrix with one column
    # rather than just a column vector
    if len(variables) == 1:
        return matrix.reshape(len(data),)
    return matrix


def backdoor_ML(Y, A, Z, data):
    """
    Compute the average causal effect E[Y(A=1)] - E[Y(A=0)] via backdoor adjustment with random forests.

    Inputs
    ------
    Y: string corresponding variable name of the outcome
    A: string corresponding variable name
    Z: list of variable names to adjust for
    data: pandas dataframe

    Return
    ------
    ACE: float corresponding to the causal effect
    """

    # some starter code to get the matrix for doing the regression
    # in the right shape
    data = data.sample(frac=0.1)
    outcome = get_numpy_matrix(data, [Y])
    predictors = get_numpy_matrix(data, [A] + Z)

    # use a RandomForestRegressor with bootstrap=False
    model = RandomForestRegressor(bootstrap = False).fit(predictors, outcome)

    DataZ = data.copy()
    DataO = data.copy()
    DataZ[A] = 0
    DataO[A] = 1
    jj = [A] + Z
    Z = get_numpy_matrix(DataZ, jj)

    O = get_numpy_matrix(DataO, jj)

    return(np.mean(model.predict(O)) - np.mean(model.predict(Z)))


def aipw_ML(Y, A, Z, dataog):
    """
    Compute the average causal effect E[Y(A=1)] - E[Y(A=0)] via augmented IPW with random forests.

    Inputs
    ------
    Y: string corresponding variable name of the outcome
    A: string corresponding variable name
    Z: list of variable names to adjust for
    data: pandas dataframe

    Return
    ------
    ACE: float corresponding to the causal effect
    """

    # some starter code to get the matrix for doing the regressions
    # in the right shape
    data = dataog.copy()
    treatment = get_numpy_matrix(data, [A])
    predictors_a = get_numpy_matrix(data, Z)

    # use a RandomForestClassifier with bootstrap=False for fitting a model for p(A|Z)
    model = RandomForestClassifier(bootstrap = False)
    model.fit(predictors_a, treatment)

    # use p(A|Z) model to predict propensity scores and get IPW weights
    pscores = model.predict_proba(predictors_a)
    # pscores = 1 * 1/pscores
    probO = pscores[:,1]
    probZ = pscores[:,0]
    data["scores"] = probO
    data = data[data["scores"] <= 0.9]
    data = data[data["scores"] >= 0.1]
    data["weights"] = 1 / data["scores"]
    # print(data["weights"])
    # data = data[data["scores"] < math.inf]  #need to trim as some prop scores were inf and resulted in error

    # fit a RandomForestRegressor model for E[Y|A, Z, W] where W are the IPW weights
    dd = Z + ["weights"] + [A]
    # dd = Z  + [A]
    outcome = get_numpy_matrix(data, [Y])
    predictors = get_numpy_matrix(data, dd)
    # outcome.astype(float)
    # predictors.astype(float)


    model = RandomForestRegressor(bootstrap = False).fit(predictors, outcome)


    # get predictions using datasets where A=1 and A=0
    DataZ = data.copy()
    DataO = data.copy()
    DataZ[A] = 0
    DataZ["scores"] =  DataZ["scores"] + (-1)
    DataZ["weights"] = 1 / DataZ["scores"]
    DataO[A] = 1
    # already set above
    jj = [A] + Z + ["scores"]

    Z = get_numpy_matrix(DataZ, jj)

    O = get_numpy_matrix(DataO, jj)

    return(np.mean(model.predict(O)) - np.mean(model.predict(Z)))




def backdoor_mean(Y, A, Z, value, data):
    """
    Compute the counterfactual mean E[Y(a)] for a given value of a via backdoor adjustment

    Inputs
    ------
    Y: string corresponding variable name of the outcome
    A: string corresponding variable name
    Z: list of variable names to adjust for
    value: float corresponding value to set A to
    data: pandas dataframe

    Return
    ------
    ACE: float corresponding to the causal effect
    """

    formula = Y + "~" + A
    if len(Z) > 0:
        formula += " + " + "+".join(Z)

    model = sm.GLM.from_formula(formula=formula, data=data, family=sm.families.Gaussian()).fit()
    data_a = data.copy()
    data_a[A] = value
    return np.mean(model.predict(data_a))

def compute_confidence_intervals(Y, A, Z, data, method_name, num_bootstraps=100, alpha=0.05, value=None):
    """
    Compute confidence intervals for backdoor adjustment via bootstrap

    Returns tuple (q_low, q_up) for the lower and upper quantiles of the confidence interval.
    """

    Ql = alpha/2
    Qu = 1 - alpha/2
    estimates = []

    for i in range(num_bootstraps):

        # resample the data with replacement
        data_sampled = data.sample(len(data), replace=True)
        data_sampled.reset_index(drop=True, inplace=True)

        # add estimate from resampled data
        if method_name == "backdoor":
            estimates.append(backdoor_adjustment(Y, A, Z, data_sampled))

        elif method_name == "backdoor_ML":
            estimates.append(backdoor_ML(Y, A, Z, data_sampled))

        elif method_name == "aipw_ML":
            estimates.append(aipw_ML(Y, A, Z, data_sampled))

        elif method_name == "backdoor_mean":
            estimates.append(backdoor_mean(Y, A, Z, value, data_sampled))

        else:
            print("Invalid method")
            estimates.append(1)

    # calculate the quantiles
    quantiles = np.quantile(estimates, q=[Ql, Qu])
    q_low = quantiles[0]
    q_up = quantiles[1]

    return q_low, q_up
