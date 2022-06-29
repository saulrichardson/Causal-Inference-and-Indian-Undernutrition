#!/usr/bin/env python

import statsmodels.api as sm
import pandas as pd
import numpy as np

def backdoor_adjustment(Y, A, Z, data):
    """
    Compute the average causal effect E[Y(A=1)] - E[Y(A=0)] via backdoor adjustment

    Inputs
    ------
    Y: string corresponding variable name of the outcome
    A: string corresponding variable name
    formula: list of variable names included the backdoor adjustment set
    data: pandas dataframe

    Return
    ------
    ACE: float corresponding to the causal effect
    """
    running = ""
    for v in Z:
        running = running + "+" + v


    modelOG = sm.GLM.from_formula(formula = Y + "~" + A + running, data =  data, family = sm.families.Gaussian()).fit()


    dataZero = data.copy(deep = True)
    dataOne = data.copy(deep = True)


    dataZero[A] = 0
    dataOne[A] = 1


    #go thru dataZero and predit each Y given A and covariates
    #then find average prediction in both datasets
    zero = modelOG.predict(dataZero).sum()/len(dataZero.index)
    one = modelOG.predict(dataOne).sum()/len(dataZero.index)


    #ACE1 = modelOne.params[A]
    ACE2 = one - zero


    return ACE2


def compute_confidence_intervalsback(Y, A, Z, data, num_bootstraps=200, alpha=0.05):
    """
    Compute confidence intervals for backdoor adjustment via bootstrap

    Returns tuple (q_low, q_up) for the lower and upper quantiles of the confidence interval.
    """

    Ql = alpha/2
    Qu = 1 - alpha/2
    estimates = []

    for i in range(num_bootstraps):
        resamp = data.sample(n = len(data.index), replace = True)
        estimates.append(backdoor_adjustment(Y, A, Z, resamp))

    q_low, q_up = np.quantile(estimates, Ql), np.quantile(estimates, Qu)
    return q_low, q_up


def main():
    """
    Add code to the main function as needed to compute the desired quantities.
    """
    cov = ["age","educ","black","hisp","marr","nodegree","re74","re75"]
    int = ["black*nodegree", "black*treat", "black*educ", "hisp*nodegree", "hisp*treat", "hisp*educ", "treat*nodegree", "treat*educ"]

    np.random.seed(0)

    nsw_randomized = pd.read_csv("nsw_randomized.txt")
    nsw_observational = pd.read_csv("nsw_observational.txt")


    # point estimate and CIs from the randomized trial
    print("ACE from randomized trial")
    print(backdoor_adjustment("re78", "treat", [], nsw_randomized))
    print(compute_confidence_intervals("re78", "treat", [], nsw_randomized))

    # point estimate and CIs from unadjusted observational data
    print("ACE without adjustment in observational data")
    print(backdoor_adjustment("re78", "treat", [], nsw_observational))
    print(compute_confidence_intervals("re78", "treat", [], nsw_observational))

    # point estimate and CIs using observational data and linear regression
    print("ACE with adjustment in observational data")
    print(backdoor_adjustment("re78", "treat", cov, nsw_observational))
    print(compute_confidence_intervals("re78", "treat", cov, nsw_observational))


    #point estimate and CIs using observational data and nonlinear regression
    print("ACE with adjustment in a non-linear regression")
    print(backdoor_adjustment("re78", "treat", cov + int, nsw_observational))
    print(compute_confidence_intervals("re78", "treat", cov + int, nsw_observational))

if __name__ == "__main__":
    main()
