#!/usr/bin/env python

import statsmodels.api as sm
import pandas as pd
import numpy as np


def ipw(Y, A, Z, data, trim=False):
    """
    Compute the average causal effect E[Y(A=1)] - E[Y(A=0)] via IPW

    Inputs
    ------
    Y: string corresponding variable name of the outcome
    A: string corresponding variable name
    Z: list of variable names included the backdoor adjustment set
    data: pandas dataframe
    trim: boolean determining whether to trim the propensity scores or not

    Return
    ------
    ACE: float corresponding to the causal effect
    """

    running = ""
    for v in Z:
        running = running + "+" + v

    model1 = sm.GLM.from_formula(formula = A + "~" + running, data = data, family = sm.families.Binomial()).fit()


    summation = 0
    skipped = 0
    treatment = data[A]
    outcome = data[Y]
    prop = model1.predict(data)

    dif = ((treatment * outcome)/prop) - (((1 - treatment) * outcome)/(1-prop))

    data["dif"] = dif
    data["prop"] = prop


    if trim:
        data = data[data["prop"] <= 0.9]
        data = data[data["prop"] >= 0.1]



    # for ind in data.index:
    #     treatment = data[A][ind] #A is a binary variable
    #     outcome = data[Y][ind]
    #     prop = model1.predict(data.loc[ind]).loc[ind]
    #     #outcome/prop.loc[ind]
    #
    #     dif = ((treatment * outcome)/prop) - (((1 - treatment) * outcome)/(1-prop))
    #
    #     if trim:
    #         if (prop < 0.1 or prop > 0.9):
    #             skipped += 1
    #             dif = 0
    #
    #     summation += dif


    # avg =  / (len(data.index) - skipped)

    return data["dif"].mean()



def augmented_ipw(Y, A, Z, data, trim=False):
    """
    Compute the average causal effect E[Y(A=1)] - E[Y(A=0)] via AIPW

    Inputs
    ------
    Y: string corresponding variable name of the outcome
    A: string corresponding variable name
    Z: list of variable names included the backdoor adjustment set
    data: pandas dataframe
    trim: boolean determining whether to trim the propensity scores or not

    Return
    ------
    ACE: float corresponding to the causal effect
    """

    running = ""
    for v in Z:
        running = running + "+" + v

    model1 = sm.GLM.from_formula(formula = A + "~" + running, data = data, family = sm.families.Binomial()).fit()
    model2 = sm.GLM.from_formula(formula = Y + "~" + A + running, data =  data, family = sm.families.Gaussian()).fit()

    # summation = 

    dataZero = data.copy(deep = True)
    dataOne = data.copy(deep = True)

    dataZero[A] = 0
    dataOne[A] = 1
    # counter = 0

    treatment = data[A] #A is a binary variable
    outcome = data[Y]
    prop = model1.predict(data)

    rowOne = dataOne

    one = ((treatment/prop) * (outcome - model2.predict(data))) + model2.predict(rowOne)

    rowZero = dataZero

    two = (((1 - treatment)/(1-prop)) * (outcome - model2.predict(data))) + model2.predict(rowZero)

    dif = one - two

    data["dif"] = dif
    data["prop"] = prop

    if trim:
        data = data[data["prop"] <= 0.9]
        data = data[data["prop"] >= 0.1]

    return data["dif"].mean()

    #
    # for ind in data.index:
    #     treatment = data[A][ind] #A is a binary variable
    #     outcome = data[Y][ind]
    #     prop = model1.predict(data.loc[ind]).loc[ind]
    #     #outcome/prop.loc[ind]
    #
    #     if trim:
    #         if not (prop < 0.1 or prop > 0.9):
    #             rowOne = dataOne.loc[ind]
    #
    #             one = ((treatment/prop) * (outcome - model2.predict(data.loc[ind]).loc[ind])) + model2.predict(rowOne).loc[ind]
    #
    #             rowZero = dataZero.loc[ind]
    #             two = (((1 - treatment)/(1-prop)) * (outcome - model2.predict(data.loc[ind]).loc[ind])) + model2.predict(rowZero).loc[ind]
    #
    #             dif = one - two
    #             summation += dif
    #             counter += 1
    #         else:
    #             continue
    #     else:
    #             rowOne = dataOne.loc[ind]
    #
    #             one = ((treatment/prop) * (outcome - model2.predict(data.loc[ind]).loc[ind])) + model2.predict(rowOne).loc[ind]
    #
    #             rowZero = dataZero.loc[ind]
    #             two = (((1 - treatment)/(1-prop)) * (outcome - model2.predict(data.loc[ind]).loc[ind])) + model2.predict(rowZero).loc[ind]
    #
    #             dif = one - two
    #             summation += dif
    #             counter += 1
    #
    #
    # avg = summation/counter
    # return avg



def compute_confidence_intervalsipw(Y, A, Z, data, method_name, num_bootstraps=200, alpha=0.05):
    """
    Compute confidence intervals for IPW or AIPW (potentially with trimming) via bootstrap.
    The input method_name can be used to decide how to compute the confidence intervals.

    Returns tuple (q_low, q_up) for the lower and upper quantiles of the confidence interval.
    """

    Ql = alpha/2
    Qu = 1 - alpha/2
    estimates = []

    if (method_name == "augmented_ipw"):
        for i in range(num_bootstraps):
            resamp = data.sample(n = len(data.index), replace = True, ignore_index = True)
            estimates.append(augmented_ipw(Y, A, Z, resamp, trim = False))
    elif (method_name == "ipw"):
        for i in range(num_bootstraps):
            resamp = data.sample(n = len(data.index), replace = True, ignore_index = True)
            estimates.append(ipw(Y, A, Z, resamp, trim = False))
    elif (method_name == "augmented_ipwtrim"):
        for i in range(num_bootstraps):
            resamp = data.sample(n = len(data.index), replace = True, ignore_index = True)
            estimates.append(augmented_ipw(Y, A, Z, resamp, trim = True))
    elif (method_name == "ipwtrim"):
        for i in range(num_bootstraps):
            resamp = data.sample(n = len(data.index), replace = True, ignore_index = True)
            estimates.append(ipw(Y, A, Z, resamp, trim = True))
    else:
        return None

    q_low, q_up = np.quantile(estimates, Ql), np.quantile(estimates, Qu)
    return q_low, q_up


def main():
    """
    Add code to the main function as needed to compute the desired quantities.
    """

    np.random.seed(100)
    nsw_randomized = pd.read_csv("nsw_randomized.txt")
    nsw_observational = pd.read_csv("nsw_observational.txt")
    # define some backdoor sets
    Z = ["age", "educ", "black", "hisp", "marr", "nodegree", "re74", "re75"]

    # point estimate and CIs using observational data and IPW
    print("ACE using IPW")
    print(ipw("re78", "treat", Z, nsw_observational, trim=False))
    print(compute_confidence_intervalsipw("re78", "treat", Z, nsw_observational, "ipw", num_bootstraps=200, alpha=0.05))
    #point estimate and CIs using observational data and IPW with trimming=
    print("ACE using IPW with trimming")
    print(ipw("re78", "treat", Z, nsw_observational, trim=True))
    print(compute_confidence_intervalsipw("re78", "treat", Z, nsw_observational, "ipwtrim", num_bootstraps=200, alpha=0.05))
    #point estimate and CIs using observational data and AIPW
    # quit()
    print("ACE using AIPW (without trimming)")
    print(augmented_ipw("re78", "treat", Z, nsw_observational, trim=False))
    print(compute_confidence_intervalsipw("re78", "treat", Z, nsw_observational, "augmented_ipw", num_bootstraps=200, alpha=0.05))
    #point estimate and CIs using observational data and AIPW
    print("ACE using AIPW (with trimming)")
    print(augmented_ipw("re78", "treat", Z, nsw_observational, trim=True))
    print(compute_confidence_intervalsipw("re78", "treat", Z, nsw_observational, "augmented_ipwtrim", num_bootstraps=200, alpha=0.05))

if __name__ == "__main__":
    main()
