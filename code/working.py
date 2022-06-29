from hw4copy import *
from hw3copy import *
from hw2copy import *
from ml import *

import numpy as np
import pandas as pd
from collections import defaultdict
import statsmodels.api as sm
from statsmodels.formula.api import ols
import random
import math



def main1():

    np.random.seed(42)
    random.seed(42)
    data = pd.read_csv("realone.csv") #full country-wide dataset


    uttarPradesh = data[data['state33'] == 1] #uttar Pradesh specific
    uttarPradesh = uttarPradesh.dropna()


    treatment = "s555"

    #optimal adjustment set
    controls = ["toilet", "lowerCaste", "rural", "b4", "bord", "v190", "v133"]

    #state-level estimations
    print("state")
    print("Height-for-age")
    outcome = "hw70ff"

    #linear AIPW
    print(augmented_ipw(outcome, treatment, controls, uttarPradesh, trim=False))
    print(compute_confidence_intervalsipw(outcome, treatment, controls, uttarPradesh, "augmented_ipw", num_bootstraps=50))
    print(augmented_ipw(outcome, treatment, controls, uttarPradesh, trim=True))
    print(compute_confidence_intervalsipw(outcome, treatment, controls, uttarPradesh, "augmented_ipwtrim", num_bootstraps=50))

    #ML backdoor
    print(backdoor_ML(outcome, treatment, controls, uttarPradesh))
    print(compute_confidence_intervals(outcome, treatment, controls, uttarPradesh, "backdoor_ML", num_bootstraps=50, alpha=0.05, value=None))

    #ML AIPW
    print(aipw_ML(outcome, treatment, controls, uttarPradesh))
    print(compute_confidence_intervals(outcome, treatment, controls, uttarPradesh, "aipw_ML", num_bootstraps=50, alpha=0.05, value=None))


    print("Weight-for-Age")
    outcome = "hw71ff"

    #linear AIPW
    print(augmented_ipw(outcome, treatment, controls, uttarPradesh, trim=False))
    print(compute_confidence_intervalsipw(outcome, treatment, controls, uttarPradesh, "augmented_ipw", num_bootstraps=50))
    print(augmented_ipw(outcome, treatment, controls, uttarPradesh, trim=True))
    print(compute_confidence_intervalsipw(outcome, treatment, controls, uttarPradesh, "augmented_ipwtrim", num_bootstraps=50))

    #ML backdoor
    print(backdoor_ML(outcome, treatment, controls, uttarPradesh))
    print(compute_confidence_intervals(outcome, treatment, controls, uttarPradesh, "backdoor_ML", num_bootstraps=50, alpha=0.05, value=None))

    #ML AIPW
    print(aipw_ML(outcome, treatment, controls, uttarPradesh))
    print(compute_confidence_intervals(outcome, treatment, controls, uttarPradesh, "aipw_ML", num_bootstraps=50, alpha=0.05, value=None))


    #country-wide estimations
    print("country")

    #parents of treatment
    stateindicators = ["state1","state10","state11","state12","state13","state14","state15","state16","state17","state18","state19","state2","state20","state21","state22","state23","state24","state25","state26","state27","state28","state29","state3","state30","state31","state32","state33","state34","state35","state36","state4","state5","state6","state7","state8","state9"]
    controls = ["toilet", "lowerCaste", "rural", "b4", "bord", "v190", "v133"] + ["hw1", "v438"] + stateindicators

    print("Height-for-age")

    outcome = "hw70ff"

    #linear AIPW
    print(augmented_ipw(outcome, treatment, controls, data, trim=False))
    print(compute_confidence_intervalsipw(outcome, treatment, controls, data, "augmented_ipw", num_bootstraps=50))
    print(augmented_ipw(outcome, treatment, controls, data, trim=True))
    print(compute_confidence_intervalsipw(outcome, treatment, controls, data, "augmented_ipwtrim", num_bootstraps=50))

    #ML backdoor
    print(backdoor_ML(outcome, treatment, controls, data))
    print(compute_confidence_intervals(outcome, treatment, controls, data, "backdoor_ML", num_bootstraps=50, alpha=0.05, value=None))

    #ML AIPW
    print(aipw_ML(outcome, treatment, controls, data))
    print(compute_confidence_intervals(outcome, treatment, controls, data, "aipw_ML", num_bootstraps=50, alpha=0.05, value=None))


    print("Weight-for-Age")
    outcome = "hw71ff"

    #linear AIPW
    print(augmented_ipw(outcome, treatment, controls, data, trim=False))
    print(compute_confidence_intervalsipw(outcome, treatment, controls, data, "augmented_ipw", num_bootstraps=50))
    print(augmented_ipw(outcome, treatment, controls, data, trim=True))
    print(compute_confidence_intervalsipw(outcome, treatment, controls, data, "augmented_ipwtrim", num_bootstraps=50))

    #ML backdoor
    print(backdoor_ML(outcome, treatment, controls, data))
    print(compute_confidence_intervals(outcome, treatment, controls, data, "backdoor_ML", num_bootstraps=50, alpha=0.05, value=None))

    #ML AIPW
    print(aipw_ML(outcome, treatment, controls, uttarPradesh))
    print(compute_confidence_intervals(outcome, treatment, controls, data, "aipw_ML", num_bootstraps=50, alpha=0.05, value=None))






    #country-level estimations
    # print("country")
    # data = pd.read_csv("realone.csv") #full dataset
    # data = data.dropna()
    #
    # #parents of treatment
    # controls = ["toilet", "lowerCaste", "rural", "b4", "bord", "v190", "v133"] +["hw1", "v438"]
    #
    # print(backdoor_ML(outcome, treatment, controls, data))
    # #change back to 0.05
    # print(compute_confidence_intervals(outcome, treatment, controls, data, "backdoor_ML", num_bootstraps=5, alpha=0.01, value=None))
    #
    # print(aipw_ML(outcome, treatment, controls, data))
    # print(compute_confidence_intervals(outcome, treatment, controls, data, "aipw_ML", num_bootstraps=5, alpha=0.01, value=None))
    #
    #
    # print(augmented_ipw(outcome, treatment, controls, data, trim=False))
    # print(compute_confidence_intervalsipw(outcome, treatment, controls, data, "augmented_ipw", num_bootstraps=25))
    #
    # print(augmented_ipw(outcome, treatment, controls, data, trim=True))
    # print(compute_confidence_intervalsipw(outcome, treatment, controls, data, "augmented_ipwtrim", num_bootstraps=25))

    # outcome = "hw71ff"
    # print(outcome)
    # print(augmented_ipw(outcome, treatment, controls, data, trim=False))
    # print(compute_confidence_intervalsipw(outcome, treatment, controls, data, "augmented_ipw", num_bootstraps=25))
    #
    # print(augmented_ipw(outcome, treatment, controls, data, trim=True))
    # print(compute_confidence_intervalsipw(outcome, treatment, controls, data, "augmented_ipwtrim", num_bootstraps=25))
    #
    # quit()
    # print(backdoor_ML(outcome, treatment, controls, data))
    # print(compute_confidence_intervals(outcome, treatment, controls, data, "backdoor_ML", num_bootstraps=25, alpha=0.01, value=None))
    #
    # print(aipw_ML(outcome, treatment, controls, data))
    # print(compute_confidence_intervals(outcome, treatment, controls, data, "aipw_ML", num_bootstraps=5, alpha=0.01, value=None))
    # quit()
    #
    #
    #
    # print(augmented_ipw(outcome, treatment, controls, data, trim=False))
    # print(compute_confidence_intervalsipw(outcome, treatment, controls, data, "augmented_ipw"))
    #
    # print(augmented_ipw(outcome, treatment, controls, data, trim=True))
    # print(compute_confidence_intervalsipw(outcome, treatment, controls, data, "augmented_ipwtrim"))
    #
    # print("hw71ff")
    # outcome = "hw71ff"
    # print(augmented_ipw(outcome, treatment, controls, data, trim=False))
    # print(compute_confidence_intervalsipw(outcome, treatment, controls, data, "augmented_ipw"))
    #
    # print(augmented_ipw(outcome, treatment, controls, data, trim=True))
    # print(compute_confidence_intervalsipw(outcome, treatment, controls, data, "augmented_ipwtrim"))

    #
    # print("hw72ff")
    # outcome = "hw72ff"
    # treatment = "s555"
    # controls = ["toilet", "lowerCaste", "rural", "b4"]
    # print(augmented_ipw(outcome, treatment, controls, data, trim=False))
    # print(compute_confidence_intervalsipw(outcome, treatment, controls, data, "augmented_ipw"))
    #
    # print("country")
    # data = pd.read_csv("realone.csv")
    # data = data.dropna()
    # outcome = "hw70ff"
    # treatment = "s555"
    # controls = ["toilet", "lowerCaste", "rural", "b4"]
    # print("hw70ff")
    # print(augmented_ipw(outcome, treatment, controls, data, trim=False))
    # print(compute_confidence_intervalsipw(outcome, treatment, controls, data, "augmented_ipw"))
    # print("hw71ff")
    # outcome = "hw71ff"
    # print(augmented_ipw(outcome, treatment, controls, data, trim=False))
    # print(compute_confidence_intervalsipw(outcome, treatment, controls, data, "augmented_ipw"))
    # print("hw72ff")
    # outcome = "hw72ff"
    # print(augmented_ipw(outcome, treatment, controls, data, trim=False))
    # print(compute_confidence_intervalsipw(outcome, treatment, controls, data, "augmented_ipw"))



if __name__ == "__main__":
    main1()
