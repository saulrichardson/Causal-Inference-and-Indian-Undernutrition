
from fcit import fcit
import numpy as np
import pandas as pd
from collections import defaultdict
import statsmodels.api as sm
from statsmodels.formula.api import ols
import random
import math


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

def get_input(name, frame):
    data = frame
    var = get_numpy_matrix(data, [name])
    # matrix = np.array(np.array([var[0]]))
    runningList = []
    for a in range(0, np.prod(var.shape)):
        j = np.array([var[a]])
        runningList.append(j)
    finalM = np.array(runningList)
    return finalM

def main():
    data = pd.read_csv("realone.csv")
    data = data[data['state33'] == 1]
    data = data.dropna()


    #testing presence between treatment and each outcome
    ee = get_input("s555", data)

    #valid adjustment set
    controls = ["toilet", "lowerCaste", "rural", "b4"]
    l = get_numpy_matrix(data, controls)

    #height for age
    oo = get_input("hw70ff", data)
    print("recieved to heightZ")

    res = []
    for bb in range(5):
        res.append(fcit.test(ee, oo, l, discrete=(True, False)))
    print(np.mean(res))


    #weight for age
    oo = get_input("hw71ff", data)
    print("recieved to weightZ")
    res = []
    for bb in range(5):
        res.append(fcit.test(ee, oo, l, discrete=(True, False)))
    print(np.mean(res))



    #testinng edge from different explanatory variables to height Z outcome


    #new adjustment set is valid for each test
    l = get_numpy_matrix(data, controls + ["s555"])

    oo = get_input("hw70ff", data) #height-for-age
    print("motheredu")
    ee = get_input("v133", data)
    res = []
    for bb in range(5):
        res.append(fcit.test(ee, oo, l, discrete=(True, False)))
    print(np.mean(res))

    print("birthOrder")
    ee = get_input("bord", data)
    res = []
    for bb in range(5):
        res.append(fcit.test(ee, oo, l, discrete=(True, False)))
    print(np.mean(res))

    print("wealth")
    ee = get_input("v190", data)
    res = []
    for bb in range(5):
        res.append(fcit.test(ee, oo, l, discrete=(True, False)))
    print(np.mean(res))

    print("mother's height")
    ee = get_input("v438", data)
    res = []
    for bb in range(5):
        res.append(fcit.test(ee, oo, l, discrete=(True, False)))
    print(np.mean(res))

    print("number of children under five")
    ee = get_input("v137", data)
    res = []
    for bb in range(5):
        res.append(fcit.test(ee, oo, l, discrete=(True, False)))
    print(np.mean(res))


    oo = get_input("hw71ff", data) #weight-for-age
    print("motheredu")
    ee = get_input("v133", data)
    res = []
    for bb in range(5):
        res.append(fcit.test(ee, oo, l, discrete=(True, False)))
    print(np.mean(res))

    print("birthOrder")
    ee = get_input("bord", data)
    res = []
    for bb in range(5):
        res.append(fcit.test(ee, oo, l, discrete=(True, False)))
    print(np.mean(res))

    print("wealth")
    ee = get_input("v190", data)
    res = []
    for bb in range(5):
        res.append(fcit.test(ee, oo, l, discrete=(True, False)))
    print(np.mean(res))

    print("mother's height")
    ee = get_input("v438", data)
    res = []
    for bb in range(5):
        res.append(fcit.test(ee, oo, l, discrete=(True, False)))
    print(np.mean(res))

    print("number of children under five")
    ee = get_input("v137", data)
    res = []
    for bb in range(5):
        res.append(fcit.test(ee, oo, l, discrete=(True, False)))
    print(np.mean(res))


if __name__ == "__main__":
    main()
