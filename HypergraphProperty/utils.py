import pandas as pd
import numpy as np
import os
import math
from scipy.stats import wasserstein_distance

# Read Properties ------------------------------------------------------------------------------------------------------

def read_properties(dataname, model, modelindex=-1):
    if "answer" == model:
        outputdir = "results/{}/{}/".format(model, dataname)
    elif modelindex == -1:
        outputdir = "results/{}/{}/".format(model, dataname)
    else:
        outputdir = "results/" + model + "/" + dataname + "/" + modelindex + "/"
    
    dist = {}
    
    for distname in ["type_2_affinity_score", "type_3_affinity_score", "type_4_affinity_score"]:
        file_path_zero = outputdir + distname + "_0.txt"
        values = []
        with open(file_path_zero, "r") as f:
            for line in f.readlines():
                values.append(list(map(float,line.rstrip().split(","))))
        file_path_one = outputdir + distname + "_1.txt"
        with open(file_path_one, "r") as f:
            for line in f.readlines():
                values.append(list(map(float,line.rstrip().split(","))))
        dist[distname] = values
    for distname in ["node_homophily_score", "hyperedge_entropy", "higher_order_hyperedge_entropy"]:
        file_path = outputdir + distname + ".npy"
        dist[distname] = np.load(file_path)

    for distname in ["degree", "pairdeg", "intersection", "size"]:
        dist[distname] = {}
        X = []
        with open(outputdir + distname + ".txt", "r") as f:
            for line in f.readlines():
                val, pdf = line.rstrip().split(",")
                val, pdf = float(val), float(pdf)
                if pdf == 0.0 or val == 0.0:
                    continue
                dist[distname][val] = pdf
                X.append(val)
        X = sorted(X)
        Y = [dist[distname][x] for x in X]
    
    with open(outputdir + "sv.txt", "r") as f:
        X, Y = [], []
        for li, line in enumerate(f.readlines()):
            tmp = line.rstrip().split(",")
            x, y = float(tmp[0]), float(tmp[1])
            Y.append(y)
            X.append(x)
        dist["sv"] = {}
        for x,y in zip(X, Y):
            dist["sv"][x] = y
    
    with open(outputdir + "sizewcc.txt") as f:
        X, Y = [], []
        for li, line in enumerate(f.readlines()):
            val = float(line.rstrip())
            Y.append(val)
            X.append(li+1)
        dist["sizewcc"] = {}
        for x,y in zip(X, Y):
            dist["sizewcc"][x] = y

    for distname in ["clusteringcoef", "density", "overlapness", "effdiameter"]:
        with open(outputdir + distname + ".txt", "r") as f:
            value = 0
            for line in f.readlines():
                value = float(line.rstrip())
            dist[distname] = value

    return dist

# Get Distance ------------------------------------------------------------------------------------------------------

def get_wasserstein_distance(x1, x2):
    return wasserstein_distance(x1, x2)
        
def get_cdf(_dict):
    cumulated_x = sorted(list(_dict.keys()))
    cdf = {}
    cum = 0

    for _x in cumulated_x:
        cum += _dict[_x]
        cdf[_x] = cum
        assert cum < 1.1
        
    return cdf

def get_cumul_dist(dict_x1, dict_x2, cdf_flag=False):
    if cdf_flag is False:
        cdf1 = get_cdf(dict_x1)
        cdf2 = get_cdf(dict_x2)
    else:
        cdf1 = dict_x1
        cdf2 = dict_x2
    x1 = list(cdf1.keys())
    x2 = list(cdf2.keys())
    
    cum1, cum2 = 0, 0
    maxdiff = 0
    for x in sorted(list(set(x1 + x2))):
        if x in x1:
            cum1 = cdf1[x]
        if x in x2:
            cum2 = cdf2[x]
        if abs(cum1 - cum2) > maxdiff:
            maxdiff = abs(cum1 - cum2)
    
    return maxdiff

def get_rmse(y1s, y2s):
    total = 0
    y1s = np.array(y1s)
    y2s = np.array(y2s)
    total = np.sum((y1s - y2s) ** 2)
    assert y1s.shape[0] > 0
    total /= y1s.shape[0]
    total = total ** 0.5
    
    return total

def get_rmse_dist(dict_x1, dict_x2, set_length=False, normalize=False, logflag=True):
    total = 0
    maxy1 = 0
    
    x1s = list(dict_x1.keys())
    x2s = list(dict_x2.keys())
    
    if set_length:
        keys = x1s
    else:
        keys = set(x1s + x2s)
    
    for x in keys:
        y1, y2 = math.log2(1e-20), math.log2(1e-20)
        if x in x1s:
            y1 = dict_x1[x]
            if logflag:
                y1 = math.log2(y1)
            if y1 > maxy1:
                maxy1 = y1
        if x in x2s:
            y2 = dict_x2[x]
            if logflag:
                y2 = math.log2(y2)
        total += (y1 - y2) ** 2
    
    total /= len(keys)
    total = total ** 0.5
    
    if normalize:
        total /= maxy1
        
    return total
