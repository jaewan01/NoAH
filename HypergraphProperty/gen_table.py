import argparse
import os
import copy
import pickle
import numpy as np
import pandas as pd
from utils import *

column_mapping = {
    "type_2_affinity_score": "Size-2 Affinity Score", "type_3_affinity_score": "Size-3 Affinity Score", "type_4_affinity_score": "Size-4 Affinity Score",
    "hyperedge_entropy": "Hyperedge Entropy", "higher_order_hyperedge_entropy": "Higher Order Hyperedge Entropy", "node_homophily_score": "Node Homophily Score",
    "degree": "Degree", "pairdeg": "Pair Degree", "size": "Size", "intersection": "Int. Size", 
    "sv": "SV", "sizewcc": "CC", "clusteringcoef": "GCC", "density": "Density", "overlapness": "Overlapness", "effdiameter": "Diameter"
}

model_mapping = {
    "cl": "HyperCL", "pa": "HyperPA", "lap": "HyperLAP",
    "ff": "HyperFF", "dk": "HyperdK", "tr": "THera", "sbm": "HyCoSBM", "hyrec": "HyRec",
    "noah": "NoAH", "bi": "Bipartite Model"
}

def get_baseline_list(dataname):
    file_names = os.listdir("analyze/ablation_result/")
    namelist = [("answer", -1)]
    for fname in file_names:
        if "cfnomix" in fname:
            continue
        with open("analyze/ablation_result/" + fname, "rb") as f:
            result = pickle.load(f)
        namelist.append((fname[:-4], result[dataname]))
        
    print(namelist)
        
    return namelist

def make_dist_table(dataname, outputdir, namelist=None):
    property_list = ["type_2_affinity_score", "type_3_affinity_score", "type_4_affinity_score", "hyperedge_entropy", "higher_order_hyperedge_entropy", "node_homophily_score"]
    
    if namelist is None:
        namelist = get_baseline_list(dataname)
    
    outputpath = outputdir + dataname + "_attribute.txt"
    columns = [column_mapping[prop] for prop in property_list]
    f = open(outputpath, "w")
    f.write(",".join(["Model", "OptParam"] + columns) + "\n")
        
    for name, modelindex in namelist:
        if modelindex == "-1":
            dist = read_properties(dataname, name, -1)
        else:
            dist = read_properties(dataname, name, modelindex)
       
        if name == "answer":
            dist_answer = dist   
            continue

        difflist = []
        for prop in property_list:
            if prop in ["type_2_affinity_score", "type_3_affinity_score", "type_4_affinity_score"]:
                cur_dist_answer = np.array(dist_answer[prop])
                cur_dist = np.array(dist[prop])
                diff = np.sum(np.abs(np.log1p(cur_dist_answer) - np.log1p(cur_dist)))
            elif prop in ["node_homophily_score", "hyperedge_entropy", "higher_order_hyperedge_entropy"]:
                diffs = []
                for k in range(dist_answer[prop].shape[1]):
                    diffs.append(get_wasserstein_distance(dist_answer[prop][:, k], dist[prop][:, k]))
                diff = np.sum(diffs) 
            difflist.append(str(diff))
        
        OptParam= str(modelindex)
        Model = name
        f.write(",".join([model_mapping[Model], OptParam] + difflist) + "\n")

    f.close()
    
    property_list = ["degree", "pairdeg", "size", "intersection", "sv", "sizewcc", "clusteringcoef", "density", "overlapness", "effdiameter"]
    
    outputpath = outputdir + dataname + "_topology.txt"
    columns = [column_mapping[prop] for prop in property_list]
    f = open(outputpath, "w")
    f.write(",".join(["Model", "OptParam"] + columns) + "\n")
        
    for name, modelindex in namelist:
        if modelindex == "-1":
            dist = read_properties(dataname, name, -1)
        else:
            dist = read_properties(dataname, name, modelindex)
       
        if name == "answer":
            dist_answer = dist   
            continue

        difflist = []
        for prop in property_list:
            if prop in ["effdiameter", "clusteringcoef", "density", "overlapness"]:
                diff = abs(dist[prop] - dist_answer[prop]) / dist_answer[prop]
            elif prop in ["degree", "size", "pairdeg", "intersection"]:
                diff = get_cumul_dist(dist_answer[prop], dist[prop])
                assert diff >= 0 and diff <= 1.1
            elif prop in ["sv", "sizewcc"]:
                diff = get_cumul_dist(dist_answer[prop], dist[prop], cdf_flag=True)
            difflist.append(str(diff))
        
        OptParam= str(modelindex)
        Model = name
        f.write(",".join([model_mapping[Model], OptParam] + difflist) + "\n")

    f.close()
            
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataname", type=str)
    parser.add_argument("--outputdir", default="analyze/csv/best_fit/", type=str)
    args = parser.parse_args()

    dataname = args.dataname
    outputdir = args.outputdir
    os.makedirs(outputdir, exist_ok=True)
    make_dist_table(dataname, outputdir)

    evallist = ['Size-2 Affinity Score', 'Size-3 Affinity Score', 'Size-4 Affinity Score', 'Hyperedge Entropy', 'Higher Order Hyperedge Entropy', 'Node Homophily Score']
    
    df = pd.read_csv(outputdir + dataname + "_attribute.txt")
    new_order = [2, 7, 4, 6, 3, 9, 8, 5, 1, 0]
    df = df.iloc[new_order].reset_index(drop=True)
    temp_df = df.copy()
    for ename in evallist:
        temp_df[ename] = temp_df[ename].abs().rank(method='min')
    ranks = temp_df[evallist]
    df['AVG Rank'] = ranks.mean(axis=1)
    df.to_csv(outputdir + dataname + "_attribute.txt", index=False)
    df = df.drop(columns=['OptParam'])
    print(df)
    print()
    
    evallist = ['Degree', 'Int. Size', 'Pair Degree', 'Size', 'SV', 'CC', 'GCC', 'Density', 'Overlapness', 'Diameter']
    
    df = pd.read_csv(outputdir + dataname + "_topology.txt")
    new_order = [2, 7, 4, 6, 3, 9, 8, 5, 1, 0]
    df = df.iloc[new_order].reset_index(drop=True)
    temp_df = df.copy()
    for ename in evallist:
        temp_df[ename] = temp_df[ename].abs().rank(method='min')
    ranks = temp_df[evallist]
    df['AVG Rank'] = ranks.mean(axis=1)
    df.to_csv(outputdir + dataname + "_topology.txt", index=False)
    df = df.drop(columns=['OptParam'])
    print(df)
    print()
