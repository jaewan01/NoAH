import argparse
import os
import pickle
import numpy as np
import pandas as pd
from utils import *

column_mapping = {
    "type_2_affinity_ratio_scores": "T2", "type_3_affinity_ratio_scores": "T3", "type_4_affinity_ratio_scores": "T4",
    "hyperedge_entropy": "HE", "higher_order_hyperedge_entropy": "HOHE", "node_homophily_score": "NHS"
}


def get_baseline_list(dataname):
    file_names = os.listdir("ablation_result/")
    namelist = [("answer", -1)]
    for fname in file_names:
        if "cfnomix" in fname:
            continue
        with open("ablation_result/" + fname, "rb") as f:
            result = pickle.load(f)
        namelist.append((fname[:-4], result[dataname]))
        
    return namelist

def make_dist_table(dataname, outputdir, namelist=None):
    property_list = ["type_2_affinity_ratio_scores", "type_3_affinity_ratio_scores", "type_4_affinity_ratio_scores", "hyperedge_entropy", "higher_order_hyperedge_entropy", "node_homophily_score"]
    
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
            if prop in ["type_2_affinity_ratio_scores", "type_3_affinity_ratio_scores", "type_4_affinity_ratio_scores"]:
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
        f.write(",".join([Model, OptParam] + difflist) + "\n")

    f.close()
            
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataname", type=str)
    parser.add_argument("--outputdir", default="csv/best_fit/", type=str)
    args = parser.parse_args()

    dataname = args.dataname
    outputdir = args.outputdir
    os.makedirs(outputdir, exist_ok=True)
    make_dist_table(dataname, outputdir)

    evallist = ['T2', 'T3', 'T4', 'HE', 'HOHE', 'NHS']
    
    df = pd.read_csv(outputdir + dataname + "_attribute.txt")
    df = df.reset_index(drop=True)
    temp_df = df.copy()
    for ename in evallist:
        temp_df[ename] = temp_df[ename].abs().rank(method='min')
    ranks = temp_df[evallist]
    df['AVG Rank'] = ranks.mean(axis=1)
    df.to_csv(outputdir + dataname + "_attribute.txt", index=False)
    df = df.drop(columns=['OptParam'])
    print(df)
    print()
