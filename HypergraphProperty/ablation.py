import argparse
import os
import pickle
from utils import *
from collections import defaultdict

column_mapping = {
    "type_2_affinity_score": "aff2", "type_3_affinity_score": "aff3", "type_4_affinity_score": "aff4",
    "hyperedge_entropy": "he", "higher_order_hyperedge_entropy": "hohe", "node_homophily_score": "nhs"
}

def get_directories(dataname, ablation_target):
    namelist = [("answer", -1)]
    file_names = os.listdir(f"../generated/{ablation_target}/{dataname}/")
    valid_file_names = [f for f in file_names if "preindexing" not in f and "indices" not in f and f.endswith(".txt")]
    
    if len(valid_file_names) == 1:
        namelist.append((ablation_target, -1))
    else:
        for _fname in valid_file_names:
            fname = _fname[:-4]
            paramname = fname[len(ablation_target) + 1:]
            namelist.append((ablation_target, paramname))
            
    return namelist

def make_ablation_table(dataname, ablation_target):
    namelist = get_directories(dataname, ablation_target)
    property_list = ["type_2_affinity_score", "type_3_affinity_score", "type_4_affinity_score", "hyperedge_entropy", "higher_order_hyperedge_entropy", "node_homophily_score"]
    outputdir = "analyze/csv/" + ablation_target + "/" 
    os.makedirs(outputdir, exist_ok=True)
    outputpath = outputdir + dataname + ".txt"
    
    columns = [column_mapping[prop] for prop in property_list]
    with open(outputpath, "w") as f:
        f.write(",".join(["Model", "OptParam"] + columns) + "\n")
        
    for name, modelindex in namelist:
        dist = read_properties(dataname, name, modelindex)
        
        if name == "answer":
            dist_answer = dist   
            continue
        if dist is None:
            continue
        
        difflist = []
        for prop in property_list:
            if prop in ["type_2_affinity_score", "type_3_affinity_score", "type_4_affinity_score"]:
                cur_dist_answer = np.array(dist_answer[prop])
                cur_dist = np.array(dist[prop])
                diff = np.sum(np.abs(np.log1p(cur_dist_answer) - np.log1p(cur_dist)))
                # diff = np.sum(np.abs(cur_dist_answer - cur_dist) / cur_dist_answer)
                # diff = np.sum(np.abs(np.log(cur_dist_answer) - np.log(cur_dist)))
                # valid_entries = ~((np.array(dist_answer[prop]) == 0) & (np.array(dist_answer[prop]) == 0))
                # diff = np.sum(np.abs(np.array(dist_answer[prop])[valid_entries] - np.array(dist[prop])[valid_entries]) / (np.array(dist_answer[prop])[valid_entries] + np.array(dist[prop])[valid_entries]) * 2)
                # diff = np.sum(np.abs(np.array(dist_answer[prop]) - np.array(dist[prop])))
                # diff = np.sum(np.abs(np.array(dist_answer[prop])[valid_entries] - np.array(dist[prop])[valid_entries]) / np.array(dist_answer[prop])[valid_entries])
            elif prop in ["node_homophily_score", "hyperedge_entropy", "higher_order_hyperedge_entropy"]:
                diffs = []
                for k in range(dist_answer[prop].shape[1]):
                    diffs.append(get_wasserstein_distance(dist_answer[prop][:, k], dist[prop][:, k])) 
                diff = np.sum(diffs)
            difflist.append(str(diff))

        with open(outputpath, "a") as f:
            OptParam = str(modelindex)
            Model = name
            f.write(",".join([Model, OptParam] + difflist) + "\n")
    
    return outputpath
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataname", type=str)
    parser.add_argument("--ablation_target", type=str)
    args = parser.parse_args()

    outputpath = make_ablation_table(args.dataname, args.ablation_target)
    evallist = ['aff2', 'aff3', 'aff4', 'he', 'hohe', 'nhs']

    # Make Ranking Result
    prefix = outputpath[:-4]
    d = pd.read_csv(outputpath)
    for ename in evallist:
        d[ename] = d[ename].abs().rank(method='min')
    ranks = d[evallist]
    d['avg'] = ranks.mean(axis=1)
    d = d.sort_values(by=["avg"], ascending=True)
    
    OptParam2sum = defaultdict(int)
    rorder = 1
    for irow, row in d.iterrows():
        if row["Model"] == args.ablation_target:
            OptParam = str(row["OptParam"])
            OptParam2sum[OptParam] += rorder
        rorder += 1
    sortedkeys = sorted(list(OptParam2sum.keys()), key=lambda x: OptParam2sum[x])

    os.makedirs("analyze/ablation_result/", exist_ok=True)
    file_path = "analyze/ablation_result/" + args.ablation_target + ".pkl"

    if os.path.isfile(file_path):
        with open(file_path, "rb") as f:
            ablation_result = pickle.load(f)
    else:
        ablation_result = {}
    
    top_opt = sortedkeys[0]
    ablation_result[args.dataname] = top_opt

    with open(file_path, "wb") as f:
        pickle.dump(ablation_result, f)
