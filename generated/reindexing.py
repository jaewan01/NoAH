import argparse
import os
import tqdm
import numpy as np
import pdb

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        "-target",
        "--target_hypergraph",
        default='cora_coauth',
        action="store",
        type=str,
        help="Select the target real-world hypergraph.",
    )
    
    np.random.seed(1)

    data_name = parser.parse_args().target_hypergraph

    gt_attribute = []
    with open(f"../dataset/{data_name}/attribute.txt", "r") as f:
        for line in f.readlines():
            gt_attribute.append(line.strip().split(","))

    models = ["noah", "bi"]

    for model in models:
        preindexed_files = os.listdir(f"{model}/{data_name}/")
        preindexed_files = [f for f in preindexed_files if f.endswith("-preindexing.txt")]
        for preindexed_file in tqdm.tqdm(preindexed_files, desc=f"Reindexing {model} {data_name}"):
            name_wo_ext = preindexed_file[:-len("-preindexing.txt")]
            # if os.path.exists(f"{model}/{data_name}/{name_wo_ext}.txt"):
            #     print(f"{name_wo_ext}.txt already exists!")
            #     continue
            preindexed_hyperedges = []
            with open(f"{model}/{data_name}/{preindexed_file}", "r") as f:
                preindexing = f.readlines()
                for line in preindexing:
                    preindexed_hyperedges.append([int(i) for i in line.strip().split(",")])
            old_to_new_index = {}
            new_hyperedges = []
            new_indices = []
            for hyperedge in preindexed_hyperedges:
                for node in hyperedge:
                    if node not in old_to_new_index.keys():
                        old_to_new_index[node] = len(old_to_new_index.keys())
                        new_indices.append(node)
                new_hyperedges.append([old_to_new_index[node] for node in hyperedge])
            with open(f"{model}/{data_name}/{name_wo_ext}.txt", "w") as f:
                for hyperedge in new_hyperedges:
                    f.write(",".join([str(i) for i in hyperedge]) + "\n")
            with open(f"{model}/{data_name}/{name_wo_ext}-indices.txt", "w") as f:
                f.write(",".join([str(i) for i in new_indices]))    
    