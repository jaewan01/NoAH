import argparse
import os
import tqdm
import numpy as np
import pdb


def reindex_single_model(data_name, model):
    model_dir = f"{model}/{data_name}/"
    if os.path.isdir(model_dir) is False:
        print(f"[skip] {model_dir} does not exist")
        return

    preindexed_files = os.listdir(model_dir)
    preindexed_files = [f for f in preindexed_files if f.endswith("-preindexing.txt")]

    keep_original_index_models = {"cl", "lap", "sbm", "dk"}
    if model.startswith("noah"):
        keep_original_index = True
    else:
        keep_original_index = model in keep_original_index_models

    for preindexed_file in tqdm.tqdm(preindexed_files, desc=f"Reindexing {model} {data_name}"):
        name_wo_ext = preindexed_file[:-len("-preindexing.txt")]
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
                    if keep_original_index:
                        new_indices.append(node)
            new_hyperedges.append([old_to_new_index[node] for node in hyperedge])

        if keep_original_index is False:
            new_indices = np.random.choice(len(old_to_new_index), len(old_to_new_index), replace=False)

        with open(f"{model}/{data_name}/{name_wo_ext}.txt", "w") as f:
            for hyperedge in new_hyperedges:
                f.write(",".join([str(i) for i in hyperedge]) + "\n")
        with open(f"{model}/{data_name}/{name_wo_ext}-indices.txt", "w") as f:
            f.write(",".join([str(i) for i in new_indices]))

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
    parser.add_argument(
        "--models",
        nargs="+",
        default=["noah"],
        type=str,
        help="Models to reindex from -preindexing.txt files.",
    )
    
    np.random.seed(1)

    args = parser.parse_args()
    data_name = args.target_hypergraph

    gt_attribute = []
    with open(f"../dataset/{data_name}/attribute.txt", "r") as f:
        for line in f.readlines():
            gt_attribute.append(line.strip().split(","))


    models = args.models
    for model in models:
        reindex_single_model(data_name, model)
    
