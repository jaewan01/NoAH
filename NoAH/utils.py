import os
import random
import numpy as np
import torch
import tqdm
from tqdm import trange
 

def fix_seed(seed):
    
    """
        Fix the seed for reproducing.
    """
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True
    
 
def prep_dataset(data_name):
    
    """
        Dataset Organization
        |__ dataset/"dataset_name"/                   
            |__ hyperedge.txt : Contains list of hyperedges. Each hyperedge is consists of nodes seperated by ",".
            |__ attribute.txt : Contains list of binary node attribute vectors. i-th line indicates node i's attribute. Each attribute vector is consists of attributes seperated by ",". 
            
        Read hypergraph infomation from hyperedge.txt & attribute.txt, and return hyperedges, node attributes, number of nodes, number of hyperedges, and dimension of node attribute.
    """
    
    path = "../dataset/" + data_name + "/"

    nodes = set()
    hyperedges = []
    
    with open(path + "hyperedge.txt", "r") as f:
        for line in f.readlines():
            cur_line = line.strip().split(",")
            nodes.update([int(i) for i in cur_line])
            hyperedges.append([int(i) for i in cur_line])
            
    num_nodes = len(nodes)
    num_edges = len(hyperedges)
    attributes = []
    
    with open(path + "attribute.txt", "r") as f:
        for line in f.readlines():
            cur_line = line.strip().split(",")
            attributes.append([float(i) for i in cur_line])
    
    attributes = torch.tensor(attributes)
    attr_dim = attributes.shape[1]
            
    return hyperedges, attributes, num_nodes, num_edges, attr_dim


def UMHS(data_name, iter):
    
    """
        Core recovery algorithm inspired from https://github.com/ilyaamburg/Hypergraph-Planted-Hitting-Set-Recovery
    """
    
    cores = set()
    hyperedges, _, n, m, _ = prep_dataset(data_name)
    
    for _ in trange(iter, desc="UMHS core-fringe split " + data_name):
        # 1. Shuffle the index of hyperedges.
        shuffled_hyperedges = hyperedges.copy()
        np.random.shuffle(shuffled_hyperedges)
        # 2. Find a maximal hitting set.
        hitting_set = set()
        n2e = [[] for _ in range(n)]
        he_idx = 0
        for hyperedge in shuffled_hyperedges:
            to_add = True
            for node in hyperedge:
                if node in hitting_set:
                    to_add = False
                n2e[node].append(he_idx)
            if to_add:
                hitting_set.update(hyperedge)
            he_idx += 1
        # 3. Find a minimal hitting set.
        minimal_hitting_set = hitting_set.copy()
        for target_node in hitting_set:
            covered_hyperedge = set()
            for node in minimal_hitting_set:
                if node != target_node:
                    covered_hyperedge |= set(n2e[node])
                if len(covered_hyperedge) == m:
                    break
            if len(covered_hyperedge) == m:
                minimal_hitting_set.remove(target_node)
        # 4. Add minimal hitting set to core set.
        cores |= (minimal_hitting_set)
        
    cores = sorted(list(cores))
    fringes = list(range(n))
    
    for core in cores:
        fringes.remove(core)
        
    return cores, fringes


def reindexing(data_name, mode):
    gt_attribute = []
    with open(f"../dataset/{data_name}/attribute.txt", "r") as f:
        for line in f.readlines():
            gt_attribute.append(line.strip().split(","))
        
    preindexed_files = os.listdir(f"../generated/{mode}/{data_name}/")
    preindexed_files = [f for f in preindexed_files if f.endswith("-preindexing.txt")]

    for preindexed_file in tqdm.tqdm(preindexed_files, desc=f"Reindexing {mode} {data_name}"):
        name_wo_ext = preindexed_file[:-len("-preindexing.txt")]
        preindexed_hyperedges = []

        with open(f"../generated/{mode}/{data_name}/{preindexed_file}", "r") as f:
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

        with open(f"../generated/{mode}/{data_name}/{name_wo_ext}.txt", "w") as f:
            for hyperedge in new_hyperedges:
                f.write(",".join([str(i) for i in hyperedge]) + "\n")

        with open(f"../generated/{mode}/{data_name}/{name_wo_ext}-indices.txt", "w") as f:
            f.write(",".join([str(i) for i in new_indices])) 

        os.remove(f"../generated/{mode}/{data_name}/" + preindexed_file)   
