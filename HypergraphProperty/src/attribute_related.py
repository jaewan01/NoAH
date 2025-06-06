import argparse
import numpy as np
import os
from math import comb

parser = argparse.ArgumentParser()
parser.add_argument('--inputpath', required=False)
parser.add_argument('--outputdir', required=False)
parser.add_argument('--dim', type=int, default=0)
parser.add_argument(
        "-target",
        "--target_hypergraph",
        default='contact_hospital',
        action="store",
        type=str,
        help="Select the target real-world hypergraph.",
    )

args = parser.parse_args()
attrpath = "../dataset/" + args.target_hypergraph + "/attribute.txt"
graphpath = args.inputpath + ".txt"
indexpath = args.inputpath + "-indices.txt"
os.makedirs(args.outputdir, exist_ok=True)

# Read Dataset
print("Start: Read Dataset " + graphpath)

hyperedges = []
node_set = set()
with open(graphpath, "r") as f:
    for line in f.readlines():
        nodes = line.rstrip().split(",")
        hyperedge = [int(node) for node in nodes]
        for node in nodes:
            node_set.add(int(node))
        hyperedges.append(hyperedge)

node2edge = [[] for _ in range(len(node_set))]       
for i, edge in enumerate(hyperedges):
    for node in edge:
        node2edge[node].append(i)

if "dataset" not in graphpath:
    with open(indexpath, "r") as f:
        index = f.readline().rstrip().split(",")
        index = [int(i) for i in index]
        
gt_attrs = []
with open(attrpath, "r") as f:
    for line in f.readlines():
        attr = line.rstrip().split(",")
        attr = [float(a) for a in attr]
        gt_attrs.append(attr)

if "dataset" in graphpath:
    attrs = gt_attrs
else:
    attrs = [gt_attrs[i % len(gt_attrs)] for i in index]
        
attrs = np.array(attrs)
attrs[attrs > 0] = 1.
attrs = attrs.astype(int)
attr_dim = attrs.shape[1]

# Metric 1. Node Homophily Score
print("Start: Node Homophily Score")

h_scores = []
for node in range(len(node_set)):
    neighbor_list = []
    for hyperedge in node2edge[node]:
        neighbors = hyperedges[hyperedge].copy()
        neighbors.remove(node)
        neighbor_list += neighbors
    if len(neighbor_list) == 0:
        continue
    score_vec = (len(neighbor_list) - np.sum(np.abs(attrs[neighbor_list] - attrs[node]), axis=0)) / len(neighbor_list)
    h_scores.append(score_vec)
       
h_scores = np.array(h_scores)
np.save(args.outputdir + "node_homophily_score.npy", h_scores)

# for i in range(attr_dim):
#     column_values = h_scores[:, i]

#     plt.figure(figsize=(8, 5))
#     sns.histplot(column_values, bins=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0], stat='count', color='skyblue', edgecolor='black')

#     plt.title(f'Distribution of Node Homophily Score (dimnesion {i + 1})')
#     plt.xlabel('Value')
#     plt.ylabel('Count')
#     plt.grid(True)
#     plt.tight_layout()
#     plt.savefig(args.outputdir + f"node_homophily_score_{i + 1}.png")
#     plt.cla()   
#     plt.clf()   
#     plt.close()

print("End: Node Homophily Score")

# Metric 2. Hyperedge Entropy
print("Start: Hyperedge Entropy")

entropies = []
for hyperedge in hyperedges:
    size = len(hyperedge)
    hyperedge_attrs = attrs[hyperedge]
    attr_sum = np.sum(hyperedge_attrs, axis=0)  
    if size == 1:
        if len(node2edge[hyperedge[0]]) == 1:
            continue
        entropy = np.zeros((attr_dim), dtype=np.float64)
    else:
        valid_entropy_0_idx = attr_sum != size
        valid_entropy_1_idx = attr_sum != 0
        entropy_0 = np.zeros_like(attr_sum, dtype=np.float64)
        entropy_1 = np.zeros_like(attr_sum, dtype=np.float64)
        entropy_0[valid_entropy_0_idx] = -((size - attr_sum[valid_entropy_0_idx]) / size) * np.log((size - attr_sum[valid_entropy_0_idx]) / size)
        entropy_1[valid_entropy_1_idx] = -(attr_sum[valid_entropy_1_idx] / size) * np.log(attr_sum[valid_entropy_1_idx] / size)
        entropy = entropy_0 + entropy_1
    entropies.append(entropy)

entropies = np.array(entropies)
np.save(args.outputdir + "hyperedge_entropy.npy", entropies)

# for i in range(attr_dim):
#     column_values = entropies[:, i]

#     plt.figure(figsize=(8, 5))
#     sns.histplot(column_values, bins=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0], stat='count', color='skyblue', edgecolor='black')

#     plt.title(f'Distribution of Hyperedge Entropy (dimnesion {i + 1})')
#     plt.xlabel('Value')
#     plt.ylabel('Count')
#     plt.grid(True)
#     plt.tight_layout()
#     plt.savefig(args.outputdir + f"hyperedge_entropy_{i + 1}.png")
#     plt.cla()   
#     plt.clf()   
#     plt.close()

# print("End: Hyperedge Entropy")

# Metric 3. Higher Order Hyperedge Entropy
print("Start: Higher Order Hyperedge Entropy")

n = len(node2edge)
m = len(hyperedges)

D_E_inv = np.zeros((m, m))
D_V_inv = np.zeros((n, n))
H = np.zeros((n, m))

for i in range(m):
    D_E_inv[i][i] = 1 / len(hyperedges[i])

for i in range(n):
    D_V_inv[i][i] = 1 / len(node2edge[i])
    H[i][node2edge[i]] = 1.

H_T = H.T

X = np.array(attrs, dtype=np.float64)
Y = np.zeros((m, attr_dim), dtype=np.float64)

for n_iter in range(10):
    for he in range(m):
        cur_edge = hyperedges[he]
        Y[he] = np.sum(X[cur_edge], axis=0) / len(cur_edge)
    for i in range(n):
        cur_node = node2edge[i]
        X[i] = np.sum(Y[cur_node], axis=0) / len(cur_node)
    
entropies = []
for hyperedge in hyperedges:
    size = len(hyperedge)
    hyperedge_attrs = X[hyperedge]
    attr_sum = np.sum(hyperedge_attrs, axis=0)
    if size == 1:
        if len(node2edge[hyperedge[0]]) == 1:
            continue
    valid_entropy_0_idx = attr_sum != size
    valid_entropy_1_idx = attr_sum != 0
    entropy_0 = np.zeros_like(attr_sum, dtype=np.float64)
    entropy_1 = np.zeros_like(attr_sum, dtype=np.float64)
    entropy_0[valid_entropy_0_idx] = -((size - attr_sum[valid_entropy_0_idx]) / size) * np.log((size - attr_sum[valid_entropy_0_idx]) / size)
    entropy_1[valid_entropy_1_idx] = -(attr_sum[valid_entropy_1_idx] / size) * np.log(attr_sum[valid_entropy_1_idx] / size)
    entropy = entropy_0 + entropy_1
    entropies.append(entropy)

entropies = np.array(entropies)
np.save(args.outputdir + "higher_order_hyperedge_entropy.npy", entropies)

# for i in range(attr_dim):
#     column_values = entropies[:, i]

#     plt.figure(figsize=(8, 5))
#     sns.histplot(column_values, bins=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0], stat='count', color='skyblue', edgecolor='black')

#     plt.title(f'Distribution of Higher Order Hyperedge Entropy (dimnesion {i + 1})')
#     plt.xlabel('Value')
#     plt.ylabel('Count')
#     plt.grid(True)
#     plt.tight_layout()
#     plt.savefig(args.outputdir + f"higher_order_hyperedge_entropy_{i + 1}.png")
#     plt.cla()   
#     plt.clf()   
#     plt.close()

print("End: Higher Order Hyperedge Entropy")

# Metric 4~6. Affinity Score 2~4
print("Start: Affinity Score 2~4")

gt_attrs = np.array(gt_attrs)
gt_attrs[gt_attrs > 0] = 1.
gt_attrs = gt_attrs.astype(int)

n = gt_attrs.shape[0]
one_per_feat = np.sum(gt_attrs, axis=0)
zero_per_feat = gt_attrs.shape[0] - one_per_feat

# n = len(node2edge)
# one_per_feat = np.sum(attrs, axis=0)
# zero_per_feat = attrs.shape[0] - one_per_feat

size_to_check = [2,3,4]

type_degree_0 = {}
type_degree_1 = {}
degree_0 = {}
degree_1 = {}
expected_prob_0 = {}
expected_prob_1 = {}
for size in size_to_check:
    type_degree_0[size] = np.zeros((size, attr_dim))
    type_degree_1[size] = np.zeros((size, attr_dim))
    degree_0[size] = np.zeros(attr_dim)
    degree_1[size] = np.zeros(attr_dim)
    expected_prob_0[size] = np.zeros((size, attr_dim))
    expected_prob_1[size] = np.zeros((size, attr_dim))
    
for hyperedge in hyperedges:
    if len(hyperedge) in size_to_check:
        attr_sum = np.sum(attrs[hyperedge], axis=0)
        for i in range(attr_dim):
            if attr_sum[i] != len(hyperedge):
                type_degree_0[len(hyperedge)][len(hyperedge) - attr_sum[i] - 1][i] += len(hyperedge) - attr_sum[i]
            if attr_sum[i] != 0:
                type_degree_1[len(hyperedge)][attr_sum[i] - 1][i] += attr_sum[i]
            degree_0[len(hyperedge)][i] += len(hyperedge) - attr_sum[i]
            degree_1[len(hyperedge)][i] += attr_sum[i]
    
for size in size_to_check:
    for i in range(1, size + 1):
        for feat in range(attr_dim):
            if zero_per_feat[feat] - 1 < i - 1 or n - zero_per_feat[feat] < size - i:
                pass
            else:
                expected_prob_0[size][i - 1][feat] = comb(zero_per_feat[feat] - 1, i - 1) * comb(n - zero_per_feat[feat], size - i) / comb(n - 1, size - 1)
                
            if one_per_feat[feat] - 1 < i - 1 or n - one_per_feat[feat] < size - i:
                continue
            else:
                expected_prob_1[size][i - 1][feat] = comb(one_per_feat[feat] - 1, i - 1) * comb(n - one_per_feat[feat], size - i) / comb(n - 1, size - 1)
                
for size in size_to_check:
    with open(args.outputdir + f"type_{size}_affinity_score_0.txt", "w") as f:
        for feat in range(attr_dim):
            for i in range(1, size + 1):
                if zero_per_feat[feat] - 1 < i - 1 or n - zero_per_feat[feat] < size - i or degree_0[size][feat] == 0:
                    val_0 = 0
                else:
                    val_0 = (type_degree_0[size][i - 1][feat] / degree_0[size][feat]) / expected_prob_0[size][i - 1][feat]
                if i != size:
                    f.write(str(val_0) + ",")
                else:
                    f.write(str(val_0) + "\n")
        
    with open(args.outputdir + f"type_{size}_affinity_score_1.txt", "w") as f:
        for feat in range(attr_dim):
            for i in range(1, size + 1):
                if one_per_feat[feat] - 1 < i - 1 or n - one_per_feat[feat] < size - i or degree_1[size][feat] == 0:
                    val_1 = 0
                else:
                    val_1 = (type_degree_1[size][i - 1][feat] / degree_1[size][feat]) / expected_prob_1[size][i - 1][feat]
                if i != size:
                    f.write(str(val_1) + ",")
                else:
                    f.write(str(val_1) + "\n")

# edge_counts_by_comb = {}
# expected_counts_by_comb = {}
# for size in size_to_check:
#     edge_counts_by_comb[size] = np.zeros((size + 1, attr_dim))
#     expected_counts_by_comb[size] = np.zeros((size + 1, attr_dim))
#     for i in range(size + 1):
#         for feat in range(attr_dim):
#             if one_per_feat[feat] - 1 < i or n - one_per_feat[feat] < size - i:
#                 continue
#             else:
#                 expected_counts_by_comb[size][i][feat] = comb(one_per_feat[feat] - 1, i) * comb(n - one_per_feat[feat], size - i) / comb(n - 1, size - 1)

# for hyperedge in hyperedges:
#     if len(hyperedge) in size_to_check:
#         attr_sum = np.sum(attrs[hyperedge], axis=0)
#         for i in range(attr_dim):
#             edge_counts_by_comb[len(hyperedge)][attr_sum[i]][i] += 1

# for size in size_to_check:
#     with open(args.outputdir + "affinity_score_size_" + str(size) + ".txt", "w") as f:
#         for i in range(size + 1):
#             for feat in range(attr_dim):
#                 if one_per_feat[feat] - 1 < i or n - one_per_feat[feat] < size - i:
#                     val = 0
#                 else:
#                     val = edge_counts_by_comb[size][i][feat] / expected_counts_by_comb[size][i][feat]
#                 if feat != attr_dim - 1 or i != size:
#                     f.write(str(val) + ",")
#                 else:
#                     f.write(str(val))
#         f.write("\n")

print("End: Affinity Score 2~4")
