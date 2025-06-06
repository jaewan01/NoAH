import os
import sys
import argparse
from collections import defaultdict
from tqdm import tqdm
import numpy as np
from itertools import chain
import networkx as nx
from numpy.linalg import svd
from scipy.sparse import coo_matrix, csc_matrix
from scipy.sparse.linalg import svds, eigs, norm
from scipy.stats import skew, kendalltau
import snap
import time
import math
import copy

# ----------------------------------------------------------------------------------------------

RESULTDIR = "../results/"

parser = argparse.ArgumentParser()
parser.add_argument('--inputpath', required=False)
parser.add_argument('--outputdir', required=False)


parser.add_argument('--effdiam', required=False, action='store_true')
parser.add_argument('--sv', required=False, action='store_true')
parser.add_argument('--dupflag', required=False, action='store_true')
parser.add_argument('--appxflag', required=False, action='store_true')


args = parser.parse_args()
outputdir = args.outputdir



print("Start " + outputdir)

appx_flag = False
appx_flag2 = False
for dataname in ["threads-ask-ubuntu", "tags-ask-ubuntu", "tags-math-sx"]:
    if dataname in args.inputpath:
        appx_flag = True
for dataname in ["coauth-DBLP-full", "threads-math-sx", "coauth-MAG-Geology-full","trivago-clicks", "walmart-trips"]:
    if dataname in args.inputpath:
        appx_flag2 = True

entire_nodeset = set([])
entire_number_of_nodes = 0
entire_number_of_hyperedges = 0
datapath = args.inputpath + ".txt"
datanode2index = {}
hyperedge_list = []
hedge2check = {}

if os.path.isfile(datapath):
    with open(datapath, "r") as f:
        for line in f.readlines():
            hyperedge = line[:-1].split(",")
            for n in hyperedge:
                if int(n) not in datanode2index:
                    datanode2index[int(n)] = len(datanode2index)
                nodeindex = datanode2index[int(n)]
                entire_nodeset.add(nodeindex)
            tmp = [datanode2index[int(_n)] for _n in hyperedge]
            hyperedge = sorted(list(set(tmp)))
            hyperedge_key = tuple(hyperedge)
            if args.dupflag or (hyperedge_key not in hedge2check):
                if hyperedge_key not in hedge2check:
                    hedge2check[hyperedge_key] = len(hedge2check)
                hyperedge_list.append(hyperedge)
                entire_number_of_hyperedges += 1
    entire_number_of_nodes = len(entire_nodeset)

    if args.effdiam:
        inputpath = datapath
        
        pg = snap.TUNGraph.New()
        nodeset = set([])
        number_of_edges = 0

        start_time = time.time()
        for hyperedge in hyperedge_list:
            number_of_edges += 1
            for i, nodeindex in enumerate(hyperedge):
                if nodeindex not in nodeset:
                    nodeset.add(nodeindex)
                    pg.AddNode(nodeindex)
            # Add clique edges
            if len(hyperedge) != 1: 
                for i in range(0, len(hyperedge)-1):
                    for j in range(i+1, len(hyperedge)):
                        i1, i2 = min(hyperedge[i], hyperedge[j]), max(hyperedge[i], hyperedge[j])
                        ret = pg.AddEdge(i1, i2)
                        
        num_nodes = len(nodeset)
        if appx_flag:
            effective_diameter = snap.GetBfsEffDiam(pg, num_nodes if num_nodes < 5000 else 1000, False)
        elif appx_flag2:
            effective_diameter = snap.GetBfsEffDiam(pg, num_nodes if num_nodes < 5000 else 100, False)
        else:
            effective_diameter = snap.GetBfsEffDiam(pg, num_nodes if num_nodes < 5000 else 5000, False)
        
        # Save Effective Diameter
        with open(outputdir + "effdiameter.txt", "w") as f:
            f.write(str(effective_diameter) + "\n")
        print("End EffDiameter")

    if args.sv:               
        inputpath = datapath
        assert(os.path.isfile(inputpath))

        outputpath = outputdir + "sv_all.txt"
        if entire_number_of_hyperedges == 0:
            sys.exit("Input Empty")

        node2edges = defaultdict(list)
        for idx, nodes in enumerate(hyperedge_list):
            for v in nodes:
                node2edges[int(v)].append(idx)

        dim = min(entire_number_of_hyperedges, entire_number_of_nodes)
        s = -1
        flag = False
        # if "tags-" in inputpath:
        #     flag = True
        #     dim = min(dim - 1, 1000)
        # elif "trivago-" in inputpath:
        #     flag = True
        #     dim = min(dim - 1, 1000)
        # elif "walmart-" in inputpath:
        #     flag = True
        #     dim = min(dim - 1, 1000)
        # elif "threads-" in inputpath:
        #     flag = True
        #     dim = min(dim - 1, 1000)
        # elif "coauth-" in inputpath:
        #     flag = True
        #     dim = min(dim - 1, 500)
        # elif args.appxflag:
        #     flag = True
        #     dim = min(dim - 1, 1000)

        if flag is False:
            # Compute full rank singular values
            try:
                incident_matrix = np.zeros(shape=(entire_number_of_nodes, entire_number_of_hyperedges), dtype=np.byte)
                for v in node2edges.keys():
                    for edge_idx in node2edges[v]:
                        incident_matrix[v,edge_idx] = 1          
                s = svd(incident_matrix, compute_uv=False)
                s = sorted(list(s), reverse=True)
                assert len(s) == dim
                nonzero_s = []
                for i in range(1,dim):
                    assert s[i-1] >= s[i]
                for i in range(dim):
                    if s[i] != 0:
                        nonzero_s.append(s[i])
                with open(outputpath, "w") as f:
                    for _sv in nonzero_s:
                        f.write(str(_sv) + "\n")
            except:
                rows, cols = zip(*chain.from_iterable([[(v, edge_idx) for edge_idx in node2edges[v]] for v in node2edges.keys()]))
                nnz = len(rows)
                incident_matrix = coo_matrix((np.ones(nnz), (rows, cols)), shape=(entire_number_of_nodes, entire_number_of_hyperedges))
                sum_of_squares = norm(incident_matrix, 'fro') ** 2
                rank = dim - 1 #min(num , dim - 1)
                _, s, _ = svds(incident_matrix.tocsc(), k=rank)
                last_sv_square = sum_of_squares - sum([_s * _s for _s in s])
                last_sv = math.sqrt(last_sv_square)
                s = list(s)
                s.append(last_sv)
                assert len(s) == dim
                s = sorted(s, reverse=True)
                for i in range(1,dim):
                    assert s[i-1] >= s[i]
                nonzero_s = []
                for i in range(dim):
                    if s[i] != 0:
                        nonzero_s.append(s[i])
                with open(outputpath, "w") as f:
                    for _sv in nonzero_s:
                        f.write(str(_sv) + "\n")

        else:
            # Compute top-dim singular values
            rows, cols = zip(*chain.from_iterable([[(v, edge_idx) for edge_idx in node2edges[v]] for v in node2edges.keys()]))
            nnz = len(rows)
            incident_matrix = coo_matrix((np.ones(nnz), (rows, cols)), shape=(entire_number_of_nodes, entire_number_of_hyperedges))
            # incident_matrix = np.zeros(shape=(entire_number_of_nodes, entire_number_of_hyperedges), dtype=np.byte)
            _, s, _ = svds(incident_matrix.tocsc(), k=dim)
            s = list(s)
            s = sorted(s, reverse=True)
            for i in range(1,dim):
                assert s[i-1] >= s[i]
            nonzero_s = []
            for i in range(dim):
                if s[i] != 0:
                    nonzero_s.append(s[i])
            with open(outputpath, "w") as f:
                for _sv in nonzero_s:
                    f.write(str(_sv) + "\n")
                    
                    
        outputpath = outputdir + "sv.txt"
        list_sv = copy.deepcopy(nonzero_s)
        singular_values = {}
        
        denom = sum([_s * _s for _s in list_sv])
        until = 0.0
        total = dim
        for idx in range(len(list_sv)):
            proportion = (idx + 1) / total
            until += list_sv[idx] ** 2
            singular_values[proportion] = until / denom
            
        with open(outputpath, "w") as f:
            for x, y in singular_values.items():
                f.write(str(x) + "," + str(y) + "\n")

        print("End Singular Value Dist")
