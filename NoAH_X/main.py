import argparse
import os
import torch
from utils import *
from fit import *
from model import *


def main(target, iter, epoch, lr_c, lr_f, w_d, w_s, n_batch_c, n_batch_f, seed, device, k=None):
    
    fix_seed(seed)

    mode = "NoAH_X"
    
    # Step 1. Read target hypergraph and split a node set into core and fringes.
    hyperedges, n, m = prep_dataset_wo_attr(target)
    assert k is not None and k > 0, "Please specify a positive attribute dimension k for NoAH_X."
    k_tag = f"k{k}"

    if os.path.exists(f"core-fringe-split/{target}/{iter}"):
        with open(f"core-fringe-split/{target}/{iter}/cores.txt", "r") as f:
            cores = [int(i) for i in f.read().strip().split(",")]

        with open(f"core-fringe-split/{target}/{iter}/fringes.txt", "r") as f:
            fringes = [int(i) for i in f.read().strip().split(",")]
    else:
        cores, fringes = UMHS(data_name = target, iter = iter)
        os.makedirs(f"core-fringe-split/{target}/{iter}")

        with open(f"core-fringe-split/{target}/{iter}/cores.txt", "w") as f:
            core_to_write = [str(i) for i in cores]
            f.write(",".join(core_to_write))
        
        with open(f"core-fringe-split/{target}/{iter}/fringes.txt", "w") as f:
            fringe_to_write = [str(i) for i in fringes]
            f.write(",".join(fringe_to_write))

    nc = len(cores)
    nf = len(fringes)

    print(f'Target Hypergraph Info: {target}, nc: {nc}, nf: {nf}, m: {m}')
    print(f'Target attribute dimension: {k}')
            
    # Step 2. Estimate all parameters.
    if os.path.exists(f"./parameters/{target}/{mode}/seed_prob-{iter}-{epoch}-{lr_c}-{w_d}-{w_s}-{seed}-{k_tag}.pt"):
        seed_prob = torch.load(f"./parameters/{target}/{mode}/seed_prob-{iter}-{epoch}-{lr_c}-{w_d}-{w_s}-{seed}-{k_tag}.pt")
        theta_c = torch.load(f"./parameters/{target}/{mode}/Tc-{iter}-{epoch}-{lr_c}-{w_d}-{w_s}-{seed}-{k_tag}.pt")
        theta_f = torch.load(f"./parameters/{target}/{mode}/Tf-{iter}-{epoch}-{lr_f}-{w_d}-{w_s}-{seed}-{k_tag}.pt")
        Fc = torch.load(f"./parameters/{target}/{mode}/Fc-{iter}-{epoch}-{lr_c}-{w_d}-{w_s}-{seed}-{k_tag}.pt")
        Ff = torch.load(f"./parameters/{target}/{mode}/Ff-{iter}-{epoch}-{lr_f}-{w_d}-{w_s}-{seed}-{k_tag}.pt")
    else:
        Ic = torch.zeros(m, nc)
        If = torch.zeros(m, nf)
        for edge_idx, nodes in enumerate(hyperedges):
            for node in nodes:
                if node in cores:
                    Ic[edge_idx, cores.index(node)] = 1.  
                if node in fringes:
                    If[edge_idx, fringes.index(node)] = 1.
        assert lr_c == lr_f, "Please set the same learning rate for core and fringe fitting in NoAH_X"
        assert n_batch_c == n_batch_f, "Please set the same number of batches for core and fringe fitting in NoAH_X"
        theta_c, theta_f, seed_prob, Fc, Ff = NoAHfit_X(Ic, If, k, epoch, lr_c, w_d, w_s, n_batch_c, seed, device)
        os.makedirs(f"./parameters/{target}/{mode}", exist_ok=True)
        torch.save(seed_prob, f"./parameters/{target}/{mode}/seed_prob-{iter}-{epoch}-{lr_c}-{w_d}-{w_s}-{seed}-{k_tag}.pt")
        torch.save(theta_c, f"./parameters/{target}/{mode}/Tc-{iter}-{epoch}-{lr_c}-{w_d}-{w_s}-{seed}-{k_tag}.pt")
        torch.save(theta_f, f"./parameters/{target}/{mode}/Tf-{iter}-{epoch}-{lr_f}-{w_d}-{w_s}-{seed}-{k_tag}.pt")
        torch.save(Fc, f"./parameters/{target}/{mode}/Fc-{iter}-{epoch}-{lr_c}-{w_d}-{w_s}-{seed}-{k_tag}.pt")
        torch.save(Ff, f"./parameters/{target}/{mode}/Ff-{iter}-{epoch}-{lr_f}-{w_d}-{w_s}-{seed}-{k_tag}.pt")

    # Step 3. Generate hypergraphs for both NoAH_wo_attr variants.
    generation_specs = [
        ("NoAH_X", NoAH_X),
        ("NoAH_X+", NoAH_X_plus),
    ]

    for dirname, model_cls in generation_specs:
        output_path = f"../generated/{dirname}/{target}/{dirname}-{iter}-{lr_c}-{lr_f}-{w_d}-{w_s}-{epoch}-{seed}-{k_tag}-preindexing.txt"

        hypergraph = model_cls(Fc, Ff, theta_c, theta_f, seed_prob, m).e2n

        os.makedirs(f"../generated/{dirname}/{target}", exist_ok=True)
        with open(output_path, "w") as f:
            for hyperedge in hypergraph:
                cur = []
                for node in hyperedge:
                    if node < len(cores):
                        cur.append(str(cores[node]))
                    else:
                        cur.append(str(fringes[node - len(cores)]))
                f.write(",".join(cur) + "\n")
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        "-target",
        "--target_hypergraph",
        default='coauth-cora',
        action="store",
        type=str,
        help="Select the target real-world hypergraph."
    )
    
    parser.add_argument(
        "-iter",
        "--recovery_iter",
        default=10,
        action="store",
        type=int,
        help="Select the number of core recovery iterations."
    )
    
    parser.add_argument(
        "-epoch",
        "--train_epoch",
        default=500,
        action="store",
        type=int,
        help="Select the training epoch."
    )
    
    parser.add_argument(
        "-lr_c",
        "--learning_rate_core",
        default=1e-2,
        action="store",
        type=float,
        help="Select the learning rate for core fitting."
    )
    
    parser.add_argument(
        "-lr_f",
        "--learning_rate_fringe",
        default=1e-2,
        action="store",
        type=float,
        help="Select the learning rate for fringe fitting."
    )
    
    parser.add_argument(
        "-w_d",
        "--weight_degree",
        default=1e-2,
        action="store",
        type=float,
        help="Select the weight of degree distribution loss."
    )
    
    parser.add_argument(
        "-w_s",
        "--weight_size",
        default=1e-2,
        action="store",
        type=float,
        help="Select the weight of size distribution loss."
    )
    
    parser.add_argument(
        "-seed",
        "--random_seed",
        default=1,
        action="store",
        type=int,
        help="Select the random seed for reproducing.",
    )
    
    parser.add_argument(
        "-device",
        "--device",
        default="cuda:0",
        action="store",
        type=str,
        help="GPU device.",
    )

    parser.add_argument(
        "-n_batch_c",
        "--core_batch_num",
        default="0",
        action="store",
        type=int,
        help="Number of batches for core fitting.",
    )

    parser.add_argument(
        "-n_batch_f",
        "--fringe_batch_num",
        default="0",
        action="store",
        type=int,
        help="Number of batches for fringe fitting.",
    )

    parser.add_argument(
        "-k",
        "--attr_dim",
        default=0,
        action="store",
        type=int,
        help="Number of attribute for fitting without attribute.",
    )
     
    args = parser.parse_args()
    main(args.target_hypergraph, args.recovery_iter, args.train_epoch, args.learning_rate_core, args.learning_rate_fringe, args.weight_degree, args.weight_size, args.core_batch_num, args.fringe_batch_num, args.random_seed, args.device, args.attr_dim) 
