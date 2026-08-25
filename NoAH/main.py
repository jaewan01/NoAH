import argparse
import os
import torch
from utils import *
from fit import *
from model import *


def main(target, mode, iter, epoch, lr_c, lr_f, w_d, w_s, n_batch_c, n_batch_f, seed, device):
    
    fix_seed(seed)
    
    print(mode)
    if mode in ["NoAH", "NoAH-dCF"]:
        # Step 1. Read target hypergraph and split a node set into core and fringes.
        hyperedges, attributes, n, m, k = prep_dataset(target)

        if mode == "NoAH":
            cores, fringes = load_or_create_umhs_split(target, iter)
        else:
            cores, fringes = load_or_create_degree_size_matched_split(target, iter)

        nc = len(cores)
        nf = len(fringes)

        print(f'Target Hypergraph: {target}, nc: {nc}, nf: {nf}, m: {m}, k: {k}')
            
        # Step 2. Estimate seed_prob & theta_c by core group construction.
        Fc = attributes[cores]
        Ic = torch.zeros(m, nc)
        for edge_idx, nodes in enumerate(hyperedges):
            for node in nodes:
                if node in cores:
                    Ic[edge_idx, cores.index(node)] = 1

        if mode == "NoAH-dCF":
            valid_edge_mask = torch.sum(Ic, dim=1) > 0
            fit_edge_indices = torch.nonzero(valid_edge_mask, as_tuple=False).squeeze(1).tolist()
            if len(fit_edge_indices) == 0:
                raise ValueError(f"{mode} found zero covered hyperedges for {target}.")
            Ic_fit = Ic[valid_edge_mask]
            print(f"{mode} fitting uses {len(fit_edge_indices)}/{m} hyperedges with at least one core node.")
        else:
            fit_edge_indices = list(range(m))
            Ic_fit = Ic

        if os.path.exists(f"./parameters/{target}/{mode}/seed_prob-{iter}-{epoch}-{lr_c}-{w_d}-{w_s}-{seed}.pt"):
            seed_prob = torch.load(f"./parameters/{target}/{mode}/seed_prob-{iter}-{epoch}-{lr_c}-{w_d}-{w_s}-{seed}.pt")
            theta_c = torch.load(f"./parameters/{target}/{mode}/Tc-{iter}-{epoch}-{lr_c}-{w_d}-{w_s}-{seed}.pt")
        else:
            theta_c, seed_prob = NoAHfit_core(Ic_fit, Fc, epoch, lr_c, w_d, w_s, n_batch_c, seed, device)
            os.makedirs(f"./parameters/{target}/{mode}", exist_ok=True)
            torch.save(seed_prob, f"./parameters/{target}/{mode}/seed_prob-{iter}-{epoch}-{lr_c}-{w_d}-{w_s}-{seed}.pt")
            torch.save(theta_c, f"./parameters/{target}/{mode}/Tc-{iter}-{epoch}-{lr_c}-{w_d}-{w_s}-{seed}.pt")
        
        # Step 3. Estimate theta_f by fringe attachment.
        Ff = attributes[fringes]
        if os.path.exists(f"./parameters/{target}/{mode}/Tf-{iter}-{epoch}-{lr_f}-{w_d}-{w_s}-{seed}.pt"):
            theta_f = torch.load(f"./parameters/{target}/{mode}/Tf-{iter}-{epoch}-{lr_f}-{w_d}-{w_s}-{seed}.pt")
        else:
            If = torch.zeros(len(fit_edge_indices), nf)
            for fit_idx, edge_idx in enumerate(fit_edge_indices):
                nodes = hyperedges[edge_idx]
                for node in nodes:
                    if node in fringes:
                        If[fit_idx, fringes.index(node)] = 1.

            Fcg = torch.zeros((len(fit_edge_indices), k), dtype=torch.float32)
            for fit_idx, edge_idx in enumerate(fit_edge_indices):
                nodes = hyperedges[edge_idx]
                cur_cg = [node for node in nodes if node in cores]
                if len(cur_cg) == 0:
                    raise ValueError(f"Encountered uncovered hyperedge during fringe fitting in {mode}.")
                if len(cur_cg) == 1:
                    Fcg[fit_idx] = torch.FloatTensor(attributes[cur_cg[0]])
                else:
                    avg_attr = torch.mean(attributes[cur_cg, :], dim=0)
                    Fcg[fit_idx] = torch.FloatTensor(avg_attr)
            theta_f = NoAHfit_fringe(If, Ff, Fcg, epoch, lr_f, w_d, w_s, n_batch_f, seed, device)

        os.makedirs(f"./parameters/{target}/{mode}", exist_ok=True)
        torch.save(theta_f, f"./parameters/{target}/{mode}/Tf-{iter}-{epoch}-{lr_f}-{w_d}-{w_s}-{seed}.pt")
            
        # Step 4. Generate a hypergraph using seed_prob, theta_c, and theta_f.
        hypergraph = NoAH(Fc, Ff, theta_c, theta_f, seed_prob, m, mode).e2n

        if mode == "NoAH":
            dirname = "NoAH"
        else:
            dirname = "NoAH_dCF"

        os.makedirs(f"../generated/{dirname}/{target}", exist_ok=True)
        with open(f"../generated/{dirname}/{target}/{mode}-{iter}-{lr_c}-{lr_f}-{w_d}-{w_s}-{epoch}-{seed}-preindexing.txt", "w") as f:
            for hyperedge in hypergraph:
                cur = []
                for node in hyperedge:
                    if node < len(cores):
                        cur.append(str(cores[node]))
                    else:
                        cur.append(str(fringes[node - len(cores)]))
                f.write(",".join(cur) + "\n")
        
        reindexing(target, dirname)
    
    elif mode == "NoAH_noCF":
        # Step 1. Read target hypergraph.
        hyperedges, attributes, n, m, k = prep_dataset(target)
        print(f'Target Hypergraph: {target}, n: {n}, m: {m}, k: {k}')
        
        # Step2. Estimate seed_prob & theta.
        if os.path.exists(f"./parameters/{target}/{mode}/seed_prob-{iter}-{epoch}-{lr_c}-{w_d}-{w_s}-{seed}.pt"):
            seed_prob = torch.load(f"./parameters/{target}/{mode}/seed_prob-{iter}-{epoch}-{lr_c}-{w_d}-{w_s}-{seed}.pt")
            theta = torch.load(f"./parameters/{target}/{mode}/T-{iter}-{epoch}-{lr_c}-{w_d}-{w_s}-{seed}.pt")
        else:
            I = torch.zeros(m, n)
            for edge_idx, nodes in enumerate(hyperedges):
                for node in nodes:
                    I[edge_idx, node] = 1
            lr = lr_c
            theta, seed_prob = NoAHfit_core(I, attributes, epoch, lr, w_d, w_s, n_batch_c, seed, device)
            os.makedirs(f"./parameters/{target}/{mode}", exist_ok=True)
            torch.save(seed_prob, f"./parameters/{target}/{mode}/seed_prob-{iter}-{epoch}-{lr_c}-{w_d}-{w_s}-{seed}.pt")
            torch.save(theta, f"./parameters/{target}/{mode}/T-{iter}-{epoch}-{lr_c}-{w_d}-{w_s}-{seed}.pt")
        
        # Step 3. Generate a hypergraph using seed_prob and theta.
        hypergraph = NoAH_noCF(attributes, theta, seed_prob, m).e2n


        os.makedirs(f"../generated/{mode}/{target}", exist_ok=True)
        with open(f"../generated/{mode}/{target}/{mode}-{iter}-{lr_c}-{lr_f}-{w_d}-{w_s}-{epoch}-{seed}-preindexing.txt", "w") as f:
            for hyperedge in hypergraph:
                cur = []
                for node in hyperedge:
                    cur.append(str(node))
                f.write(",".join(cur) + "\n")

        reindexing(target, mode)
    
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        "-target",
        "--target_hypergraph",
        default='cora_coauth',
        action="store",
        type=str,
        help="Select the target real-world hypergraph."
    )
    
    parser.add_argument(
        "-mode",
        "--mode",
        default='NoAH',
        action="store",
        type=str,
        help="Choose from [NoAH, NoAH_dCF, NoAH_noCF]."
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
    
    args = parser.parse_args()
    main(args.target_hypergraph, args.mode, args.recovery_iter, args.train_epoch, args.learning_rate_core, args.learning_rate_fringe, args.weight_degree, args.weight_size, args.core_batch_num, args.fringe_batch_num, args.random_seed, args.device) 
