import argparse
import os
import torch
from utils import *
from fit import *
from model import *


def main(
    target,
    mode,
    iter,
    epoch,
    lr_c,
    lr_f,
    w_d,
    w_s,
    n_batch_c,
    n_batch_f,
    seed,
    device,
    k=None,
    neural_hidden_dim=128
):
    
    fix_seed(seed)
    
    if mode == "NoAH_continuous":
        # Step 1. Read target hypergraph and split a node set into core and fringes.
        hyperedges, attributes, n, m, k = prep_dataset_raw_attr(target)

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

        print(f'Target Hypergraph Info: {target}, nc: {nc}, nf: {nf}, m: {m}, k: {k}')
            
        # Step 2. Estimate all parameters.
        Fc = attributes[cores]
        Ff = attributes[fringes]
        if os.path.exists(f"./parameters/{target}/{mode}/seed_prob-{iter}-{epoch}-{lr_c}-{w_d}-{w_s}-{seed}.pt"):
            seed_prob = torch.load(f"./parameters/{target}/{mode}/seed_prob-{iter}-{epoch}-{lr_c}-{w_d}-{w_s}-{seed}.pt")
            theta_c = torch.load(f"./parameters/{target}/{mode}/Tc-{iter}-{epoch}-{lr_c}-{w_d}-{w_s}-{seed}.pt")
            theta_f = torch.load(f"./parameters/{target}/{mode}/Tf-{iter}-{epoch}-{lr_c}-{w_d}-{w_s}-{seed}.pt")
            alphas = torch.load(f"./parameters/{target}/{mode}/alphas-{iter}-{epoch}-{lr_c}-{w_d}-{w_s}-{seed}.pt")
            betas = torch.load(f"./parameters/{target}/{mode}/betas-{iter}-{epoch}-{lr_c}-{w_d}-{w_s}-{seed}.pt")
        else:
            Ic = torch.zeros(m, nc)
            for edge_idx, nodes in enumerate(hyperedges):
                for node in nodes:
                    if node in cores:
                        Ic[edge_idx, cores.index(node)] = 1.  
            If = torch.zeros(m, nf)
            for edge_idx, nodes in enumerate(hyperedges):
                for node in nodes:
                    if node in fringes:
                        If[edge_idx, fringes.index(node)] = 1.
            theta_c, theta_f, alphas, betas, seed_prob = NoAHfit_continuous(Ic, If, Fc, Ff, epoch, lr_c, w_d, w_s, n_batch_c, seed, device)
            os.makedirs(f"./parameters/{target}/{mode}", exist_ok=True)
            torch.save(seed_prob, f"./parameters/{target}/{mode}/seed_prob-{iter}-{epoch}-{lr_c}-{w_d}-{w_s}-{seed}.pt")
            torch.save(theta_c, f"./parameters/{target}/{mode}/Tc-{iter}-{epoch}-{lr_c}-{w_d}-{w_s}-{seed}.pt")
            torch.save(theta_f, f"./parameters/{target}/{mode}/Tf-{iter}-{epoch}-{lr_c}-{w_d}-{w_s}-{seed}.pt")
            torch.save(alphas, f"./parameters/{target}/{mode}/alphas-{iter}-{epoch}-{lr_c}-{w_d}-{w_s}-{seed}.pt")
            torch.save(betas, f"./parameters/{target}/{mode}/betas-{iter}-{epoch}-{lr_c}-{w_d}-{w_s}-{seed}.pt")

        # Step 3. Generate a hypergraph using seed_prob, theta_c, and theta_f.
        dirname = "noah_continuous"
        if not os.path.exists(f"../generated/{dirname}/{target}/{dirname}-{iter}-{lr_c}-{lr_f}-{w_d}-{w_s}-{epoch}-{seed}-preindexing.txt"):
            hypergraph = NoAH_continuous(Fc, Ff, theta_c, theta_f, alphas, betas, seed_prob, m).e2n
            os.makedirs(f"../generated/{dirname}/{target}", exist_ok=True)
            with open(f"../generated/{dirname}/{target}/{dirname}-{iter}-{lr_c}-{lr_f}-{w_d}-{w_s}-{epoch}-{seed}-preindexing.txt", "w") as f:
                for hyperedge in hypergraph:
                    cur = []
                    for node in hyperedge:
                        if node < len(cores):
                            cur.append(str(cores[node]))
                        else:
                            cur.append(str(fringes[node - len(cores)]))
                    f.write(",".join(cur) + "\n")
    
    elif mode == "NoAH_neural":
        # Step 1. Read target hypergraph and split a node set into core and fringes.
        hyperedges, attributes, n, m, k = prep_dataset_raw_attr(target)

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

        print(f'Target Hypergraph Info: {target}, nc: {nc}, nf: {nf}, m: {m}, k: {k}')

        # Step 2. Estimate all parameters.
        Fc = attributes[cores]
        Ff = attributes[fringes]
        seed_prob_path = f"./parameters/{target}/{mode}/seed_prob-{iter}-{epoch}-{lr_c}-{w_d}-{w_s}-{seed}.pt"
        core_mlp_path = f"./parameters/{target}/{mode}/core_mlp-{iter}-{epoch}-{lr_c}-{w_d}-{w_s}-{seed}.pt"
        fringe_mlp_path = f"./parameters/{target}/{mode}/fringe_mlp-{iter}-{epoch}-{lr_f}-{w_d}-{w_s}-{seed}.pt"

        Ic = torch.zeros(m, nc)
        for edge_idx, nodes in enumerate(hyperedges):
            for node in nodes:
                if node in cores:
                    Ic[edge_idx, cores.index(node)] = 1.
        If = torch.zeros(m, nf)
        for edge_idx, nodes in enumerate(hyperedges):
            for node in nodes:
                if node in fringes:
                    If[edge_idx, fringes.index(node)] = 1.

        core_group_count = torch.sum(Ic, dim=1, keepdim=True).clamp_min(1.0)
        Fcg = torch.matmul(Ic, torch.FloatTensor(Fc)) / core_group_count

        if os.path.exists(seed_prob_path) and os.path.exists(core_mlp_path):
            seed_prob = torch.load(seed_prob_path, map_location="cpu")
            core_mlp_ckpt = torch.load(core_mlp_path, map_location="cpu")
        else:
            core_mlp_ckpt, seed_prob = NoAHfit_continuous_neural_core(
                Ic,
                Fc,
                epoch,
                lr_c,
                w_d,
                w_s,
                n_batch_c,
                seed,
                device,
                hidden_dim=neural_hidden_dim
            )
            os.makedirs(f"./parameters/{target}/{mode}", exist_ok=True)
            torch.save(seed_prob, seed_prob_path)
            torch.save(core_mlp_ckpt, core_mlp_path)

        if os.path.exists(fringe_mlp_path):
            fringe_mlp_ckpt = torch.load(fringe_mlp_path, map_location="cpu")
        else:
            fringe_mlp_ckpt = NoAHfit_continuous_neural_fringe(
                If,
                Ff,
                Fcg,
                epoch,
                lr_f,
                w_d,
                w_s,
                n_batch_f,
                seed,
                device,
                hidden_dim=neural_hidden_dim
            )
            os.makedirs(f"./parameters/{target}/{mode}", exist_ok=True)
            torch.save(fringe_mlp_ckpt, fringe_mlp_path)
        
        # Step 3. Generate a hypergraph using neural interaction modules.
        dirname = "noah_neural"
        output_path = f"../generated/{dirname}/{target}/{dirname}-{iter}-{lr_c}-{lr_f}-{w_d}-{w_s}-{epoch}-{seed}-preindexing.txt"
        if not os.path.exists(output_path):
            hypergraph = NoAH_neural(Fc, Ff, core_mlp_ckpt, fringe_mlp_ckpt, seed_prob, m, mode).e2n
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
        "-mode",
        "--mode",
        default='HyperCF',
        action="store",
        type=str,
        help="Choose from [NoAH, Bipartite, NoAH_continuous_v2, NoAH_continuous_neural]."
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
        default="0",
        action="store",
        type=int,
        help="Number of attribute for fitting without attribute.",
    )

    parser.add_argument(
        "--neural_hidden_dim",
        default=8,
        action="store",
        type=int,
        help="Hidden dimension for neural NoAH continuous mode.",
    )

    args = parser.parse_args()
    main(
        args.target_hypergraph,
        args.mode,
        args.recovery_iter,
        args.train_epoch,
        args.learning_rate_core,
        args.learning_rate_fringe,
        args.weight_degree,
        args.weight_size,
        args.core_batch_num,
        args.fringe_batch_num,
        args.random_seed,
        args.device,
        args.attr_dim,
        args.neural_hidden_dim,
    )
