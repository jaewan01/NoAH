import torch
import tqdm
from torch import nn

class NoAH_continuous:
    
    """
        NoAH is a model which generates a hypergraph soley based on node attributes.
        
        step 1. Construct core group using core node attributes, seed core probability, and core affinity matrix.
        step 2. Mix the node attributes within each core group to generate the attribute of core group.
        step 3. Construct hyperedge by attaching fringe nodes to core groups using fringe node attributes, core group attributes, and fringe affinity matrix.  
    """
    
    def __init__(self, core_attr, fringe_attr, core_affinity_matrix, fringe_affinity_matrix, alphas, betas,seed_prob, edge_num):
        self.nc = core_attr.shape[0]
        self.nf = fringe_attr.shape[0]
        self.n = self.nc + self.nf
        self.c2a = torch.nn.Sigmoid()(alphas * core_attr + betas)
        self.f2a = torch.nn.Sigmoid()(alphas * fringe_attr + betas)
        self.m = edge_num
        self.k = core_attr.shape[1]
        self.Tc = core_affinity_matrix
        self.Tf = fringe_affinity_matrix
        self.seed_prob = seed_prob
        self.e2n = [[] for _ in range(self.m)]
        self.construct_hypergraph()
   
    def construct_hypergraph(self):
        core_core_probs = torch.ones((self.nc, self.nc))
        for i in range(self.k):
            cur_core_prob_0_0 = (1 - self.c2a[:, i].reshape(-1, 1)) * (1 - self.c2a[:, i].reshape(1, -1)) * self.Tc[i][0]
            cur_core_prob_0_1 = ((1 - self.c2a[:, i].reshape(-1, 1)) * self.c2a[:, i].reshape(1, -1) + 
                                 self.c2a[:, i].reshape(-1, 1) * (1 - self.c2a[:, i].reshape(1, -1))) * self.Tc[i][1]
            cur_core_prob_1_1 = self.c2a[:, i].reshape(-1, 1) * self.c2a[:, i].reshape(1, -1) * self.Tc[i][2]
            core_core_probs *= (cur_core_prob_0_0 + cur_core_prob_0_1 + cur_core_prob_1_1)

        core_core_probs.fill_diagonal_(0)

        for num_edge in tqdm.trange(self.m):
            core_seed = torch.multinomial(self.seed_prob, 1).item()
            core_group = [core_seed]

            cur_core_probs = core_core_probs[core_seed]
            cur_attached_cores = torch.bernoulli(cur_core_probs).to(torch.int)
            cur_attached_cores = torch.nonzero(cur_attached_cores).squeeze().tolist()

            if isinstance(cur_attached_cores, int): 
                cur_attached_cores = [cur_attached_cores]
            if len(cur_attached_cores) > 0:
                core_group.extend(cur_attached_cores)

            if len(core_group) > 1:
                e2p = torch.mean(self.c2a[core_group], dim=0)
            else:
                e2p = self.c2a[core_group[0]]
            edge_attr = e2p.expand(self.nf, -1)

            cur_fringe_probs = torch.ones(self.nf)
            for i in range(self.k):
                cur_fringe_prob_0_0 = (1 - self.f2a[:, i]) * (1 - edge_attr[:, i]) * self.Tf[i][0]
                cur_fringe_prob_0_1 = ((1 - self.f2a[:, i]) * edge_attr[:, i] + 
                                       self.f2a[:, i] * (1 - edge_attr[:, i])) * self.Tf[i][1]
                cur_fringe_prob_1_1 = self.f2a[:, i] * edge_attr[:, i] * self.Tf[i][2]
                cur_fringe_probs *= (cur_fringe_prob_0_0 + cur_fringe_prob_0_1 + cur_fringe_prob_1_1)

            cur_attached_fringes = torch.bernoulli(cur_fringe_probs).to(torch.int)
            cur_attached_fringes = torch.nonzero(cur_attached_fringes).squeeze().tolist()

            if isinstance(cur_attached_fringes, int):
                cur_attached_fringes = [cur_attached_fringes]
            if len(cur_attached_fringes) > 0:
                self.e2n[num_edge] = core_group + [node + self.nc for node in cur_attached_fringes]
            else:
                self.e2n[num_edge] = core_group


class AttributeWiseMLP(nn.Module):
    def __init__(self, k, hidden_dim=4):
        super().__init__()
        self.k = k
        self.feature = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.ReLU(),
        )
        self.out_w = nn.Parameter(torch.ones(k, hidden_dim))
        self.out_b = nn.Parameter(torch.zeros(k))

    def pair_log_probs(self, left_attr, right_attr):
        n_left, k = left_attr.shape
        n_right = right_attr.shape[0]

        left = left_attr.T.unsqueeze(-1)
        right = right_attr.T.unsqueeze(1)

        left = left.expand(-1, n_left, n_right)
        right = right.expand(-1, n_left, n_right)

        cur_in = torch.stack((left, right), dim=-1).reshape(-1, 2)

        h = self.feature(cur_in)

        h = h.view(k, n_left, n_right, -1)

        logits = (h * self.out_w[:, None, None, :]).sum(dim=-1) + self.out_b[:, None, None]

        prob = torch.sigmoid(logits)

        pair_prob_log = torch.log(prob).sum(dim=0)

        return pair_prob_log

class NoAH_neural:
    
    def __init__(self, core_attr, fringe_attr, core_bank_ckpt, fringe_bank_ckpt, seed_prob, edge_num):
        self.nc = core_attr.shape[0]
        self.nf = fringe_attr.shape[0]
        self.n = self.nc + self.nf
        self.c2a = core_attr
        self.f2a = fringe_attr
        self.m = edge_num
        self.k = core_attr.shape[1]
        self.seed_prob = seed_prob
        self.e2n = [[] for _ in range(self.m)]

        self.core_bank = AttributeWiseMLP(core_bank_ckpt["k"], core_bank_ckpt["hidden_dim"])
        self.core_bank.load_state_dict(core_bank_ckpt["state_dict"])
        self.core_bank.eval()


        self.fringe_bank = AttributeWiseMLP(fringe_bank_ckpt["k"], fringe_bank_ckpt["hidden_dim"])
        self.fringe_bank.load_state_dict(fringe_bank_ckpt["state_dict"])
        self.fringe_bank.eval()

        self.construct_hypergraph()
    
    def construct_hypergraph(self):
        with torch.no_grad():
            core_core_probs_log = self.core_bank.pair_log_probs(self.c2a, self.c2a)
            core_core_probs = torch.exp(core_core_probs_log)
            core_core_probs.fill_diagonal_(0)

            for num_edge in tqdm.trange(self.m):
                core_seed = torch.multinomial(self.seed_prob, 1).item()
                core_group = [core_seed]

                cur_core_probs = core_core_probs[core_seed]
                cur_attached_cores = torch.bernoulli(cur_core_probs).to(torch.int)
                cur_attached_cores = torch.nonzero(cur_attached_cores).squeeze().tolist()

                if isinstance(cur_attached_cores, int):
                    cur_attached_cores = [cur_attached_cores]
                if len(cur_attached_cores) > 0:
                    core_group.extend(cur_attached_cores)

                if len(core_group) > 1:
                    edge_attr = torch.mean(self.c2a[core_group], dim=0)
                else:
                    edge_attr = self.c2a[core_group[0]]


                cur_fringe_probs_log = self.fringe_bank.pair_log_probs(edge_attr.unsqueeze(0), self.f2a)
                cur_fringe_probs = torch.exp(cur_fringe_probs_log).squeeze(0)
                cur_attached_fringes = torch.bernoulli(cur_fringe_probs).to(torch.int)
                cur_attached_fringes = torch.nonzero(cur_attached_fringes).squeeze().tolist()

                if isinstance(cur_attached_fringes, int):
                    cur_attached_fringes = [cur_attached_fringes]
                if len(cur_attached_fringes) > 0:
                    self.e2n[num_edge] = core_group + [node + self.nc for node in cur_attached_fringes]
                else:
                    self.e2n[num_edge] = core_group
