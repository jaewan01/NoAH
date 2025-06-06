import torch
import tqdm


class NoAH:
    
    """
        NoAH is a model which generates a hypergraph soley based on node attributes.
        
        step 1. Construct core group using core node attributes, seed core probability, and core affinity matrix.
        step 2. Mix the node attributes within each core group to generate the attribute of core group.
        step 3. Construct hyperedge by attaching fringe nodes to core groups using fringe node attributes, core group attributes, and fringe affinity matrix.  
    """
    
    def __init__(self, core_attr, fringe_attr, core_affinity_matrix, fringe_affinity_matrix, seed_prob, edge_num, mode):
        self.nc = core_attr.shape[0]
        self.nf = fringe_attr.shape[0]
        self.n = self.nc + self.nf
        self.c2a = core_attr
        self.f2a = fringe_attr
        self.m = edge_num
        self.k = core_attr.shape[1]
        self.Tc = core_affinity_matrix
        self.Tf = fringe_affinity_matrix
        self.seed_prob = seed_prob
        self.e2n = [[] for _ in range(self.m)]
        self.mode = mode
        self.construct_hypergraph()
   
    def construct_hypergraph(self):
        core_core_probs = torch.ones((self.nc, self.nc))
        for i in range(self.k):
            cur_attr_sum = (self.c2a[:, i].reshape(-1, 1) + self.c2a[:, i].reshape(1, -1)).to(torch.int)
            core_core_probs *= self.Tc[i][cur_attr_sum]

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

            if self.mode == "NoAH":
                if len(core_group) > 1:
                    e2p = torch.mean(self.c2a[core_group], dim=0)
                else:
                    e2p = self.c2a[core_group[0]]
                edge_attr = torch.bernoulli(e2p.expand(self.nf, -1)).to(torch.int)
            
            attr_sum = (self.f2a + edge_attr).to(torch.int)

            cur_fringe_probs = torch.ones(self.nf)
            for i in range(self.k):
                cur_fringe_probs *= self.Tf[i][attr_sum[:, i]]

            cur_attached_fringes = torch.bernoulli(cur_fringe_probs).to(torch.int)
            cur_attached_fringes = torch.nonzero(cur_attached_fringes).squeeze().tolist()

            if isinstance(cur_attached_fringes, int):
                cur_attached_fringes = [cur_attached_fringes]
            if len(cur_attached_fringes) > 0:
                self.e2n[num_edge] = core_group + [node + self.nc for node in cur_attached_fringes]
            else:
                self.e2n[num_edge] = core_group


class Bipartite:
    
    """
        Bipartite is single step version of HyperCF, so Bipartite does not utilize core-fringe structure and attribute mixing.
    """
    
    def __init__(self, attr, affinity_matrix, seed_prob, edge_num):
        self.n = attr.shape[0]
        self.n2a = attr
        self.m = edge_num
        self.k = attr.shape[1]
        self.T = affinity_matrix
        self.seed_prob = seed_prob
        self.e2n = [[] for _ in range(self.m)]
        self.construct_hypergraph()
   
    def construct_hypergraph(self):
        node_node_probs = torch.ones((self.n, self.n))
        for i in range(self.k):
            cur_attr_sum = (self.n2a[:, i].reshape(-1, 1) + self.n2a[:, i].reshape(1, -1)).to(torch.int)
            node_node_probs *= self.T[i][cur_attr_sum]

        node_node_probs.fill_diagonal_(0)

        for num_edge in tqdm.trange(self.m):
            seed = torch.multinomial(self.seed_prob, 1).item()
            hyperedge = [seed]

            cur_probs = node_node_probs[seed]
            cur_attached_nodes = torch.bernoulli(cur_probs).to(torch.int)
            cur_attached_nodes = torch.nonzero(cur_attached_nodes).squeeze().tolist()

            if isinstance(cur_attached_nodes, int): 
                cur_attached_nodes = [cur_attached_nodes]
            if len(cur_attached_nodes) > 0:
                hyperedge.extend(cur_attached_nodes)

            self.e2n[num_edge] = hyperedge
