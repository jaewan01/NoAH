import tqdm
import torch
import numpy as np
from torch import nn, optim

class FullHypergraph(nn.Module):
    
    def __init__(self, Ic, If, Fc_raw, Ff_raw, theta_c_init, theta_f_init, seed_init, w_d, w_s, device):
        super(FullHypergraph, self).__init__()
        self.Ic = Ic
        self.If = If
        self.m = Ic.shape[0]
        self.nc, self.k = Fc_raw.shape
        self.nf = Ff_raw.shape[0]
        self.theta_c_log = nn.Parameter(theta_c_init)
        self.theta_f_log = nn.Parameter(theta_f_init)
        self.c2a_log = nn.Parameter(Fc_raw)
        self.f2a_log = nn.Parameter(Ff_raw)
        self.seed_prob = nn.Parameter(seed_init)
        self.w_d = w_d
        self.w_s = w_s
        self.device = device
             
    def forward(self, edge_split=None):

        if edge_split is None:
            cur_Ic = self.Ic
            cur_If = self.If
        else:
            cur_Ic = self.Ic[edge_split]
            cur_If = self.If[edge_split]

        Fc_exp_log = torch.log(torch.nn.Sigmoid()(self.c2a_log))
        Ff_exp_log = torch.log(torch.nn.Sigmoid()(self.f2a_log))

        theta_c_log_expanded = torch.log(torch.nn.Sigmoid()(self.theta_c_log)).permute(1,0).unsqueeze(1).unsqueeze(1)
        theta_f_log_expanded = torch.log(torch.nn.Sigmoid()(self.theta_f_log)).permute(1,0).unsqueeze(1).unsqueeze(1)

        seed_prob = torch.nn.Softmax(dim=0)(self.seed_prob)
        
        Ic_exp_log = torch.zeros_like(cur_Ic).to(self.device)

        loss = 0 

        Ac_exp_log = torch.zeros(4, self.nc, self.nc, self.k).to(self.device)
        Ac_exp_log[0] = torch.log1p(-torch.exp(Fc_exp_log)).unsqueeze(1) + torch.log1p(-torch.exp(Fc_exp_log)).unsqueeze(0) + theta_c_log_expanded[0]
        Ac_exp_log[1] = torch.log1p(-torch.exp(Fc_exp_log)).unsqueeze(1) + Fc_exp_log.unsqueeze(0) + theta_c_log_expanded[1]
        Ac_exp_log[2] = Fc_exp_log.unsqueeze(1) + torch.log1p(-torch.exp(Fc_exp_log)).unsqueeze(0) + theta_c_log_expanded[1]
        Ac_exp_log[3] = Fc_exp_log.unsqueeze(1) + Fc_exp_log.unsqueeze(0) + theta_c_log_expanded[2]
        Ac_exp_log = torch.logsumexp(Ac_exp_log, dim=0)
        Ac_exp_log = torch.sum(Ac_exp_log, dim=2)

        core_group_attr = torch.zeros(self.m, self.k).to(self.device)

        # Iterate over each hyperedge
        for edge in range(cur_Ic.shape[0]):
            # Find current core group
            cur_core_group_mask = cur_Ic[edge, :] == 1
            cur_core_group_indices = cur_core_group_mask.nonzero(as_tuple=True)[0]
            cur_other_core_mask = cur_Ic[edge, :] == 0

            # Calculate log likelihood loss
            cur_seed_probs = seed_prob[cur_core_group_mask]
            cur_seed_probs_log = torch.log(cur_seed_probs / (torch.sum(cur_seed_probs)))
            cur_core_prob_log = Ac_exp_log[cur_core_group_indices]
            self_diag_mask = torch.zeros_like(cur_core_prob_log, dtype=torch.bool)
            self_diag_mask[torch.arange(cur_core_group_mask.sum()), cur_core_group_indices] = True
            cur_core_prob_log = cur_core_prob_log.masked_fill(self_diag_mask, 0)
            cur_core_prob_log = cur_core_prob_log + cur_seed_probs_log.unsqueeze(-1)
            cur_core_prob_log = torch.logsumexp(cur_core_prob_log, dim=0)

            # Loss for cores in the core group
            loss = loss - torch.sum(cur_core_prob_log[cur_core_group_mask])

            # Loss for cores not in the core group
            cur_core_inv_log_others = torch.log1p(-torch.exp(cur_core_prob_log[cur_other_core_mask]))
            loss = loss - torch.sum(cur_core_inv_log_others)

            # expected Ic for degree & cardinality loss
            Ic_exp_log[edge] = cur_core_prob_log

            core_group_attr[edge] = torch.mean(torch.exp(Fc_exp_log[cur_core_group_mask]), dim=0)
        
        # Calculate expected degree & cardinality
        degree_exp = torch.exp(torch.logsumexp(Ic_exp_log, dim=0))  
        size_exp = torch.exp(torch.logsumexp(Ic_exp_log, dim=1))
        degree_exp, _ = torch.sort(degree_exp, descending = True)
        size_exp, _ = torch.sort(size_exp, descending = True)
        degree_answer, _ = torch.sort(torch.sum(cur_Ic, dim=0), descending = True)
        size_answer, _ = torch.sort(torch.sum(cur_Ic, dim=1), descending = True)

        # Loss for degree & cardinality
        criterion = torch.nn.MSELoss()
        degree_loss = criterion(degree_exp, degree_answer)
        size_loss = criterion(size_exp, size_answer)
        loss = loss + degree_loss * self.w_d + size_loss * self.w_s

        # Expand core-group attributes: shape (m, 1, k)
        core_group_attr_log = torch.log(core_group_attr).unsqueeze(1)

        If_exp_log = torch.zeros(4, self.m, self.nf, self.k).to(self.device)
        If_exp_log[0] = torch.log1p(-torch.exp(Ff_exp_log)).unsqueeze(0) + torch.log1p(-torch.exp(core_group_attr_log)) + theta_f_log_expanded[0]
        If_exp_log[1] = torch.log1p(-torch.exp(Ff_exp_log)).unsqueeze(0) + core_group_attr_log + theta_f_log_expanded[1]
        If_exp_log[2] = Ff_exp_log.unsqueeze(0) + torch.log1p(-torch.exp(core_group_attr_log)) + theta_f_log_expanded[1]
        If_exp_log[3] = Ff_exp_log.unsqueeze(0) + core_group_attr_log + theta_f_log_expanded[2]
        If_exp_log = torch.logsumexp(If_exp_log, dim=0)
        If_exp_log = torch.sum(If_exp_log, dim=2)

        # Loss for attached fringe nodes
        loss_attached = - torch.sum(If_exp_log[cur_If == 1])

        # For fringe nodes that are not attached
        loss_not_attached = - torch.sum(torch.log1p(-torch.exp(If_exp_log[cur_If == 0])))
        loss = loss + loss_attached + loss_not_attached

        # Calculate expected degree & cardinality
        degree_exp = torch.exp(torch.logsumexp(If_exp_log, dim=0))  
        size_exp = torch.exp(torch.logsumexp(If_exp_log, dim=1))      
        degree_exp, _ = torch.sort(degree_exp, descending = True)
        size_exp, _ = torch.sort(size_exp, descending = True)
        degree_answer, _ = torch.sort(torch.sum(cur_If, dim=0), descending = True)
        size_answer, _ = torch.sort(torch.sum(cur_If, dim=1), descending = True)

        # Loss for degree & cardinality
        degree_loss = criterion(degree_exp, degree_answer)
        size_loss = criterion(size_exp, size_answer)
        loss = loss + degree_loss * self.w_d + size_loss * self.w_s
        
        return loss

def NoAHfit_X(Ic, If, k, epoch, lr, w_d, w_s, n_batch, seed, device):
    
    """
        Estimate all parameters by NoAH without attributes.
    """

    np.random.seed(seed)

    nc = Ic.shape[1]
    nf = If.shape[1]
    
    # Initialize theta_c.
    theta_c_init = torch.ones(k, 3)
    init_val = (torch.sum(Ic) - nc) / (nc - 1) / nc
    init_val = init_val ** (1 / k)
    if init_val > 1:
        init_val = 0.5
    theta_c_init[:, :] = -np.log(1 / init_val - 1)

    # Initialize theta_f.
    theta_f_init = torch.ones(k, 3)
    init_val = torch.mean(If) ** (1 / k)
    theta_f_init[:, :] = -np.log(1 / init_val - 1)

    # Initialize seed_prob.
    seed_init = torch.zeros(nc)
    for e in range(Ic.shape[0]):
        cur_cores = Ic[e, :].nonzero()
        seed_init[cur_cores] += 1 / Ic.shape[0] / len(cur_cores)
    seed_init = torch.log(seed_init)

    # Initialize Fc.
    Fc_raw = torch.rand(nc, k)
    Fc_raw = torch.clamp(Fc_raw, min=0.1, max=0.9)
    Fc_raw = -torch.log(1 / Fc_raw - 1)

    # Initialize Ff.
    Ff_raw = torch.rand(nf, k)
    Ff_raw = torch.clamp(Ff_raw, min=0.1, max=0.9)
    Ff_raw = -torch.log(1 / Ff_raw - 1)

    
    # Fit all parameters.
    Ic = Ic.to(device)
    If = If.to(device)
    model = FullHypergraph(Ic, If, Fc_raw, Ff_raw, theta_c_init, theta_f_init, seed_init, w_d, w_s, device).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    model.train()
    min_loss = np.inf
    tol = 0
    pbar = tqdm.tqdm(range(epoch), desc="Core Group", unit="epoch")
    for _ in pbar:
        cur_theta_c = torch.nn.Sigmoid()(model.theta_c_log.clone())
        cur_theta_f = torch.nn.Sigmoid()(model.theta_f_log.clone())
        cur_seed_prob = torch.nn.Softmax(dim=0)(model.seed_prob.clone())
        cur_Fc = torch.nn.Sigmoid()(model.c2a_log.clone())
        cur_Ff = torch.nn.Sigmoid()(model.f2a_log.clone())

        if n_batch == 0:
            loss = model()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        else:
            loss = 0

            indices = np.arange(Ic.shape[0])
            np.random.shuffle(indices)
            edge_split = np.array_split(indices, n_batch)

            for b in range(n_batch):
                optimizer.zero_grad()
                cur_edge_split = edge_split[b]
                cur_loss = model(cur_edge_split)
                cur_loss.backward()
                optimizer.step()
                loss += cur_loss
                del cur_loss
        
        pbar.set_postfix(loss=f"{loss.item():.4f}")
        
        if min_loss > loss:
            tol = 0
            min_loss = loss
            best_theta_c = cur_theta_c
            best_theta_f = cur_theta_f
            best_seed_prob = cur_seed_prob
            best_Fc = cur_Fc
            best_Ff = cur_Ff
        else:
            tol += 1


        if tol == 30:
            print("early stop!")
            break
            
    best_theta_c = best_theta_c.detach().cpu()
    best_theta_f = best_theta_f.detach().cpu()
    best_seed_prob = best_seed_prob.detach().cpu() 
    best_Fc = best_Fc.detach().cpu()
    best_Ff = best_Ff.detach().cpu()

    return best_theta_c, best_theta_f, best_seed_prob, best_Fc, best_Ff
