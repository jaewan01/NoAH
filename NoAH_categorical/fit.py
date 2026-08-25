import os
import tqdm
import torch
import numpy as np
from torch import nn, optim

class CoreGroupConstructionCat(nn.Module):
    
    def __init__(self, Ic, Fc, cat_cnts, theta_init, seed_init, w_d, w_s, device):
        super(CoreGroupConstructionCat, self).__init__()
        self.Ic = Ic
        self.m = Ic.shape[0]
        self.c2a = Fc
        self.nc, self.k = Fc.shape
        self.cat_cnts = cat_cnts
        self.theta_log = nn.ParameterList(theta_init)
        self.seed_prob = nn.Parameter(seed_init)
        self.w_d = w_d
        self.w_s = w_s
        self.device = device
             
    def forward(self, edge_split=None):

        if edge_split is None:
            cur_Ic = self.Ic
        else:
            cur_Ic = self.Ic[edge_split]

        theta_c_log_expanded = []
        for l in range(self.k):
            theta_c_log = torch.log(torch.nn.Sigmoid()(self.theta_log[l]))
            theta_c_log_expanded_l = torch.zeros(self.cat_cnts[l], self.cat_cnts[l]).to(self.device)
            for i in range(self.cat_cnts[l]):
                for j in range(self.cat_cnts[l]):
                    if i >= j:
                        theta_c_log_expanded_l[i, j] = theta_c_log[i * (i + 1) // 2 + j]
                    else:
                        theta_c_log_expanded_l[i, j] = theta_c_log[j * (j + 1) // 2 + i]
            theta_c_log_expanded.append(theta_c_log_expanded_l)
            
        seed_prob = torch.nn.Softmax(dim=0)(self.seed_prob)
        
        Ic_exp_log = torch.zeros_like(cur_Ic).to(self.device)

        loss = 0 

        # Iterate over each hyperedge
        for edge in range(cur_Ic.shape[0]):
            # Find current core group
            cur_core_group_mask = cur_Ic[edge, :] == 1
            cur_core_group_indices = cur_core_group_mask.nonzero(as_tuple=True)[0]
            cur_other_core_mask = cur_Ic[edge, :] == 0

            # Calculate log likelihood loss
            cur_seed_probs = seed_prob[cur_core_group_mask]
            cur_seed_probs_log = torch.log(cur_seed_probs / (torch.sum(cur_seed_probs)))

            cur_core_prob_log = torch.zeros(cur_core_group_mask.sum(), self.nc).to(self.device)

            for l in range(self.k):
                theta_c_log_expanded_l = theta_c_log_expanded[l]
                cur_core_prob_log_l = theta_c_log_expanded_l[self.c2a[cur_core_group_indices, l]]
                cur_core_prob_log_l = cur_core_prob_log_l[ :, self.c2a[:, l]]  # shape: (|core_group|, nc)

                cur_core_prob_log = cur_core_prob_log + cur_core_prob_log_l

            cur_core_prob_log[torch.arange(cur_core_group_mask.sum()), cur_core_group_indices] = 0
            cur_core_prob_log = cur_core_prob_log + cur_seed_probs_log.unsqueeze(-1)
            cur_core_prob_log = torch.logsumexp(cur_core_prob_log, dim=0)

            # Loss for cores in the core group
            loss = loss - torch.sum(cur_core_prob_log[cur_core_group_mask])

            # Loss for cores not in the core group
            cur_core_inv_log_others = torch.log1p(-torch.exp(cur_core_prob_log[cur_other_core_mask]))
            loss = loss - torch.sum(cur_core_inv_log_others)

            # expected Ic for degree & cardinality loss
            Ic_exp_log[edge] = cur_core_prob_log
        
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
        
        return loss

class FringeAttachmentCat(nn.Module):
    
    def __init__(self, Ic, If, Fc, Ff, cat_cnts, theta_init, w_d, w_s, device):
        super(FringeAttachmentCat, self).__init__()
        self.Ic = Ic
        self.If = If
        self.m, self.nf = If.shape
        self.c2a = Fc
        self.f2a = Ff
        self.cat_cnts = cat_cnts
        self.k = Ff.shape[1]
        self.theta_log = nn.ParameterList(theta_init)
        self.w_d = w_d
        self.w_s = w_s
        self.device = device

    def forward(self, edge_split=None):
        if edge_split is None:
            cur_If = self.If
        else:
            cur_If = self.If[edge_split]
        
        cur_m = cur_If.shape[0]

        theta_f_expanded = []
        for l in range(self.k):
            theta_f = torch.nn.Sigmoid()(self.theta_log[l])
            theta_f_expanded_l = torch.zeros(self.cat_cnts[l], self.cat_cnts[l]).to(self.device)
            for i in range(self.cat_cnts[l]):
                for j in range(self.cat_cnts[l]):
                    if i >= j:
                        theta_f_expanded_l[i, j] = theta_f[i * (i + 1) // 2 + j]
                    else:
                        theta_f_expanded_l[i, j] = theta_f[j * (j + 1) // 2 + i]
            theta_f_expanded.append(theta_f_expanded_l)
        
        cur_fringe_prob_log = torch.zeros(cur_m, self.nf).to(self.device)

        for l in range(self.k):
            theta_f_expanded_l = theta_f_expanded[l]
            cur_core_fringe_probs_l = theta_f_expanded_l[self.c2a[:, l]][ :, self.f2a[:, l]]  # shape: (nc, nf)
            cur_fringe_prob_l = self.Ic @ cur_core_fringe_probs_l 
            cur_fringe_prob_l = (1 / torch.sum(self.Ic, dim=1, keepdim=True)) * cur_fringe_prob_l 
            cur_fringe_prob_log = cur_fringe_prob_log + torch.log(cur_fringe_prob_l)
        
        # Loss for attached fringe nodes
        loss_attached = - torch.sum(cur_fringe_prob_log[cur_If == 1])

        # For fringe nodes that are not attached
        loss_not_attached = - torch.sum(torch.log1p(-torch.exp(cur_fringe_prob_log[cur_If == 0])))
        loss = loss_attached + loss_not_attached

        # Calculate expected degree & cardinality
        degree_exp = torch.exp(torch.logsumexp(cur_fringe_prob_log, dim=0))  
        size_exp = torch.exp(torch.logsumexp(cur_fringe_prob_log, dim=1))      
        degree_exp, _ = torch.sort(degree_exp, descending = True)
        size_exp, _ = torch.sort(size_exp, descending = True)
        degree_answer, _ = torch.sort(torch.sum(cur_If, dim=0), descending = True)
        size_answer, _ = torch.sort(torch.sum(cur_If, dim=1), descending = True)

        # Loss for degree & cardinality
        criterion = torch.nn.MSELoss()
        degree_loss = criterion(degree_exp, degree_answer)
        size_loss = criterion(size_exp, size_answer)
        loss = loss + degree_loss * self.w_d + size_loss * self.w_s
        
        return loss


def NoAHfit_cat_core(Ic, Fc, cat_cnts, epoch, lr, w_d, w_s, n_batch_c, seed, device):
    
    """
        Estimate seed_prob & theta_c by core group construction.
    """

    np.random.seed(seed)

    nc = Ic.shape[1]
    k = Fc.shape[1]
    
    # Initialize theta_c.
    theta_init_list = []
    init_val = (torch.sum(Ic) - nc) / (nc - 1) / nc
    init_val = init_val ** (1 / k)
    if init_val > 1:
        init_val = 0.5

    for i in range(k):
        theta_init_list.append((-np.log(1 / init_val - 1)) * torch.ones(cat_cnts[i] * (cat_cnts[i] + 1) // 2))
    
    # Initialize seed_prob.
    seed_init = torch.zeros(nc)
    for e in range(Ic.shape[0]):
        cur_cores = Ic[e, :].nonzero()
        seed_init[cur_cores] += 1 / Ic.shape[0] / len(cur_cores)
    seed_init = torch.log(seed_init)
    
    # Fit theta_c & seed_prob.
    Ic = Ic.to(device)
    Fc = Fc.to(torch.int32).to(device)
    model = CoreGroupConstructionCat(Ic, Fc, cat_cnts, theta_init_list, seed_init, w_d, w_s, device).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    model.train()
    min_loss = np.inf
    tol = 0
    pbar = tqdm.tqdm(range(epoch), desc="Core Group", unit="epoch")
    for _ in pbar:
        cur_theta_c = []
        for l in range(k):
            cur_theta_c_flat = torch.nn.Sigmoid()(model.theta_log[l].clone()).detach().cpu()
            cur_theta_c_l = torch.zeros(cat_cnts[l], cat_cnts[l])
            for i in range(cat_cnts[l]):
                for j in range(cat_cnts[l]):
                    if i >= j:
                        cur_theta_c_l[i, j] = cur_theta_c_flat[i * (i + 1) // 2 + j]
                    else:
                        cur_theta_c_l[i, j] = cur_theta_c_flat[j * (j + 1) // 2 + i]
            cur_theta_c.append(cur_theta_c_l)
        cur_theta_c = torch.nn.ParameterList(cur_theta_c)
        cur_seed_prob = torch.nn.Softmax(dim=0)(model.seed_prob.clone())

        if n_batch_c == 0:
            loss = model()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        else:
            loss = 0

            indices = np.arange(Ic.shape[0])
            np.random.shuffle(indices)
            edge_split = np.array_split(indices, n_batch_c)

            for b in range(n_batch_c):
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
            best_seed_prob = cur_seed_prob
        else:
            tol += 1
        
        if tol == 30:
            print("early stop!")
            break
            
    best_seed_prob = best_seed_prob.detach().cpu() 

    return best_theta_c, best_seed_prob


def NoAHfit_cat_fringe(Ic, If, Fc, Ff, cat_cnts, epoch, lr, w_d, w_s, n_batch_f, seed, device):
        
    """
        Estimate theta_f by fringe attachment.
    """

    np.random.seed(seed)

    k = Fc.shape[1]
    
    # Initialize theta_f.
    theta_init_list = []
    init_val = torch.mean(If) ** (1 / k)

    for i in range(k):
        theta_init_list.append((-np.log(1 / init_val - 1)) * torch.ones(cat_cnts[i] * (cat_cnts[i] + 1) // 2))

    # Fit theta_f.
    Ic = Ic.to(device)
    If = If.to(device)
    Fc = Fc.to(torch.int32).to(device)
    Ff = Ff.to(torch.int32).to(device)
    model = FringeAttachmentCat(Ic, If, Fc, Ff, cat_cnts, theta_init_list, w_d, w_s, device).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    model.train()
    min_loss = np.inf
    tol = 0
    pbar = tqdm.tqdm(range(epoch), desc="Fringe Attachment", unit="epoch")
    for _ in pbar:
        cur_theta_f = []
        for l in range(k):
            cur_theta_f_flat = torch.nn.Sigmoid()(model.theta_log[l].clone()).detach().cpu()
            cur_theta_f_l = torch.zeros(cat_cnts[l], cat_cnts[l])
            for i in range(cat_cnts[l]):
                for j in range(cat_cnts[l]):
                    if i >= j:
                        cur_theta_f_l[i, j] = cur_theta_f_flat[i * (i + 1) // 2 + j]
                    else:
                        cur_theta_f_l[i, j] = cur_theta_f_flat[j * (j + 1) // 2 + i]
            cur_theta_f.append(cur_theta_f_l)
        cur_theta_f = torch.nn.ParameterList(cur_theta_f)

        if n_batch_f == 0:
            loss = model()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        else:
            loss = 0

            indices = np.arange(If.shape[0])
            np.random.shuffle(indices)
            edge_split = np.array_split(indices, n_batch_f)

            for b in range(n_batch_f):
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
            best_theta_f = cur_theta_f
        else:
            tol += 1

        if tol == 30:
            print("early stop!")
            break
            
    return best_theta_f