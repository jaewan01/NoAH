import os
import tqdm
import torch
import numpy as np
from torch import nn, optim

class CoreGroupConstruction(nn.Module):
    
    def __init__(self, Ic, Fc, theta_init, seed_init, w_d, w_s, device):
        super(CoreGroupConstruction, self).__init__()
        self.Ic = Ic
        self.m = Ic.shape[0]
        self.c2a = Fc
        self.nc, self.k = Fc.shape
        self.theta_log = nn.Parameter(theta_init)
        self.seed_prob = nn.Parameter(seed_init)
        self.w_d = w_d
        self.w_s = w_s
        self.device = device
             
    def forward(self, edge_split=None):

        if edge_split is None:
            cur_Ic = self.Ic
        else:
            cur_Ic = self.Ic[edge_split]

        theta_c_log_expanded = torch.log(torch.nn.Sigmoid()(self.theta_log)).permute(1,0).unsqueeze(1).unsqueeze(1)

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
            cur_attr_sum = self.c2a[cur_core_group_indices].unsqueeze(1) + self.c2a.unsqueeze(0)  
            cur_core_prob_log = theta_c_log_expanded[0] * (cur_attr_sum == 0) + theta_c_log_expanded[1] * (cur_attr_sum == 1) + theta_c_log_expanded[2] * (cur_attr_sum == 2)
            cur_core_prob_log = torch.sum(cur_core_prob_log, dim=2) 
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
    

class FringeAttachment(nn.Module):
    
    def __init__(self, If, Ff, Fcg, theta_init, w_d, w_s, device):
        super(FringeAttachment, self).__init__()
        self.If = If
        self.m, self.nf = If.shape
        self.f2a = Ff
        self.k = Ff.shape[1]
        self.cg2a = Fcg
        self.theta_log = nn.Parameter(theta_init)
        self.w_d = w_d
        self.w_s = w_s
        self.device = device

    def forward(self, edge_split=None):
        if edge_split is None:
            cur_If = self.If
            cur_cg2a = self.cg2a
        else:
            cur_If = self.If[edge_split]
            cur_cg2a = self.cg2a[edge_split]

        theta_f = torch.nn.Sigmoid()(self.theta_log)

        # Expand fringe attributes: shape (1, nf, k)
        fringe_attr = self.f2a.unsqueeze(0)  
        # Expand core-group attributes: shape (m, 1, k)
        core_group_attr = cur_cg2a.unsqueeze(1) 
        
        # Create binary masks for fringe attributes:
        fringe_mask0 = (fringe_attr == 0).float()  
        fringe_mask1 = (fringe_attr == 1).float() 
        
        # Compute terms for each attribute dimension using theta_f:
        theta0 = theta_f[:, 0].view(1, 1, self.k)  
        theta1 = theta_f[:, 1].view(1, 1, self.k) 
        theta2 = theta_f[:, 2].view(1, 1, self.k) 

        term0 = theta0 * fringe_mask0 * (1 - core_group_attr) 
        term1 = theta1 * fringe_mask1 * (1 - core_group_attr)    
        term2 = theta1 * fringe_mask0 * core_group_attr    
        term3 = theta2 * fringe_mask1 * core_group_attr         

        # Sum the terms to get the fringe probability per attribute dimension:
        fringe_prob_per_attr = term0 + term1 + term2 + term3  

        # Take log and then sum over attribute dimensions (k) to get per-hyperedge, per-fringe log probability.
        cur_fringe_prob_log = torch.log(fringe_prob_per_attr)  
        cur_fringe_prob_log = torch.sum(cur_fringe_prob_log, dim=2)  

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


def NoAHfit_core(Ic, Fc, epoch, lr, w_d, w_s, n_batch_c, seed, device):
    
    """
        Estimate seed_prob & theta_c by core group construction.
    """

    np.random.seed(seed)

    nc = Ic.shape[1]
    k = Fc.shape[1]
    
    # Initialize theta_c.
    theta_init = torch.ones(k, 3)
    init_val = (torch.sum(Ic) - nc) / (nc - 1) / nc
    init_val = init_val ** (1 / k)
    if init_val > 1:
        init_val = 0.5
    theta_init[:, :] = -np.log(1 / init_val - 1)

    # Initialize seed_prob.
    seed_init = torch.zeros(nc)
    for e in range(Ic.shape[0]):
        cur_cores = Ic[e, :].nonzero()
        seed_init[cur_cores] += 1 / Ic.shape[0] / len(cur_cores)
    seed_init = torch.log(seed_init)
    
    # Fit theta_c & seed_prob.
    Ic = Ic.to(device)
    Fc = torch.FloatTensor(Fc).to(device)
    model = CoreGroupConstruction(Ic, Fc, theta_init, seed_init, w_d, w_s, device).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    model.train()
    min_loss = np.inf
    tol = 0
    pbar = tqdm.tqdm(range(epoch), desc="Core Group", unit="epoch")
    for _ in pbar:
        cur_theta_c = torch.nn.Sigmoid()(model.theta_log.clone())
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
            
    best_theta_c = best_theta_c.detach().cpu()
    best_seed_prob = best_seed_prob.detach().cpu() 

    return best_theta_c, best_seed_prob


def NoAHfit_fringe(If, Ff, Fcg, epoch, lr, w_d, w_s, n_batch_f, seed, device):
        
    """
        Estimate theta_f by fringe attachment.
    """

    np.random.seed(seed)

    k = Fcg.shape[1]
    
    # Initialize theta_f.
    theta_init = torch.ones(k, 3)
    init_val = torch.mean(If) ** (1 / k)
    theta_init[:, :] = -np.log(1 / init_val - 1)

    # Fit theta_f.
    If = If.to(device)
    Ff = torch.FloatTensor(Ff).to(device)
    Fcg = Fcg.to(device)
    model = FringeAttachment(If, Ff, Fcg, theta_init, w_d, w_s, device).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    model.train()
    min_loss = np.inf
    tol = 0
    pbar = tqdm.tqdm(range(epoch), desc="Fringe Attachment", unit="epoch")
    for _ in pbar:
        cur_theta_f = torch.nn.Sigmoid()(model.theta_log.clone())

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
            
    best_theta_f = best_theta_f.detach().cpu()

    return best_theta_f

