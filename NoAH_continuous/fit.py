import os
import copy
import tqdm
import torch
import numpy as np
from torch import nn, optim

import pdb

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
            cur_core_prob_log[torch.arange(cur_core_group_mask.sum(), device=self.device), cur_core_group_indices] = 0
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

class CoreGroupConstructionContinuous(nn.Module):
    
    def __init__(self, Ic, Fc, theta_init, seed_init, w_d, w_s, device):
        super(CoreGroupConstructionContinuous, self).__init__()
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

        theta_c_expanded = torch.nn.Sigmoid()(self.theta_log).permute(1,0).unsqueeze(1).unsqueeze(1)

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
            cur_core_prob_0_0 = (1 - self.c2a[cur_core_group_indices].unsqueeze(1)) * (1 - self.c2a.unsqueeze(0)) * theta_c_expanded[0]
            cur_core_prob_0_1 = (1 - self.c2a[cur_core_group_indices].unsqueeze(1)) * self.c2a.unsqueeze(0) * theta_c_expanded[1]
            cur_core_prob_1_0 = self.c2a[cur_core_group_indices].unsqueeze(1) * (1 - self.c2a.unsqueeze(0)) * theta_c_expanded[1]
            cur_core_prob_1_1 = self.c2a[cur_core_group_indices].unsqueeze(1) * self.c2a.unsqueeze(0) * theta_c_expanded[2]
            cur_core_prob_log = torch.log(cur_core_prob_0_0 + cur_core_prob_0_1 + cur_core_prob_1_0 + cur_core_prob_1_1)
            cur_core_prob_log = torch.sum(cur_core_prob_log, dim=2) 
            cur_core_prob_log[torch.arange(cur_core_group_mask.sum(), device=self.device), cur_core_group_indices] = 0
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

class FringeAttachment_no_mix(nn.Module):
    
    def __init__(self, Ic, If, Fc, Ff, seed_prob, theta_init, w_d, w_s, device):
        super(FringeAttachment_no_mix, self).__init__()
        self.Ic = Ic
        self.If = If
        self.m, self.nc = Ic.shape
        self.nf = If.shape[1]
        self.c2a = Fc
        self.f2a = Ff
        self.k = Ff.shape[1]
        self.seed_prob = seed_prob
        self.theta_log = nn.Parameter(theta_init)
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

        theta_f_log_expanded = torch.log(torch.nn.Sigmoid()(self.theta_log)).permute(1,0).unsqueeze(1).unsqueeze(1)

        If_exp_log = torch.zeros_like(cur_If).to(self.device)

        # Iterate over each hyperedge
        for edge in range(cur_If.shape[0]):
            # Find current core group
            cur_core_group_mask = cur_Ic[edge, :] == 1
            cur_seed_probs = self.seed_prob[cur_core_group_mask]
            cur_seed_probs_log = torch.log(cur_seed_probs / (torch.sum(cur_seed_probs)))

            # Calculate expected If
            cur_attr_sum = self.c2a[cur_core_group_mask].unsqueeze(1) + self.f2a.unsqueeze(0) 
            cur_core_fringe_probs_log = theta_f_log_expanded[0] * (cur_attr_sum == 0) + theta_f_log_expanded[1] * (cur_attr_sum == 1) + theta_f_log_expanded[2] * (cur_attr_sum == 2)
            cur_core_fringe_probs_log = torch.sum(cur_core_fringe_probs_log, dim=2)  
            cur_core_fringe_probs_log = cur_core_fringe_probs_log + cur_seed_probs_log.unsqueeze(1)  
            cur_core_fringe_probs_log = torch.logsumexp(cur_core_fringe_probs_log, dim=0)  

            If_exp_log[edge] = cur_core_fringe_probs_log

        # Loss for attached fringe nodes
        loss_attached = - torch.sum(If_exp_log[cur_If == 1])

        # For fringe nodes that are not attached
        loss_not_attached = - torch.sum(torch.log1p(-torch.exp(If_exp_log[cur_If == 0])))
        loss = loss_attached + loss_not_attached

        # Calculate expected degree & cardinality
        degree_exp = torch.exp(torch.logsumexp(If_exp_log, dim=0))  
        size_exp = torch.exp(torch.logsumexp(If_exp_log, dim=1))      
        degree_exp, _ = torch.sort(degree_exp, descending = True)
        size_exp, _ = torch.sort(size_exp, descending = True)
        degree_answer, _ = torch.sort(torch.sum(cur_If, dim=0), descending = True)
        size_answer, _ = torch.sort(torch.sum(cur_If, dim=1), descending = True)
        criterion = torch.nn.MSELoss()

        # Loss for degree & cardinality
        degree_loss = criterion(degree_exp, degree_answer)
        size_loss = criterion(size_exp, size_answer)
        loss = loss + degree_loss * self.w_d + size_loss * self.w_s
        
        return loss

class FringeAttachmentContinuous(nn.Module):
    
    def __init__(self, If, Ff, Fcg, theta_init, w_d, w_s, device):
        super(FringeAttachmentContinuous, self).__init__()
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
        
        # Compute terms for each attribute dimension using theta_f:
        theta0 = theta_f[:, 0].view(1, 1, self.k)  
        theta1 = theta_f[:, 1].view(1, 1, self.k) 
        theta2 = theta_f[:, 2].view(1, 1, self.k) 

        term0 = theta0 * (1 - fringe_attr) * (1 - core_group_attr) 
        term1 = theta1 * fringe_attr * (1 - core_group_attr)    
        term2 = theta1 * (1 - fringe_attr) * core_group_attr    
        term3 = theta2 * fringe_attr * core_group_attr         

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

class FullHypergraph(nn.Module):
    
    def __init__(self, Ic, If, Fc, Ff, theta_c_init, theta_f_init, seed_init, w_d, w_s, device):
        super(FullHypergraph, self).__init__()
        self.Ic = Ic
        self.If = If
        self.m = Ic.shape[0]
        self.nc, self.k = Fc.shape
        self.nf = Ff.shape[0]
        self.theta_c_log = nn.Parameter(theta_c_init)
        self.theta_f_log = nn.Parameter(theta_f_init)
        self.c2a = Fc
        self.f2a = Ff
        self.seed_prob = nn.Parameter(seed_init)
        self.w_d = w_d
        self.w_s = w_s
        self.alphas = nn.Parameter(torch.ones(self.k))
        self.betas = nn.Parameter(torch.zeros(self.k))
        self.device = device
             
    def forward(self, edge_split=None):

        if edge_split is None:
            cur_Ic = self.Ic
            cur_If = self.If
        else:
            cur_Ic = self.Ic[edge_split]
            cur_If = self.If[edge_split]

        theta_c_expanded = torch.nn.Sigmoid()(self.theta_c_log).permute(1,0).unsqueeze(1).unsqueeze(1)
        theta_f = torch.nn.Sigmoid()(self.theta_f_log)


        seed_prob = torch.nn.Softmax(dim=0)(self.seed_prob)
        
        Ic_exp_log = torch.zeros_like(cur_Ic).to(self.device)

        c2a = torch.nn.Sigmoid()(self.alphas.unsqueeze(0) * self.c2a + self.betas.unsqueeze(0))
        f2a = torch.nn.Sigmoid()(self.alphas.unsqueeze(0) * self.f2a + self.betas.unsqueeze(0)).unsqueeze(0)

        loss = 0 

        core_group_attr = torch.zeros(cur_Ic.shape[0], self.k).to(self.device)

        # Iterate over each hyperedge
        for edge in range(cur_Ic.shape[0]):
            # Find current core group
            cur_core_group_mask = cur_Ic[edge, :] == 1
            cur_core_group_indices = cur_core_group_mask.nonzero(as_tuple=True)[0]
            cur_other_core_mask = cur_Ic[edge, :] == 0

            # Calculate log likelihood loss
            cur_seed_probs = seed_prob[cur_core_group_mask]
            cur_seed_probs_log = torch.log(cur_seed_probs / (torch.sum(cur_seed_probs)))
            cur_core_prob_0_0 = (1 - c2a[cur_core_group_indices].unsqueeze(1)) * (1 - c2a.unsqueeze(0)) * theta_c_expanded[0]
            cur_core_prob_0_1 = (1 - c2a[cur_core_group_indices].unsqueeze(1)) * c2a.unsqueeze(0) * theta_c_expanded[1]
            cur_core_prob_1_0 = c2a[cur_core_group_indices].unsqueeze(1) * (1 - c2a.unsqueeze(0)) * theta_c_expanded[1]
            cur_core_prob_1_1 = c2a[cur_core_group_indices].unsqueeze(1) * c2a.unsqueeze(0) * theta_c_expanded[2]
            cur_core_prob_log = torch.log(cur_core_prob_0_0 + cur_core_prob_0_1 + cur_core_prob_1_0 + cur_core_prob_1_1)
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

            core_group_attr[edge] = torch.mean(c2a[cur_core_group_mask], dim=0)
        
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
        core_group_attr = core_group_attr.unsqueeze(1)

        # Compute terms for each attribute dimension using theta_f:
        theta0 = theta_f[:, 0].view(1, 1, self.k)  
        theta1 = theta_f[:, 1].view(1, 1, self.k) 
        theta2 = theta_f[:, 2].view(1, 1, self.k) 

        If_exp = torch.zeros(4, cur_If.shape[0], self.nf, self.k).to(self.device)
        If_exp[0] = (1 - f2a) * (1 - core_group_attr) * theta0
        If_exp[1] = f2a * (1 - core_group_attr) * theta1
        If_exp[2] = (1 - f2a) * core_group_attr * theta1
        If_exp[3] = f2a * core_group_attr * theta2
        If_exp = torch.sum(If_exp, dim=0)
        If_exp_log = torch.log(If_exp)
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

def NoAHfit_fringe_no_mix(Ic, If, Fc, Ff, seed_prob, epoch, lr, w_d, w_s, n_batch_f, seed, device):
        
    """
        Estimate theta_f by fringe attachment.
    """

    np.random.seed(seed)

    k = Fc.shape[1]
    
    # Initialize theta_f.
    theta_init = torch.ones(k, 3)
    init_val = torch.mean(If) ** (1 / k)
    theta_init[:, :] = -np.log(1 / init_val - 1)

    # Fit theta_f.
    Ic = Ic.to(device)
    If = If.to(device)
    Fc = torch.FloatTensor(Fc).to(device)
    Ff = torch.FloatTensor(Ff).to(device)
    seed_prob = seed_prob.to(device)
    model = FringeAttachment_no_mix(Ic, If, Fc, Ff, seed_prob, theta_init, w_d, w_s, device).to(device)
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

def NoAHfit_continuous(Ic, If, Fc, Ff, epoch, lr, w_d, w_s, n_batch, seed, device):
    
    """
        Estimate all parameters by NoAH without attributes.
    """

    np.random.seed(seed)

    nc = Ic.shape[1]
    nf = If.shape[1]
    k = Fc.shape[1]
    
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
    
    # Fit theta_c & seed_prob.
    Ic = Ic.to(device)
    If = If.to(device)
    Fc = torch.FloatTensor(Fc).to(device)
    Ff = torch.FloatTensor(Ff).to(device)
    model = FullHypergraph(Ic, If, Fc, Ff, theta_c_init, theta_f_init, seed_init, w_d, w_s, device).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    model.train()
    min_loss = np.inf
    tol = 0
    pbar = tqdm.tqdm(range(epoch), desc="Core Group", unit="epoch")
    for _ in pbar:
        cur_theta_c = torch.nn.Sigmoid()(model.theta_c_log.clone())
        cur_theta_f = torch.nn.Sigmoid()(model.theta_f_log.clone())
        cur_seed_prob = torch.nn.Softmax(dim=0)(model.seed_prob.clone())
        cur_alphas = model.alphas.clone()
        cur_betas = model.betas.clone()

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
            best_alphas = cur_alphas
            best_betas = cur_betas
            best_seed_prob = cur_seed_prob
        else:
            tol += 1


        if tol == 30:
            print("early stop!")
            break
            
    best_theta_c = best_theta_c.detach().cpu()
    best_theta_f = best_theta_f.detach().cpu()
    best_alphas = best_alphas.detach().cpu()
    best_betas = best_betas.detach().cpu()
    best_seed_prob = best_seed_prob.detach().cpu() 

    return best_theta_c, best_theta_f, best_alphas, best_betas, best_seed_prob


def NoAHfit_continuous_core(Ic, Fc, epoch, lr, w_d, w_s, n_batch, seed, device):
    
    """
        Estimate all parameters by NoAH without attributes.
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
    model = CoreGroupConstructionContinuous(Ic, Fc, theta_init, seed_init, w_d, w_s, device).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    model.train()
    min_loss = np.inf
    tol = 0
    pbar = tqdm.tqdm(range(epoch), desc="Core Group", unit="epoch")
    for _ in pbar:
        cur_theta_c = torch.nn.Sigmoid()(model.theta_log.clone())
        cur_seed_prob = torch.nn.Softmax(dim=0)(model.seed_prob.clone())

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
            best_seed_prob = cur_seed_prob
        else:
            tol += 1


        if tol == 30:
            print("early stop!")
            break
            
    best_theta_c = best_theta_c.detach().cpu()
    best_seed_prob = best_seed_prob.detach().cpu() 

    return best_theta_c, best_seed_prob

def NoAHfit_continuous_fringe(If, Fc, Fcg, epoch, lr, w_d, w_s, n_batch, seed, device):
    
    """
        Estimate all parameters by NoAH without attributes.
    """

    np.random.seed(seed)

    k = Fcg.shape[1]

    # Initialize theta_f.
    theta_init = torch.ones(k, 3)
    init_val = torch.mean(If) ** (1 / k)
    theta_init[:, :] = -np.log(1 / init_val - 1)

    # Fit all parameters.
    If = If.to(device)
    Ff = torch.FloatTensor(Fc).to(device)
    Fcg = Fcg.to(device)
    model = FringeAttachmentContinuous(If, Ff, Fcg, theta_init, w_d, w_s, device).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    model.train()
    min_loss = np.inf
    tol = 0
    pbar = tqdm.tqdm(range(epoch), desc="Fringe Attachment", unit="epoch")
    for _ in pbar:
        cur_theta_f = torch.nn.Sigmoid()(model.theta_log.clone())

        if n_batch == 0:
            loss = model()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        else:
            loss = 0

            indices = np.arange(If.shape[0])
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
            best_theta_f = cur_theta_f
        else:
            tol += 1


        if tol == 30:
            print("early stop!")
            break
            
    best_theta_f = best_theta_f.detach().cpu()

    return best_theta_f


class FullAttributeMLP(nn.Module):
    
    def __init__(self, k, hidden_dim=16):
        super(FullAttributeMLP, self).__init__()
        self.k = k
        self.hidden_dim = hidden_dim
        self.mlp = nn.Sequential(
            nn.Linear(2 * k, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    def pair_log_probs(self, left_attr, right_attr, eps=1e-12):
        n_left = left_attr.shape[0]
        n_right = right_attr.shape[0]
        left = left_attr.unsqueeze(1).expand(-1, n_right, -1)
        right = right_attr.unsqueeze(0).expand(n_left, -1, -1)
        cur_in = torch.cat((left, right), dim=-1).reshape(-1, 2 * self.k)
        cur_prob = self.mlp(cur_in).reshape(n_left, n_right).clamp(min=eps, max=1 - eps)
        pair_prob_log = torch.log(cur_prob)
        return pair_prob_log

# class AttributeWiseMLP(nn.Module):
#     def __init__(self, k, hidden_dim=16):
#         super().__init__()
#         self.k = k
#         self.hidden_dim = hidden_dim

#         self.w1 = nn.Parameter(torch.ones(k, hidden_dim, 2))
#         self.b1 = nn.Parameter(torch.zeros(k, hidden_dim))
#         self.w2 = nn.Parameter(torch.ones(k, 1, hidden_dim))
#         self.b2 = nn.Parameter(torch.zeros(k, 1))

#     def forward(self, x):
#         h = torch.einsum("knd,khd->knh", x, self.w1) + self.b1[:, None, :]
#         h = torch.relu(h)
#         out = torch.einsum("knh,koh->kno", h, self.w2) + self.b2[:, None, :]
#         out = torch.sigmoid(out).squeeze(-1)
#         return out
    
#     def pair_log_probs(self, left_attr, right_attr):
#         n_left = left_attr.shape[0]
#         n_right = right_attr.shape[0]

#         left = left_attr.T[:, :, None]      
#         right = right_attr.T[:, None, :] 

#         s = left.expand(-1, -1, n_right)
#         t = right.expand(-1, n_left, -1)

#         cur_in = torch.stack((s, t), dim=-1).reshape(self.k, -1, 2)
#         cur_prob = self.forward(cur_in).reshape(self.k, n_left, n_right)

#         pair_prob_log = torch.log(cur_prob).sum(dim=0)   # [n_left, n_right]
#         return pair_prob_log

# class AttributeWiseMLP(nn.Module): 
#     def __init__(self, k, hidden_dim=16): 
#         super(AttributeWiseMLP, self).__init__() 
#         self.k = k
#         self.hidden_dim = hidden_dim
#         self.mlps = nn.ModuleList([ nn.Sequential( nn.Linear(2, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1), nn.Sigmoid(), ) for _ in range(k) ])
    
#     def pair_log_probs(self, left_attr, right_attr): 
#         n_left = left_attr.shape[0] 
#         n_right = right_attr.shape[0] 
#         pair_prob_log = torch.zeros(n_left, n_right, device=left_attr.device) 
#         for i in range(self.k): 
#             left = left_attr[:, i].unsqueeze(1).expand(-1, n_right) 
#             right = right_attr[:, i].unsqueeze(0).expand(n_left, -1) 
#             cur_in = torch.stack((left, right), dim=-1).reshape(-1, 2) 
#             cur_prob = self.mlps[i](cur_in).reshape(n_left, n_right)
#             pair_prob_log = pair_prob_log + torch.log(cur_prob) 
#         return pair_prob_log

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


def _build_neural_bank(k, hidden_dim):
    return AttributeWiseMLP(k, hidden_dim)



class CoreGroupConstructionNeural(nn.Module):
    
    def __init__(
        self,
        Ic,
        Fc,
        seed_init,
        w_d,
        w_s,
        device,
        hidden_dim=8
    ):
        super(CoreGroupConstructionNeural, self).__init__()
        self.Ic = Ic
        self.m = Ic.shape[0]
        self.nc, self.k = Fc.shape
        self.c2a = Fc
        self.seed_prob = nn.Parameter(seed_init)
        self.core_bank = _build_neural_bank(self.k, hidden_dim)
        self.w_d = w_d
        self.w_s = w_s
        self.device = device

    def forward(self, edge_split=None):
        if edge_split is None:
            cur_Ic = self.Ic
        else:
            cur_Ic = self.Ic[edge_split]

        seed_prob = torch.nn.Softmax(dim=0)(self.seed_prob)
        criterion = torch.nn.MSELoss()

        Ic_exp_log = torch.zeros_like(cur_Ic).to(self.device)

        loss = 0

        for edge in range(cur_Ic.shape[0]):
            cur_core_group_mask = cur_Ic[edge, :] == 1
            cur_core_group_indices = cur_core_group_mask.nonzero(as_tuple=True)[0]
            cur_other_core_mask = cur_Ic[edge, :] == 0


            cur_seed_probs = seed_prob[cur_core_group_mask]
            cur_seed_probs_log = torch.log(cur_seed_probs / (torch.sum(cur_seed_probs)))

            cur_attr = self.c2a[cur_core_group_indices]

            core_pair_prob_log = self.core_bank.pair_log_probs(cur_attr, self.c2a)
            core_pair_prob_log[torch.arange(cur_core_group_mask.sum()), cur_core_group_indices] = 0
            core_pair_prob_log = core_pair_prob_log + cur_seed_probs_log.unsqueeze(-1)
            core_pair_prob_log = torch.logsumexp(core_pair_prob_log, dim=0)

            Ic_exp_log[edge, :] = core_pair_prob_log
            loss = loss - torch.sum(core_pair_prob_log[cur_core_group_mask])
            cur_core_inv_log_others = torch.log1p(-torch.exp(core_pair_prob_log[cur_other_core_mask]))
            loss = loss - torch.sum(cur_core_inv_log_others)

        degree_exp = torch.exp(torch.logsumexp(Ic_exp_log, dim=0))
        size_exp = torch.exp(torch.logsumexp(Ic_exp_log, dim=1))
        degree_exp, _ = torch.sort(degree_exp, descending=True)
        size_exp, _ = torch.sort(size_exp, descending=True)
        degree_answer, _ = torch.sort(torch.sum(cur_Ic, dim=0), descending=True)
        size_answer, _ = torch.sort(torch.sum(cur_Ic, dim=1), descending=True)
        degree_loss = criterion(degree_exp, degree_answer)
        size_loss = criterion(size_exp, size_answer)
        loss = loss + degree_loss * self.w_d + size_loss * self.w_s

        return loss


class FringeAttachmentNeural(nn.Module):
    
    def __init__(self, If, Ff, Fcg, w_d, w_s, device, hidden_dim=8):
        super(FringeAttachmentNeural, self).__init__()
        self.If = If
        self.m = If.shape[0]
        self.nf, self.k = Ff.shape
        self.f2a = Ff
        self.Fcg = Fcg
        self.fringe_bank = _build_neural_bank(self.k, hidden_dim)
        self.w_d = w_d
        self.w_s = w_s
        self.device = device

    def forward(self, edge_split=None):
        if edge_split is None:
            cur_If = self.If
            cur_Fcg = self.Fcg
        else:
            cur_If = self.If[edge_split]
            cur_Fcg = self.Fcg[edge_split]

        criterion = torch.nn.MSELoss()
        If_exp_log = self.fringe_bank.pair_log_probs(cur_Fcg, self.f2a)

        loss_attached = -torch.sum(If_exp_log[cur_If == 1])
        loss_not_attached = -torch.sum(torch.log1p(-torch.exp(If_exp_log[cur_If == 0])))
        loss = loss_attached + loss_not_attached

        degree_exp = torch.exp(torch.logsumexp(If_exp_log, dim=0))
        size_exp = torch.exp(torch.logsumexp(If_exp_log, dim=1))
        degree_exp, _ = torch.sort(degree_exp, descending=True)
        size_exp, _ = torch.sort(size_exp, descending=True)
        degree_answer, _ = torch.sort(torch.sum(cur_If, dim=0), descending=True)
        size_answer, _ = torch.sort(torch.sum(cur_If, dim=1), descending=True)
        degree_loss = criterion(degree_exp, degree_answer)
        size_loss = criterion(size_exp, size_answer)
        loss = loss + degree_loss * self.w_d + size_loss * self.w_s

        return loss


def NoAHfit_continuous_neural_core(
    Ic,
    Fc,
    epoch,
    lr,
    w_d,
    w_s,
    n_batch,
    seed,
    device,
    hidden_dim=8
):
    np.random.seed(seed)
    nc = Ic.shape[1]

    seed_init = torch.zeros(nc)
    for e in range(Ic.shape[0]):
        cur_cores = Ic[e, :].nonzero()
        seed_init[cur_cores] += 1 / Ic.shape[0] / len(cur_cores)
    seed_init = torch.log(seed_init)

    Ic = Ic.to(device)
    Fc = torch.FloatTensor(Fc).to(device)
    model = CoreGroupConstructionNeural(Ic, Fc, seed_init, w_d, w_s, device, hidden_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    model.train()
    min_loss = np.inf
    tol = 0
    pbar = tqdm.tqdm(range(epoch), desc="Neural Core Group", unit="epoch")
    for _ in pbar:
        cur_seed_prob = torch.nn.Softmax(dim=0)(model.seed_prob.clone())

        if n_batch == 0:
            loss = model()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        else:
            loss = torch.tensor(0.0, device=device)
            indices = np.arange(Ic.shape[0])
            np.random.shuffle(indices)
            edge_split = np.array_split(indices, n_batch)

            for b in range(n_batch):
                cur_edge_split = edge_split[b]
                if cur_edge_split.size == 0:
                    continue
                optimizer.zero_grad()
                cur_loss = model(cur_edge_split)
                cur_loss.backward()
                optimizer.step()
                loss = loss + cur_loss.detach()
                del cur_loss

        pbar.set_postfix(loss=f"{loss.item():.4f}")
        cur_loss_value = loss.item()
        if min_loss > cur_loss_value:
            tol = 0
            min_loss = cur_loss_value
            best_seed_prob = cur_seed_prob.detach().cpu()
            best_core_state = copy.deepcopy(model.core_bank.state_dict())
        else:
            tol += 1

        if tol == 30:
            print("early stop!")
            break

    core_bank_ckpt = {
        "k": model.k,
        "hidden_dim": hidden_dim,
        "state_dict": best_core_state,
    }
    return core_bank_ckpt, best_seed_prob


def NoAHfit_continuous_neural_fringe(
    If,
    Ff,
    Fcg,
    epoch,
    lr,
    w_d,
    w_s,
    n_batch,
    seed,
    device,
    hidden_dim=8
):
    np.random.seed(seed)

    If = If.to(device)
    Ff = torch.FloatTensor(Ff).to(device)
    Fcg = Fcg.to(device)
    model = FringeAttachmentNeural(
        If,
        Ff,
        Fcg,
        w_d,
        w_s,
        device,
        hidden_dim
    ).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    model.train()
    min_loss = np.inf
    tol = 0
    pbar = tqdm.tqdm(range(epoch), desc="Neural Fringe Attachment", unit="epoch")
    for _ in pbar:
        if n_batch == 0:
            loss = model()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        else:
            loss = torch.tensor(0.0, device=device)
            indices = np.arange(If.shape[0])
            np.random.shuffle(indices)
            edge_split = np.array_split(indices, n_batch)

            for b in range(n_batch):
                cur_edge_split = edge_split[b]
                if cur_edge_split.size == 0:
                    continue
                optimizer.zero_grad()
                cur_loss = model(cur_edge_split)
                cur_loss.backward()
                optimizer.step()
                loss = loss + cur_loss.detach()
                del cur_loss

        pbar.set_postfix(loss=f"{loss.item():.4f}")
        cur_loss_value = loss.item()
        if min_loss > cur_loss_value:
            tol = 0
            min_loss = cur_loss_value
            best_fringe_state = copy.deepcopy(model.fringe_bank.state_dict())
        else:
            tol += 1

        if tol == 30:
            print("early stop!")
            break

    fringe_bank_ckpt = {
        "k": model.k,
        "hidden_dim": hidden_dim,
        "state_dict": best_fringe_state,
    }
    return fringe_bank_ckpt

