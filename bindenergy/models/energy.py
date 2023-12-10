import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random, math

from bindenergy.utils.utils import _expansion, _density, _score
from bindenergy.models.frame import *


class FARigidModel(nn.Module):

    def __init__(self):
        super(FARigidModel, self).__init__()

    def mean(self, X, mask):
        return (X * mask[...,None]).sum(dim=1) / mask[...,None].sum(dim=1).clamp(min=1e-6)

    def inertia(self, X, mask):
        inner = (X ** 2).sum(dim=-1)
        inner = inner[...,None,None] * torch.eye(3).to(X)[None,None,...]  # [B,N,3,3]
        outer = X.unsqueeze(2) * X.unsqueeze(3)  # [B,N,3,3]
        inertia = (inner - outer) * mask[...,None,None]
        return 0.1 * inertia.sum(dim=1)  # [B,3,3]

    def rotate(self, X, w):
        B, N = X.size(0), X.size(1)
        X = X.reshape(B, N * 14, 3)
        w = w.unsqueeze(1).expand_as(X)  # [B,N,3]
        c = w.norm(dim=-1, keepdim=True)  # [B,N,1]
        c1 = torch.sin(c) / c.clamp(min=1e-6)
        c2 = (1 - torch.cos(c)) / (c ** 2).clamp(min=1e-6)
        cross = lambda a,b : torch.cross(a, b, dim=-1)
        X = X + c1 * cross(w, X) + c2 * cross(w, cross(w, X))
        return X.view(B, N, 14, 3)


class FAEnergyModel(FARigidModel):

    def __init__(self, args):
        super(FAEnergyModel, self).__init__()
        self.mse_loss = nn.MSELoss()
        self.threshold = args.threshold
        self.encoder = FAEncoder(args)
        self.U_o = nn.Linear(args.hidden_size, args.hidden_size)
        self.W_o = nn.Sequential(
                nn.SiLU(),
                nn.Linear(args.hidden_size, 1)
        )
        self.theta_range = np.linspace(0.1, np.pi, 100)
        self.sigma_range = np.linspace(0, 10.0, 100) + 0.1
        self.expansion = [_expansion(self.theta_range, sigma) for sigma in self.sigma_range]
        self.density = [_density(exp, self.theta_range) for exp in self.expansion]
        self.score = [_score(exp, self.theta_range, sigma) for exp, sigma in zip(self.expansion, self.sigma_range)]

    def forward(self, binder, target):
        true_X, bind_S, bind_A, _ = binder
        tgt_X, tgt_S, tgt_A, _ = target

        B, N, M = bind_S.size(0), bind_S.size(1), tgt_X.size(1)
        bind_mask = bind_A[:,:,1].clamp(max=1).float()
        tgt_mask = tgt_A[:,:,1].clamp(max=1).float()
        mask_2D = bind_mask.unsqueeze(2) * tgt_mask.unsqueeze(1)

        # Random rigid transform
        sidx = [random.randint(0, 99) for _ in range(B)]  # 0 is padding
        sigma = torch.tensor([self.sigma_range[i] for i in sidx]).float().cuda()
        tidx = [np.random.choice(list(range(100)), p=self.density[i]) for i in sidx]
        theta = torch.tensor([self.theta_range[i] for i in tidx]).float().cuda()
        w = torch.randn(B, 3).cuda() 
        hat_w = F.normalize(w, dim=-1)
        w = hat_w * theta.unsqueeze(-1)
        eps = np.random.uniform(0.1, 1.0, size=B)
        eps = torch.tensor(eps).float().cuda().unsqueeze(-1)
        hat_t = torch.randn(B, 3).cuda() * eps
        # Apply
        center = self.mean(true_X[:,:,1], bind_mask)
        bind_X = true_X - center[:,None,None,:]
        bind_X = self.rotate(bind_X, w) + hat_t[:,None,None,:]
        bind_X = bind_X + center[:,None,None,:]
        bind_X = bind_X.requires_grad_()
        # Contact map
        D = (bind_X[:,:,None,1,:] - tgt_X[:,None,:,1,:]).norm(dim=-1)  # [B,N,M]
        mask_2D = mask_2D * (D < self.threshold).float()

        h = self.encoder((bind_X, bind_S, bind_A, None), target)  # [B,N+M,H]
        h = self.U_o(h[:,:N]).unsqueeze(2) + self.U_o(h[:,N:]).unsqueeze(1)  # [B,N,M,H]
        energy = self.W_o(h).squeeze(-1)  # [B,N,M]
        energy = (energy * mask_2D).sum(dim=(1,2))  # [B]

        # Translation force
        f = torch.autograd.grad(energy.sum(), bind_X, create_graph=True, retain_graph=True)[0]
        f = f[:,:,1]  # force [B,N,3]
        t = self.mean(f, bind_mask)

        # Euler's rotation equation
        center = self.mean(bind_X[:,:,1], bind_mask)
        bind_X = bind_X - center[:,None,None,:]    # set rotation center to zero
        G = torch.cross(bind_X[:,:,1], f, dim=-1)  # [B,N,3]
        G = (G * bind_mask[...,None]).sum(dim=1)   # [B,3] angular momentum
        I = self.inertia(bind_X[:,:,1], bind_mask) # [B,3,3] inertia matrix
        w = torch.linalg.solve(I.detach(), G)  # angular velocity

        # Score matching loss
        score = torch.tensor([self.score[i][j] for i,j in zip(sidx, tidx)]).float().cuda()
        wloss = self.mse_loss(w, hat_w * score.unsqueeze(-1))
        tloss = self.mse_loss(t * eps, -hat_t / eps)
        return wloss + tloss

    def predict(self, binder, target):
        bind_X, bind_S, bind_A, _ = binder
        tgt_X, tgt_S, tgt_A, _ = target

        B, N, M = bind_S.size(0), bind_S.size(1), tgt_X.size(1)
        bind_mask = bind_A[:,:,1].clamp(max=1).float()
        tgt_mask = tgt_A[:,:,1].clamp(max=1).float()
        mask_2D = bind_mask.unsqueeze(2) * tgt_mask.unsqueeze(1)

        D = (bind_X[:,:,None,1,:] - tgt_X[:,None,:,1,:]).norm(dim=-1)  # [B,N,M]
        mask_2D = mask_2D * (D < self.threshold).float()

        h = self.encoder((bind_X, bind_S, bind_A, None), target)  # [B,N+M,H]
        h = self.U_o(h[:,:N]).unsqueeze(2) + self.U_o(h[:,N:]).unsqueeze(1)  # [B,N,M,H]
        energy = self.W_o(h).squeeze(-1)  # [B,N,M]
        energy = (energy * mask_2D).sum(dim=(1,2))  # [B]
        return energy


class AllAtomEnergyModel(FARigidModel):

    def __init__(self, args):
        super(AllAtomEnergyModel, self).__init__()
        self.mse_loss = nn.MSELoss()
        self.threshold = args.threshold
        self.encoder = AllAtomEncoder(args)
        self.W_o = nn.Sequential(
                nn.Linear(args.hidden_size, args.hidden_size),
                nn.SiLU(),
                nn.Linear(args.hidden_size, args.hidden_size),
        )
        self.U_o = nn.Sequential(
                nn.Linear(args.hidden_size, args.hidden_size),
                nn.SiLU(),
        )
        self.theta_range = np.linspace(0.1, np.pi, 100)
        self.sigma_range = np.linspace(0, 10.0, 100) + 0.1
        self.expansion = [_expansion(self.theta_range, sigma) for sigma in self.sigma_range]
        self.density = [_density(exp, self.theta_range) for exp in self.expansion]
        self.score = [_score(exp, self.theta_range, sigma) for exp, sigma in zip(self.expansion, self.sigma_range)]
        self.eye = torch.eye(3).unsqueeze(0).cuda()

    def sc_rotate(self, X, w):
        w = w.unsqueeze(1).expand_as(X)  # [B,N,3]
        c = w.norm(dim=-1, keepdim=True)  # [B,N,1]
        c1 = torch.sin(c) / c.clamp(min=1e-6)
        c2 = (1 - torch.cos(c)) / (c ** 2).clamp(min=1e-6)
        cross = lambda a,b : torch.cross(a, b, dim=-1)
        return X + c1 * cross(w, X) + c2 * cross(w, cross(w, X))

    def forward(self, binder, target, use_sidechain=True):
        true_X, bind_S, bind_A, _ = binder
        tgt_X, tgt_S, tgt_A, _ = target

        B, N, M = bind_S.size(0), bind_S.size(1), tgt_X.size(1)
        bind_mask = bind_A[:,:,1].clamp(max=1).float()
        tgt_mask = tgt_A[:,:,1].clamp(max=1).float()
        bind_A = bind_A * (true_X.norm(dim=-1) > 1e-4).long() 
        tgt_A = tgt_A * (tgt_X.norm(dim=-1) > 1e-4).long() 
        sc_mask = (bind_A[:,:,4:] > 0).float().view(B*N, 10)
        has_sc = sc_mask.sum(dim=-1).clamp(max=1)

        # Random backbone rotation + translation
        sidx = [random.randint(0, 99) for _ in range(B)]
        sigma = torch.tensor([self.sigma_range[i] for i in sidx]).float().cuda()
        tidx = [np.random.choice(list(range(100)), p=self.density[i]) for i in sidx]
        theta = torch.tensor([self.theta_range[i] for i in tidx]).float().cuda()
        w = torch.randn(B, 3).cuda()
        hat_w = F.normalize(w, dim=-1)
        w = hat_w * theta.unsqueeze(-1)
        eps = np.random.uniform(0.1, 1.0, size=B)
        eps = torch.tensor(eps).float().cuda().unsqueeze(-1)
        hat_t = torch.randn(B, 3).cuda() * eps
        # Apply
        center = self.mean(true_X[:,:,1], bind_mask)
        bind_X = true_X - center[:,None,None,:]
        bind_X = self.rotate(bind_X, w) + hat_t[:,None,None,:]
        bind_X = bind_X + center[:,None,None,:]

        # Random side chain rotation
        aidx = [random.randint(0, 99) for _ in range(B * N)]
        sigma = torch.tensor([self.sigma_range[i] for i in aidx]).float().cuda()
        bidx = [np.random.choice(list(range(100)), p=self.density[i]) for i in aidx]
        theta = torch.tensor([self.theta_range[i] for i in bidx]).float().cuda()
        u = torch.randn(B*N, 3).cuda() 
        hat_u = F.normalize(u, dim=-1)
        u = hat_u * theta.unsqueeze(-1)
        # Apply
        backbone = bind_X[:,:,:4].clone()
        center = bind_X[:,:,1:2,:].clone()  # CA is the rotation center
        bind_X = bind_X - center
        bind_X = bind_X.view(B*N, 14, 3)
        bind_X = self.sc_rotate(bind_X, u)
        bind_X = bind_X.view(B, N, 14, 3) + center
        bind_X = torch.cat((backbone, bind_X[:,:,4:]), dim=-2)
        bind_X = bind_X * (bind_A > 0).float().unsqueeze(-1)
        bind_X = bind_X.requires_grad_()

        # Contact map
        mask_2D = (bind_A > 0).float().view(B,N*14,1) * (tgt_A > 0).float().view(B,1,M*14)
        D = (bind_X.view(B,N*14,1,3) - tgt_X.view(B,1,M*14,3)).norm(dim=-1)  # [B,N*14,M*14]
        mask_2D = mask_2D * (D < self.threshold).float()
        # Energy
        h = self.encoder(
                (bind_X, bind_S, bind_A, None),
                (tgt_X, tgt_S, tgt_A, None),
        )  # [B,N+M,14,H]
        bind_h = self.W_o(h[:, :N]).view(B, N*14, -1)
        tgt_h = self.U_o(h[:, N:]).view(B, M*14, -1)
        energy = torch.matmul(bind_h, tgt_h.transpose(1,2))  # [B,N*14,M*14]
        energy = (energy * mask_2D).sum(dim=(1,2))  # [B]

        # Translation force
        f = torch.autograd.grad(energy.sum(), bind_X, create_graph=True, retain_graph=True)[0]
        t = self.mean(f[:,:,1], bind_mask)

        # Backbone torque
        center = self.mean(bind_X[:,:,1], bind_mask)
        bind_X = bind_X - center[:,None,None,:]    # set rotation center to zero
        G = torch.cross(bind_X[:,:,1], f[:,:,1], dim=-1)  # [B,N,3]
        G = (G * bind_mask[...,None]).sum(dim=1)   # [B,3] angular momentum
        I = self.inertia(bind_X[:,:,1], bind_mask) # [B,3,3] inertia matrix
        w = torch.linalg.solve(I.detach(), G)  # angular velocity

        # Side chain torque
        center = bind_X[:,:,1:2,:]  # CA is the rotation center
        bind_X = bind_X[:,:,4:] - center   # set rotation center to zero
        bind_X = bind_X.view(B*N, 10, 3)
        f = f[:,:,4:].reshape(B*N, 10, 3)
        G = torch.cross(bind_X, f, dim=-1)
        G = (G * sc_mask[...,None]).sum(dim=1)   # [B*N,3] angular momentum
        I = self.inertia(bind_X, sc_mask)    # [B*N,3,3] inertia matrix
        I = I + self.eye * (1 - has_sc)[:,None,None]  # avoid zero inertia
        u = torch.linalg.solve(I.detach(), G)  # [B*N, 3] angular velocity

        # Backbone score matching loss
        score = torch.tensor([self.score[i][j] for i,j in zip(sidx, tidx)]).float().cuda()
        wloss = self.mse_loss(w, hat_w * score.unsqueeze(-1))
        tloss = self.mse_loss(t * eps, -hat_t / eps)
        # Side chain score matching loss
        score = torch.tensor([self.score[i][j] for i,j in zip(aidx, bidx)]).float().cuda()
        uloss = (u - hat_u * score.unsqueeze(-1)) ** 2
        uloss = (uloss.sum(-1) * has_sc).sum() / has_sc.sum().clamp(min=1e-6)
        return wloss + tloss + uloss * int(use_sidechain)

    def predict(self, binder, target, visual=False):
        bind_X, bind_S, bind_A, _ = binder
        tgt_X, tgt_S, tgt_A, _ = target

        B, N, M = bind_S.size(0), bind_S.size(1), tgt_X.size(1)
        bind_mask = bind_A[:,:,1].clamp(max=1).float()
        tgt_mask = tgt_A[:,:,1].clamp(max=1).float()
        bind_A = bind_A * (bind_X.norm(dim=-1) > 1e-4).long() 
        tgt_A = tgt_A * (tgt_X.norm(dim=-1) > 1e-4).long() 

        mask_2D = (bind_A > 0).float().view(B,N*14,1) * (tgt_A > 0).float().view(B,1,M*14)
        D = (bind_X.view(B,N*14,1,3) - tgt_X.view(B,1,M*14,3)).norm(dim=-1)  # [B,N*14,M*14]
        mask_2D = mask_2D * (D < self.threshold).float()

        h = self.encoder(
                (bind_X, bind_S, bind_A, None),
                (tgt_X, tgt_S, tgt_A, None),
        )  # [B,N+M,14,H]

        bind_h = self.W_o(h[:, :N]).view(B, N*14, -1)
        tgt_h = self.U_o(h[:, N:]).view(B, M*14, -1)
        energy = torch.matmul(bind_h, tgt_h.transpose(1,2))  # [B,N*14,M*14]
        if visual:
            energy = energy.view(B, N, 14, M, 14).transpose(2, 3)
            mask_2D = mask_2D.view(B, N, 14, M, 14).transpose(2, 3)
            return (energy * mask_2D).sum(dim=(-1,-2))
        else:
            return (energy * mask_2D).sum(dim=(1,2))  # [B]

    def gaussian_forward(self, binder, target):
        true_X, bind_S, bind_A, _ = binder
        tgt_X, tgt_S, tgt_A, _ = target

        B, N, M = bind_S.size(0), bind_S.size(1), tgt_X.size(1)
        bind_mask = bind_A[:,:,1].clamp(max=1).float()
        tgt_mask = tgt_A[:,:,1].clamp(max=1).float()
        bind_A = bind_A * (true_X.norm(dim=-1) > 1e-4).long()
        tgt_A = tgt_A * (tgt_X.norm(dim=-1) > 1e-4).long()
        atom_mask = (bind_A > 0).float().unsqueeze(-1)

        eps = np.random.uniform(0.1, 1.0, size=B)
        eps = torch.tensor(eps).float().cuda()[:,None,None,None]
        hat_t = torch.randn_like(true_X).cuda() * eps
        bind_X = true_X + hat_t * atom_mask
        bind_X = bind_X.requires_grad_()

        # Contact map
        mask_2D = (bind_A > 0).float().view(B,N*14,1) * (tgt_A > 0).float().view(B,1,M*14)
        D = (bind_X.view(B,N*14,1,3) - tgt_X.view(B,1,M*14,3)).norm(dim=-1)  # [B,N*14,M*14]
        mask_2D = mask_2D * (D < self.threshold).float()

        # Energy
        h = self.encoder(
                (bind_X, bind_S, bind_A, None),
                (tgt_X, tgt_S, tgt_A, None),
        )  # [B,N+M,14,H]
        bind_h = self.W_o(h[:, :N]).view(B, N*14, -1)
        tgt_h = self.U_o(h[:, N:]).view(B, M*14, -1)
        energy = torch.matmul(bind_h, tgt_h.transpose(1,2))  # [B,N*14,M*14]
        energy = (energy * mask_2D).sum(dim=(1,2))  # [B]

        # force
        force = torch.autograd.grad(energy.sum(), bind_X, create_graph=True, retain_graph=True)[0]
        loss = F.mse_loss(force * eps, -hat_t / eps, reduction='none')
        loss = (loss * atom_mask).sum() / atom_mask.sum()
        return loss


class FADecoyModel(AllAtomEnergyModel):

    def __init__(self, args):
        super(FADecoyModel, self).__init__(args)
        self.ce_loss = nn.CrossEntropyLoss()

    def forward(self, binder, target, num_decoy=1):
        true_X, bind_S, bind_A, _ = binder
        tgt_X, tgt_S, tgt_A, _ = target

        B, N, M = bind_S.size(0), bind_S.size(1), tgt_X.size(1)
        bind_mask = bind_A[:,:,1].clamp(max=1).float()
        tgt_mask = tgt_A[:,:,1].clamp(max=1).float()
        bind_A = bind_A * (true_X.norm(dim=-1) > 1e-4).long()
        tgt_A = tgt_A * (tgt_X.norm(dim=-1) > 1e-4).long()

        # Random rigid transform
        decoy_X = []
        for _ in range(num_decoy):
            sidx = [random.randint(0, 99) for _ in range(B)]  # 0 is padding
            sigma = torch.tensor([self.sigma_range[i] for i in sidx]).float().cuda()
            tidx = [np.random.choice(list(range(100)), p=self.density[i]) for i in sidx]
            theta = torch.tensor([self.theta_range[i] for i in tidx]).float().cuda()
            w = torch.randn(B, 3).cuda() 
            hat_w = F.normalize(w, dim=-1)
            w = hat_w * theta.unsqueeze(-1)
            eps = np.random.uniform(0.1, 1.0, size=B)
            eps = torch.tensor(eps).float().cuda().unsqueeze(-1)
            hat_t = torch.randn(B, 3).cuda() * eps
            # Apply
            center = self.mean(true_X[:,:,1], bind_mask)
            bind_X = true_X - center[:,None,None,:]
            bind_X = self.rotate(bind_X, w) + hat_t[:,None,None,:]
            bind_X = bind_X + center[:,None,None,:]
            decoy_X.append(bind_X)

        bind_X = torch.cat([true_X] + decoy_X, dim=0)
        bind_S = torch.cat([bind_S] * (num_decoy+1), dim=0)
        bind_A = torch.cat([bind_A] * (num_decoy+1), dim=0)
        tgt_X = torch.cat([tgt_X] * (num_decoy+1), dim=0)
        tgt_S = torch.cat([tgt_S] * (num_decoy+1), dim=0)
        tgt_A = torch.cat([tgt_A] * (num_decoy+1), dim=0)
        BB = B * (num_decoy + 1)
        # Contact map
        mask_2D = (bind_A > 0).float().view(BB,N*14,1) * (tgt_A > 0).float().view(BB,1,M*14)
        D = (bind_X.view(BB,N*14,1,3) - tgt_X.view(BB,1,M*14,3)).norm(dim=-1)  # [B,N*14,M*14]
        mask_2D = mask_2D * (D < self.threshold).float()
        # Energy
        h = self.encoder(
                (bind_X, bind_S, bind_A, None),
                (tgt_X, tgt_S, tgt_A, None),
        )  # [B,N+M,14,H]
        bind_h = self.W_o(h[:, :N]).view(BB, N*14, -1)
        tgt_h = self.U_o(h[:, N:]).view(BB, M*14, -1)
        energy = torch.matmul(bind_h, tgt_h.transpose(1,2))  # [B,N*14,M*14]
        energy = (energy * mask_2D).sum(dim=(1,2))  # [B]
        energy = torch.stack([energy[B*i:B*i+B] for i in range(num_decoy+1)], dim=-1)  # [B, K+1]
        label = torch.tensor([0] * B).cuda() # 0 is true energy
        loss = self.ce_loss(energy, label)
        return loss
 
