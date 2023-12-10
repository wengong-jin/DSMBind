import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random, math


    def predict_rigid(self, bind_h, bind_X, bind_mask, h, X, mask, W_t, U_t, T_t, W_r, U_r, T_r):
        # translation force
        mask_2D = bind_mask.unsqueeze(2) * mask.unsqueeze(1)  # [B,N,N+M]
        mij = W_t(bind_h).unsqueeze(2) + U_t(h).unsqueeze(1)  # [B,N,N+M,H]
        xij = bind_X.unsqueeze(2) - X.unsqueeze(1)  # [B,N,N+M,14,3]
        xij = xij[:,:,:,1] * T_t(mij)  # [B,N,N+M,3]
        t = torch.sum(xij * mask_2D[...,None], dim=(1,2))
        t = t / mask_2D[...,None].sum(dim=(1,2)).clamp(min=1e-6)  # [B,3]
        # rotation force
        mij = W_r(bind_h).unsqueeze(2) + U_r(h).unsqueeze(1)  # [B,N,N+M,H]
        xij = bind_X.unsqueeze(2) - X.unsqueeze(1)  # [B,N,N+M,14,3]
        xij = xij[:,:,:,1] * T_r(mij)  # [B,N,N+M,3]
        f = torch.sum(xij * mask[:,None,:,None], dim=2)   # [B,N,3]
        f = f / mask[:,None,:,None].sum(dim=2).clamp(min=1e-6)  # [B,N,3]
        # Euler equation
        G = torch.cross(bind_X[:,:,1], f, dim=-1)  # [B,N,3]
        G = (G * bind_mask[...,None]).sum(dim=1)   # [B,3] angular momentum
        I = self.inertia(bind_X[:,:,1], bind_mask) # [B,3,3] inertia matrix
        w = torch.linalg.solve(I.detach(), G)  # angular velocity
        return w, t

