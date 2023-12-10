import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def kabsch(A, B):
    a_mean = A.mean(dim=1, keepdims=True)
    b_mean = B.mean(dim=1, keepdims=True)
    A_c = A - a_mean
    B_c = B - b_mean
    # Covariance matrix
    H = torch.bmm(A_c.transpose(1,2), B_c)  # [B, 3, 3]
    U, S, V = torch.svd(H)
    # Flip
    sign = (torch.det(U) * torch.det(V) < 0.0)
    if sign.any():
        S[sign] = S[sign] * (-1)
        U[sign,:] = U[sign,:] * (-1)
    # Rotation matrix
    R = torch.bmm(V, U.transpose(1,2))  # [B, 3, 3]
    # Translation vector
    t = b_mean - torch.bmm(R, a_mean.transpose(1,2)).transpose(1,2)
    A_aligned = torch.bmm(R, A.transpose(1,2)).transpose(1,2) + t
    return A_aligned, R, t

# X: [B, N, 4, 3], R: [B, 3, 3], t: [B, 3]
def rigid_transform(X, R, t):
    B, N, L = X.size(0), X.size(1), X.size(2)
    X = X.reshape(B, N * L, 3)
    X = torch.bmm(R, X.transpose(1,2)).transpose(1,2) + t
    return X.view(B, N, L, 3)

# A: [B, N, 3], B: [B, N, 3], mask: [B, N]
def compute_rmsd(A, B, mask):
    A_aligned, _, _ = kabsch(A, B)
    rmsd = ((A_aligned - B) ** 2).sum(dim=-1)
    rmsd = torch.sum(rmsd * mask, dim=-1) / (mask.sum(dim=-1) + 1e-6)
    return rmsd.sqrt()

# A: [B, N, 3], B: [B, N, 3], mask: [B, N]
def compute_rmsd_no_align(A, B, mask):
    rmsd = ((A - B) ** 2).sum(dim=-1)
    rmsd = torch.sum(rmsd * mask, dim=-1) / (mask.sum(dim=-1) + 1e-6)
    return rmsd.sqrt()

def eig_coord(X, mask):
    D, mask_2D = self_square_dist(X, torch.ones_like(mask))
    return eig_coord_from_dist(D)

def eig_coord_from_dist(D):
    M = (D[:, :1, :] + D[:, :, :1] - D) / 2
    L, V = torch.linalg.eigh(M)
    L = torch.diag_embed(L)
    X = torch.matmul(V, L.clamp(min=0).sqrt())
    return X[:, :, -3:].detach()

def self_square_dist(X, mask):
    X = X[:, :, 1] 
    dX = X.unsqueeze(1) - X.unsqueeze(2)  # [B, 1, N, 3] - [B, N, 1, 3]
    D = torch.sum(dX**2, dim=-1)
    mask_2D = mask.unsqueeze(1) * mask.unsqueeze(2)  # [B, 1, N] x [B, N, 1]
    mask_2D = mask_2D * (1 - torch.eye(mask.size(1))[None,:,:]).to(mask_2D)
    return D, mask_2D

def cross_square_dist(X, Y, xmask, ymask):
    X, Y = X[:, :, 1], Y[:, :, 1]
    dxy = X.unsqueeze(2) - Y.unsqueeze(1)  # [B, N, 1, 3] - [B, 1, M, 3]
    D = torch.sum(dxy ** 2, dim=-1)
    mask_2D = xmask.unsqueeze(2) * ymask.unsqueeze(1)  # [B, N, 1] x [B, 1, M]
    return D, mask_2D

def full_square_dist(X, Y, XA, YA, contact=False, remove_diag=False):
    B, N, M, L = X.size(0), X.size(1), Y.size(1), Y.size(2)
    X = X.view(B, N * L, 3)
    Y = Y.view(B, M * L, 3)
    dxy = X.unsqueeze(2) - Y.unsqueeze(1)  # [B, NL, 1, 3] - [B, 1, ML, 3]
    D = torch.sum(dxy ** 2, dim=-1)
    D = D.view(B, N, L, M, L)
    D = D.transpose(2, 3).reshape(B, N, M, L*L)

    xmask = XA.clamp(max=1).float().view(B, N * L)
    ymask = YA.clamp(max=1).float().view(B, M * L)
    mask = xmask.unsqueeze(2) * ymask.unsqueeze(1)  # [B, NL, 1] x [B, 1, ML]
    mask = mask.view(B, N, L, M, L)
    mask = mask.transpose(2, 3).reshape(B, N, M, L*L)
    if remove_diag:
        mask = mask * (1 - torch.eye(N)[None,:,:,None]).to(mask)

    if contact:
        D = D + 1e6 * (1 - mask)
        return D.amin(dim=-1), mask.amax(dim=-1)
    else:
        return D, mask

def compute_dihedrals(X, eps=1e-7):
    # First 3 coordinates are N, CA, C
    X = X[:,:,:3,:].reshape(X.shape[0], 3*X.shape[1], 3)

    # Shifted slices of unit vectors
    dX = X[:,1:,:] - X[:,:-1,:]
    U = F.normalize(dX, dim=-1)
    u_2 = U[:,:-2,:]
    u_1 = U[:,1:-1,:]
    u_0 = U[:,2:,:]

    # Backbone normals
    n_2 = F.normalize(torch.cross(u_2, u_1), dim=-1)
    n_1 = F.normalize(torch.cross(u_1, u_0), dim=-1)

    # Angle between normals
    cosD = (n_2 * n_1).sum(-1)
    cosD = torch.clamp(cosD, -1+eps, 1-eps)
    D = torch.sign((u_2 * n_1).sum(-1)) * torch.acos(cosD)

    D = F.pad(D, (3,0), 'constant', 0)
    D = D.view((D.size(0), int(D.size(1)/3), 3))
    phi, psi, omega = torch.unbind(D,-1)

    # Lift angle representations to the circle
    D_features = torch.cat((torch.cos(D), torch.sin(D)), 2)
    return D_features

def _expansion(theta, sigma, L=2000):  # the summation term only
    p = 0
    for l in range(L):
        p += (2 * l + 1) * np.exp(-l * (l + 1) * sigma**2) * np.sin(theta * (l + 1 / 2)) / np.sin(theta / 2)
    return p

def _density(expansion, theta):
    density = expansion * (1 - np.cos(theta)) / np.pi
    density = np.clip(density, 0, 1000)
    return density / density.sum()

def _score(exp, theta, sigma, L=2000):
    dSigma = 0
    for l in range(L):
        hi = np.sin(theta * (l + 1 / 2))
        dhi = (l + 1 / 2) * np.cos(theta * (l + 1 / 2))
        lo = np.sin(theta / 2)
        dlo = 1 / 2 * np.cos(theta / 2)
        dSigma += (2 * l + 1) * np.exp(-l * (l + 1) * sigma**2) * (lo * dhi - hi * dlo) / (lo ** 2)
    return dSigma / exp + np.sin(theta) / (1 - np.cos(theta))
