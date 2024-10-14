import torch
import torch.nn as nn
import math
from copy import deepcopy
import torch.nn.functional as F
from bindenergy.models.frame import AllAtomEncoder, FAEncoder
from openfold.utils.rigid_utils import Rigid
from openfold.utils.feats import (
    frames_and_literature_positions_to_atom14_pos,
    torsion_angles_to_frames,
)
from openfold.np.residue_constants import (
    restype_rigid_group_default_frame,
    restype_atom14_to_rigid_group,
    restype_atom14_mask,
    restype_atom14_rigid_group_positions,
)
from bindenergy.utils.utils import self_square_dist, inner_square_dist

def mask_mean(X, mask):
    return (X * mask[...,None]).sum(dim=1) / mask[...,None].sum(dim=1).clamp(min=1e-6)


class SideChainDocker(nn.Module):

    def __init__(self, args):
        super(SideChainDocker, self).__init__()
        self.encoder = AllAtomEncoder(args)
        self.W_o = nn.Sequential(
                nn.Linear(args.hidden_size, args.hidden_size),
                nn.ReLU(),
                nn.Linear(args.hidden_size, 14),
        )

    def _forward(self, binder, target):
        true_X, bind_S, bind_A, bind_aa = binder
        tgt_X, tgt_S, tgt_A, _ = target
        B, N = true_X.size(0), true_X.size(1)
        bind_mask = (bind_A > 0).float()

        bind_X = true_X.clone()
        bind_X[:,:,4:] = 0  # remove all side chains
        bind_A = bind_A * (bind_X.norm(dim=-1) > 1e-4).long()
        tgt_A = tgt_A * (tgt_X.norm(dim=-1) > 1e-4).long()
        bind_S[:,:,0] = 0  # 0 is <pad>
        tgt_S[:,:,0] = 0   # 0 is <pad>

        h = self.encoder(
                (bind_X, bind_S, bind_A, None),
                (tgt_X, tgt_S, tgt_A, None),
        )  # [B,N+M,14,H]

        angles = self.W_o(h[:, :N, 1]).view(B,N,7,2)
        angles = F.normalize(angles, dim=-1)

        backb_to_global = Rigid.from_3_points(
            bind_X[:,:,0], bind_X[:,:,1], bind_X[:,:,2]
        )
        all_frames_to_global = self.torsion_angles_to_frames(
            backb_to_global,
            angles,
            (bind_aa - 1).clamp(min=0),  # our zero is <pad>
        )
        pred_X = self.frames_and_literature_positions_to_atom14_pos(
            all_frames_to_global,
            (bind_aa - 1).clamp(min=0),  # our zero is <pad>
        )
        pred_X = pred_X * bind_mask[...,None]
        loss = F.mse_loss(pred_X[:,:,4:], true_X[:,:,4:], reduction='none').sum(-1)
        loss = (loss * bind_mask[:,:,4:]).sum() / bind_mask[:,:,4:].sum().clamp(min=1e-4)
        pred_X = torch.cat((true_X[:,:,:4], pred_X[:,:,4:]), dim=2)
        return loss, pred_X

    def forward(self, binder, target):
        return self._forward(binder, target)[0]

    def predict(self, binder, target):
        return self._forward(binder, target)[1]

    def _init_residue_constants(self, float_dtype, device):
        if not hasattr(self, "default_frames"):
            self.register_buffer(
                "default_frames",
                torch.tensor(
                    restype_rigid_group_default_frame,
                    dtype=float_dtype,
                    device=device,
                    requires_grad=False,
                ),
                persistent=False,
            )
        if not hasattr(self, "group_idx"):
            self.register_buffer(
                "group_idx",
                torch.tensor(
                    restype_atom14_to_rigid_group,
                    device=device,
                    requires_grad=False,
                ),
                persistent=False,
            )
        if not hasattr(self, "atom_mask"):
            self.register_buffer(
                "atom_mask",
                torch.tensor(
                    restype_atom14_mask,
                    dtype=float_dtype,
                    device=device,
                    requires_grad=False,
                ),
                persistent=False,
            )
        if not hasattr(self, "lit_positions"):
            self.register_buffer(
                "lit_positions",
                torch.tensor(
                    restype_atom14_rigid_group_positions,
                    dtype=float_dtype,
                    device=device,
                    requires_grad=False,
                ),
                persistent=False,
            )

    def torsion_angles_to_frames(self, r, alpha, f):
        # Lazily initialize the residue constants on the correct device
        self._init_residue_constants(alpha.dtype, alpha.device)
        # Separated purely to make testing less annoying
        return torsion_angles_to_frames(r, alpha, f, self.default_frames)

    def frames_and_literature_positions_to_atom14_pos(
        self, r, f  # [*, N, 8]  # [*, N]
    ):
        # Lazily initialize the residue constants on the correct device
        self._init_residue_constants(r.get_rots().dtype, r.get_rots().device)
        return frames_and_literature_positions_to_atom14_pos(
            r,
            f,
            self.default_frames,
            self.group_idx,
            self.atom_mask,
            self.lit_positions,
        )


class FADocker(nn.Module):

    def __init__(self, args):
        super(FADocker, self).__init__()
        self.mse_loss = nn.MSELoss(reduction='none')
        self.rstep = args.nsteps
        self.encoder = FAEncoder(args)
        self.W_x = nn.Linear(args.hidden_size, args.hidden_size)
        self.U_x = nn.Linear(args.hidden_size, args.hidden_size)
        self.T_x = nn.Sequential(nn.ReLU(), nn.Linear(args.hidden_size, 14))

    def _forward(self, binder, target):
        true_X, bind_S, bind_A, _ = binder
        tgt_X, tgt_S, tgt_A, _ = target

        B, N, M = bind_S.size(0), bind_S.size(1), tgt_X.size(1)
        atom_mask = bind_A[:,:,:4].clamp(max=1).float()
        bind_mask = bind_A[:,:,1].clamp(max=1).float()
        tgt_mask = tgt_A[:,:,1].clamp(max=1).float()
        mask = torch.cat([bind_mask, tgt_mask], dim=1)
        true_C, cmask = inner_square_dist(true_X[:,:,:4], atom_mask)

        cdr_mean = mask_mean(true_X[:,:,1], bind_mask)
        bind_X = cdr_mean[:,None,None,:] + torch.randn_like(true_X)

        # Refine
        xloss = dloss = closs = 0
        for _ in range(self.rstep):
            h = self.encoder(
                    (bind_X, bind_S, bind_A, None),
                    (tgt_X, tgt_S, tgt_A, None),
            )
            X = torch.cat([bind_X, tgt_X], dim=1)
            mij = self.W_x(h).unsqueeze(2) + self.U_x(h).unsqueeze(1)  # [B,N,N,H]
            xij = X.unsqueeze(2) - X.unsqueeze(1)  # [B,N,N,L,3]
            xij = xij * self.T_x(mij).unsqueeze(-1)  # [B,N,N,L,3]
            f = torch.sum(xij * mask[:,None,:,None,None], dim=2)  # [B,N,N,L,3] * [B,1,N,1,1]
            f = f / (1e-6 + mask.sum(dim=1)[:,None,None,None])    # [B,N,L,3] / [B,1,1,1]
            X = X + f.clamp(min=-20.0, max=20.0)
            bind_X = X[:, :N]
            # loss
            bind_D, dmask = self_square_dist(bind_X, bind_mask)
            bind_C, cmask = inner_square_dist(bind_X[:,:,:4], atom_mask)
            dloss_t = F.relu(3.8**2 - bind_D)
            closs_t = self.mse_loss(bind_C, true_C)
            xloss_t = self.mse_loss(bind_X[:,:,:4], true_X[:,:,:4]).sum(dim=2)
            xloss = xloss + xloss_t * bind_mask.unsqueeze(-1)
            dloss = dloss + dloss_t * dmask
            closs = closs + closs_t * cmask

        loss = xloss.sum() / bind_mask.sum() + 10 * dloss.sum() / dmask.sum() + closs.sum() / cmask.sum()
        return loss, bind_X

    def forward(self, binder, target):
        loss, _ = self._forward(binder, target)
        return loss

    def predict(self, binder, target):
        _, bind_X = self._forward(binder, target)
        return bind_X.detach()
