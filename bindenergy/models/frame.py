import torch
import torch.nn as nn
import numpy as np
from sru import SRUpp
from bindenergy.utils.utils import * 


class FrameAveraging(nn.Module):

    def __init__(self):
        super(FrameAveraging, self).__init__()
        self.ops = torch.tensor([
                [i,j,k] for i in [-1,1] for j in [-1,1] for k in [-1,1]
        ]).cuda()

    def create_frame(self, X, mask):
        mask = mask.unsqueeze(-1)
        center = (X * mask).sum(dim=1) / mask.sum(dim=1)
        X = X - center.unsqueeze(1) * mask  # [B,N,3]
        C = torch.bmm(X.transpose(1,2), X)  # [B,3,3] (Cov)
        _, V = torch.symeig(C.detach(), True)  # [B,3,3]
        F_ops = self.ops.unsqueeze(1).unsqueeze(0) * V.unsqueeze(1)  # [1,8,1,3] x [B,1,3,3] -> [B,8,3,3]
        h = torch.einsum('boij,bpj->bopi', F_ops.transpose(2,3), X)  # transpose is inverse [B,8,N,3]
        h = h.view(X.size(0) * 8, X.size(1), 3)
        return h, F_ops.detach(), center

    def invert_frame(self, X, mask, F_ops, center):
        X = torch.einsum('boij,bopj->bopi', F_ops, X)
        X = X.mean(dim=1)  # frame averaging
        X = X + center.unsqueeze(1)
        return X * mask.unsqueeze(-1)


class FAEncoder(FrameAveraging):

    def __init__(self, args):
        super(FAEncoder, self).__init__()
        self.bind_embedding = nn.Embedding(args.vocab_size, args.hidden_size)
        self.tgt_embedding = nn.Embedding(args.vocab_size, args.hidden_size)
        self.W_bind = nn.Linear(args.bert_size, args.hidden_size)
        self.W_tgt = nn.Linear(args.esm_size, args.hidden_size)
        self.encoder = SRUpp(
                args.hidden_size + 3,
                args.hidden_size // 2,
                args.hidden_size // 2,
                num_layers=args.depth,
                dropout=args.dropout,
                bidirectional=True,
        )

    def forward(self, binder, target):
        bind_X, bind_S, bind_A, _ = binder
        tgt_X, tgt_S, tgt_A, _ = target

        B, N, M = bind_S.size(0), bind_S.size(1), tgt_S.size(1)
        bind_mask = bind_A[:,:,1].clamp(max=1).float()
        tgt_mask = tgt_A[:,:,1].clamp(max=1).float()
        mask = torch.cat([bind_mask, tgt_mask], dim=1)

        if bind_S.dim() == 2:
            bind_S = self.bind_embedding(bind_S)
        else:
            bind_S = self.W_bind(bind_S)

        if tgt_S.dim() == 2:
            tgt_S = self.tgt_embedding(tgt_S)
        else:
            tgt_S = self.W_tgt(tgt_S)

        h_S = torch.cat([bind_S, tgt_S], dim=1)
        X = torch.cat([bind_X, tgt_X], dim=1)
        h_X, _, _ = self.create_frame(X[:,:,1], mask)
        h_S = h_S.unsqueeze(1).expand(-1, 8, -1, -1).reshape(B*8, N+M, -1)
        mask = mask.unsqueeze(1).expand(-1, 8, -1).reshape(B*8, N+M)

        h = torch.cat([h_X, h_S], dim=-1)
        h, _, _ = self.encoder(
                h.transpose(0, 1),
                mask_pad=(~mask.transpose(0, 1).bool())
        )
        h = h.transpose(0, 1).view(B, 8, N+M, -1)
        return h.mean(dim=1)  # frame averaging


class AllAtomEncoder(FrameAveraging):

    def __init__(self, args):
        super(AllAtomEncoder, self).__init__()
        self.W_bind = nn.Linear(args.bert_size, args.hidden_size)
        self.W_tgt = nn.Linear(args.esm_size, args.hidden_size)
        self.A_bind = nn.Embedding(args.vocab_size, args.hidden_size)
        self.A_tgt = nn.Embedding(args.vocab_size, args.hidden_size)
        self.encoder = SRUpp(
                args.hidden_size * 2 + 3,
                args.hidden_size // 2,
                args.hidden_size // 2,
                num_layers=args.depth,
                dropout=args.dropout,
                bidirectional=True,
        )

    def forward(self, binder, target):
        bind_X, bind_S, bind_A, _ = binder
        tgt_X, tgt_S, tgt_A, _ = target

        # flatten
        B, N, M = bind_S.size(0), bind_S.size(1), tgt_S.size(1)
        bind_S = bind_S[:,:,None,:].expand(-1,-1,14,-1).reshape(B, N*14, -1)
        tgt_S = tgt_S[:,:,None,:].expand(-1,-1,14,-1).reshape(B, M*14, -1)
        bind_A = bind_A.view(B, N*14)
        bind_X = bind_X.view(B, N*14, 3)
        tgt_A = tgt_A.view(B, M*14)
        tgt_X = tgt_X.view(B, M*14, 3)

        bind_mask = bind_A.clamp(max=1).float()
        tgt_mask = tgt_A.clamp(max=1).float()
        mask = torch.cat([bind_mask, tgt_mask], dim=1)

        bind_S = torch.cat([self.W_bind(bind_S), self.A_bind(bind_A)], dim=-1)
        tgt_S = torch.cat([self.W_tgt(tgt_S), self.A_tgt(tgt_A)], dim=-1)

        h_S = torch.cat([bind_S, tgt_S], dim=1)
        X = torch.cat([bind_X, tgt_X], dim=1)
        h_X, _, _ = self.create_frame(X, mask)
        h_S = h_S.unsqueeze(1).expand(-1,8,-1,-1).reshape(B*8, (N+M)*14, -1)
        mask = mask.unsqueeze(1).expand(-1,8,-1).reshape(B*8, (N+M)*14)

        h = torch.cat([h_X, h_S], dim=-1)
        h, _, _ = self.encoder(
                h.transpose(0, 1),
                mask_pad=(~mask.transpose(0, 1).bool())
        )
        h = h.transpose(0, 1).view(B, 8, (N+M)*14, -1)
        h = h.mean(dim=1)  # frame averaging
        return h.view(B, N+M, 14, -1)


class SingleChainEncoder(FrameAveraging):

    def __init__(self, args):
        super(SingleChainEncoder, self).__init__()
        self.W_i = nn.Linear(args.esm_size, args.hidden_size)
        self.A_i = nn.Embedding(args.vocab_size, args.hidden_size)
        self.encoder = SRUpp(
                args.hidden_size * 2 + 3,
                args.hidden_size // 2,
                args.hidden_size // 2,
                num_layers=args.depth,
                dropout=args.dropout,
                bidirectional=True,
        )

    def forward(self, X, S, A, V):
        B, N = S.size(0), S.size(1)
        S = S.unsqueeze(2).expand(-1,-1,14,-1).reshape(B, N*14, -1)
        A = A.view(B, N*14)
        X = X.view(B, N*14, 3)
        mask = A.clamp(max=1).float()

        h_S = torch.cat([self.W_i(S), self.A_i(A)], dim=-1)
        h_X, _, _ = self.create_frame(X, mask)
        h_S = h_S.unsqueeze(1).expand(-1,8,-1,-1).reshape(B*8, N*14, -1)
        mask = mask.unsqueeze(1).expand(-1,8,-1).reshape(B*8, N*14)

        h = torch.cat([h_X, h_S], dim=-1)
        h, _, _ = self.encoder(
                h.transpose(0, 1),
                mask_pad=(~mask.transpose(0, 1).bool())
        )
        h = h.transpose(0, 1).view(B, 8, N*14, -1)
        return h.mean(dim=1)  # frame averaging