import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random, math
from sidechainnet.utils.measure import get_seq_coords_and_angles
from prody import *
from rdkit import Chem
from rdkit.Chem import PandasTools
from bindenergy.utils.utils import _expansion, _density, _score
from bindenergy.models.frame import FAEncoder, AllAtomEncoder
from bindenergy.models.energy import FARigidModel
from bindenergy.data.drug import DrugDataset
from bindenergy.utils import load_esm_embedding
from chemprop.features import BatchMolGraph, get_atom_fdim, get_bond_fdim, mol2graph
from chemprop.nn_utils import index_select_ND


class MPNEncoder(nn.Module):

    def __init__(self, args):
        super(MPNEncoder, self).__init__()
        self.atom_fdim = get_atom_fdim()
        self.bond_fdim = get_bond_fdim()
        self.hidden_size = args.hidden_size
        self.depth = args.mpn_depth
        self.dropout = args.dropout
        self.bias = False
        self.layers_per_message = 1
        self.undirected = False
        self.atom_messages = False
        self.features_only = False
        self.use_input_features = False
        self.args = args

        # Dropout
        self.dropout_layer = nn.Dropout(p=self.dropout)

        # Activation
        self.act_func = nn.ReLU()

        # Input
        input_dim = self.atom_fdim if self.atom_messages else self.bond_fdim
        self.W_i = nn.Linear(input_dim, self.hidden_size, bias=self.bias)

        if self.atom_messages:
            w_h_input_size = self.hidden_size + self.bond_fdim
        else:
            w_h_input_size = self.hidden_size

        # Shared weight matrix across depths (default)
        self.W_h = nn.Linear(w_h_input_size, self.hidden_size, bias=self.bias)

        self.W_o = nn.Linear(self.atom_fdim + self.hidden_size, self.hidden_size)

    def forward(self, mol_graph: BatchMolGraph) -> torch.FloatTensor:
        f_atoms, f_bonds, a2b, b2a, b2revb, a_scope, b_scope = mol_graph.get_components()

        if self.atom_messages:
            a2a = mol_graph.get_a2a()
            a2a = a2a.cuda()

        f_atoms, f_bonds, a2b, b2a, b2revb = f_atoms.cuda(), f_bonds.cuda(), a2b.cuda(), b2a.cuda(), b2revb.cuda()

        # Input
        if self.atom_messages:
            input = self.W_i(f_atoms)  # num_atoms x hidden_size
        else:
            input = self.W_i(f_bonds)  # num_bonds x hidden_size
        message = self.act_func(input)  # num_bonds x hidden_size

        # Message passing
        for depth in range(self.depth - 1):
            if self.undirected:
                message = (message + message[b2revb]) / 2

            if self.atom_messages:
                nei_a_message = index_select_ND(message, a2a)  # num_atoms x max_num_bonds x hidden
                nei_f_bonds = index_select_ND(f_bonds, a2b)  # num_atoms x max_num_bonds x bond_fdim
                nei_message = torch.cat((nei_a_message, nei_f_bonds), dim=2)  # num_atoms x max_num_bonds x hidden + bond_fdim
                message = nei_message.sum(dim=1)  # num_atoms x hidden + bond_fdim
            else:
                # m(a1 -> a2) = [sum_{a0 \in nei(a1)} m(a0 -> a1)] - m(a2 -> a1)
                # message      a_message = sum(nei_a_message)      rev_message
                nei_a_message = index_select_ND(message, a2b)  # num_atoms x max_num_bonds x hidden
                a_message = nei_a_message.sum(dim=1)  # num_atoms x hidden
                rev_message = message[b2revb]  # num_bonds x hidden
                message = a_message[b2a] - rev_message  # num_bonds x hidden

            message = self.W_h(message)
            message = self.act_func(input + message)  # num_bonds x hidden_size
            message = self.dropout_layer(message)  # num_bonds x hidden

        a2x = a2a if self.atom_messages else a2b
        nei_a_message = index_select_ND(message, a2x)  # num_atoms x max_num_bonds x hidden
        a_message = nei_a_message.sum(dim=1)  # num_atoms x hidden
        a_input = torch.cat([f_atoms, a_message], dim=1)  # num_atoms x (atom_fdim + hidden)
        atom_hiddens = self.act_func(self.W_o(a_input))  # num_atoms x hidden
        atom_hiddens = self.dropout_layer(atom_hiddens)  # num_atoms x hidden

        mol_vecs = []
        L = max([a_size for (a_start, a_size) in a_scope])
        for i, (a_start, a_size) in enumerate(a_scope):
            h = atom_hiddens.narrow(0, a_start, a_size)
            h = F.pad(h, (0,0,0,L-a_size))
            mol_vecs.append(h)

        return torch.stack(mol_vecs, dim=0)


class DrugEnergyModel(FARigidModel):

    def __init__(self, args):
        super(DrugEnergyModel, self).__init__()
        self.mse_loss = nn.MSELoss()
        self.threshold = args.threshold
        self.mpn = MPNEncoder(args)
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
        true_X, mol_batch, bind_A, _ = binder
        tgt_X, tgt_S, tgt_A, _ = target
        bind_S = self.mpn(mol2graph(mol_batch))

        B, N, M = bind_S.size(0), bind_S.size(1), tgt_X.size(1)
        bind_mask = bind_A[:,:,1].clamp(max=1).float()
        tgt_mask = tgt_A[:,:,1].clamp(max=1).float()
        mask = torch.cat([bind_mask, tgt_mask], dim=1)
        mask_2D = mask.unsqueeze(2) * mask.unsqueeze(1)

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
        X = torch.cat((bind_X[:,:,1], tgt_X[:,:,1]), dim=1)
        D = (X[:,:,None,:] - X[:,None,:,:]).norm(dim=-1)  # [B,N+M,N+M]
        mask_2D = mask_2D * (D < self.threshold).float()

        h = self.encoder((bind_X, bind_S, bind_A, None), target)  # [B,N+M,H]
        h = self.U_o(h).unsqueeze(2) + self.U_o(h).unsqueeze(1)  # [B,N,M,H]
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
        bind_X, mol_batch, bind_A, _ = binder
        tgt_X, tgt_S, tgt_A, _ = target
        bind_S = self.mpn(mol2graph(mol_batch))

        B, N, M = bind_S.size(0), bind_S.size(1), tgt_X.size(1)
        bind_mask = bind_A[:,:,1].clamp(max=1).float()
        tgt_mask = tgt_A[:,:,1].clamp(max=1).float()
        mask = torch.cat([bind_mask, tgt_mask], dim=1)
        mask_2D = mask.unsqueeze(2) * mask.unsqueeze(1)

        X = torch.cat((bind_X[:,:,1], tgt_X[:,:,1]), dim=1)
        D = (X[:,:,None,:] - X[:,None,:,:]).norm(dim=-1)  # [B,N+M,N+M]
        mask_2D = mask_2D * (D < self.threshold).float()

        h = self.encoder((bind_X, bind_S, bind_A, None), target)  # [B,N+M,H]
        h = self.U_o(h).unsqueeze(2) + self.U_o(h).unsqueeze(1)  # [B,N,M,H]
        energy = self.W_o(h).squeeze(-1)  # [B,N+M,N+M]
        energy = (energy * mask_2D).sum(dim=(1,2))  # [B]
        return energy
 

class DrugAllAtomEnergyModel(FARigidModel):

    def __init__(self, args):
        super(DrugAllAtomEnergyModel, self).__init__()
        self.mse_loss = nn.MSELoss()
        self.threshold = args.threshold
        self.args = args
        self.mpn = MPNEncoder(args)
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
        true_X, mol_batch, bind_A, _ = binder
        tgt_X, tgt_S, tgt_A, _ = target
        bind_S = self.mpn(mol2graph(mol_batch))

        B, N, M = bind_S.size(0), bind_S.size(1), tgt_X.size(1)
        bind_mask = bind_A[:,:,1].clamp(max=1).float()
        tgt_mask = tgt_A[:,:,1].clamp(max=1).float()
        bind_A = bind_A * (true_X.norm(dim=-1) > 1e-4).long() 
        tgt_A = tgt_A * (tgt_X.norm(dim=-1) > 1e-4).long() 
        sc_mask = (tgt_A[:,:,4:] > 0).float().view(B*M, 10)
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
        bind_X = bind_X.requires_grad_()

        # Random side chain rotation
        aidx = [random.randint(0, 99) for _ in range(B * M)]
        sigma = torch.tensor([self.sigma_range[i] for i in aidx]).float().cuda()
        bidx = [np.random.choice(list(range(100)), p=self.density[i]) for i in aidx]
        theta = torch.tensor([self.theta_range[i] for i in bidx]).float().cuda()
        u = torch.randn(B*M, 3).cuda() 
        hat_u = F.normalize(u, dim=-1)
        u = hat_u * theta.unsqueeze(-1)
        # Apply
        backbone = tgt_X[:,:,:4].clone()
        center = tgt_X[:,:,1:2,:].clone()  # CA is the rotation center
        tgt_X = tgt_X - center
        tgt_X = tgt_X.view(B*M, 14, 3)
        tgt_X = self.sc_rotate(tgt_X, u)
        tgt_X = tgt_X.view(B, M, 14, 3) + center
        tgt_X = torch.cat((backbone, tgt_X[:,:,4:]), dim=-2)
        tgt_X = tgt_X * (tgt_A > 0).float().unsqueeze(-1)
        tgt_X = tgt_X.requires_grad_()

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
        f_bind, f_tgt = torch.autograd.grad(energy.sum(), [bind_X, tgt_X], create_graph=True, retain_graph=True)
        t = self.mean(f_bind[:,:,1], bind_mask)

        # Backbone torque
        center = self.mean(bind_X[:,:,1], bind_mask)
        bind_X = bind_X - center[:,None,None,:]    # set rotation center to zero
        G = torch.cross(bind_X[:,:,1], f_bind[:,:,1], dim=-1)  # [B,N,3]
        G = (G * bind_mask[...,None]).sum(dim=1)   # [B,3] angular momentum
        I = self.inertia(bind_X[:,:,1], bind_mask) # [B,3,3] inertia matrix
        w = torch.linalg.solve(I.detach(), G)  # angular velocity

        # Side chain torque
        center = tgt_X[:,:,1:2,:]  # CA is the rotation center
        tgt_X = tgt_X[:,:,4:] - center   # set rotation center to zero
        tgt_X = tgt_X.view(B*M, 10, 3)
        f_tgt = f_tgt[:,:,4:].reshape(B*M, 10, 3)
        G = torch.cross(tgt_X, f_tgt, dim=-1)
        G = (G * sc_mask[...,None]).sum(dim=1)   # [B*N,3] angular momentum
        I = self.inertia(tgt_X, sc_mask)    # [B*N,3,3] inertia matrix
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
        bind_X, mol_batch, bind_A, _ = binder
        tgt_X, tgt_S, tgt_A, _ = target
        bind_S = self.mpn(mol2graph(mol_batch))

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
            return (energy * mask_2D).sum(dim=(-1, -2))
        else:
            return (energy * mask_2D).sum(dim=(1,2))  # [B]
    
    def virtual_screen(self, protein_pdb, sdf_list, batch_size=200):
        hchain = parsePDB(protein_pdb, model=1)
        _, hcoords, hseq, _, _ = get_seq_coords_and_angles(hchain)
        hcoords = hcoords.reshape((len(hseq), 14, 3))

        all_data = []
        for ligand_sdf in sdf_list:
            df = PandasTools.LoadSDF(ligand_sdf, molColName='Molecule', includeFingerprints=False)
            mol = df['Molecule'][0]
            entry = {
                "binder_mol": mol, "target_seq": hseq, "target_coords": hcoords,
            }
            all_data.append(entry)
        
        embedding = load_esm_embedding(all_data[0], ['target_seq'])
        all_data = DrugDataset.process(all_data, self.args.patch_size)
        all_score = []

        for i in range(0, len(all_data), batch_size):
            batch = all_data[i : i + batch_size]
            binder, target = DrugDataset.make_bind_batch(batch, embedding, self.args)
            score = self.predict(binder, target)
            for entry,aff in zip(batch, score):
                smiles = Chem.MolToSmiles(entry['binder_mol'])
                all_score.append((smiles, aff))
        return all_score

    def gaussian_forward(self, binder, target):
        true_X, mol_batch, bind_A, _ = binder
        tgt_X, tgt_S, tgt_A, _ = target
        bind_S = self.mpn(mol2graph(mol_batch))

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
        f_bind = torch.autograd.grad(energy.sum(), bind_X, create_graph=True, retain_graph=True)[0]
        loss = (f_bind * eps + hat_t / eps) * atom_mask
        return loss.sum() / atom_mask.sum()


class DrugDecoyModel(DrugAllAtomEnergyModel):

    def __init__(self, args):
        super(DrugDecoyModel, self).__init__(args)
        self.ce_loss = nn.CrossEntropyLoss()

    def forward(self, binder, target, num_decoy=1):
        true_X, mol_batch, bind_A, _ = binder
        tgt_X, tgt_S, tgt_A, _ = target
        bind_S = self.mpn(mol2graph(mol_batch))

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
 
