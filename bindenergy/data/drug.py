import torch
import torch.nn.functional as F
import numpy as np
import pickle
import random

from tqdm import tqdm
from copy import deepcopy
from bindenergy.data.constants import *
from bindenergy.data.loader import *
from rdkit import Chem


class DrugDataset():

    def __init__(self, data_path, patch_size):
        with open(data_path, 'rb') as f:
            data = pickle.load(f)
            
        self.data = []
        for entry in tqdm(data):
            entry['target_coords'] = torch.tensor(entry['target_coords']).float()
            entry['target_dihedrals'] = torch.zeros(len(entry['target_seq']), 6)
            entry['target_atypes'] = torch.tensor(
                [[ATOM_TYPES.index(a) for a in RES_ATOM14[ALPHABET.index(s)]] for s in entry['target_seq']]
            )
            mol = entry['binder_mol']
            conf = mol.GetConformer()
            coords = [conf.GetAtomPosition(i) for i,atom in enumerate(mol.GetAtoms())]
            entry['binder_coords'] = torch.tensor([[p.x, p.y, p.z] for p in coords]).float()
            # make pocket
            dist = entry['target_coords'][:, 1] - entry['binder_coords'].mean(dim=0, keepdims=True)
            entry['pocket_idx'] = idx = dist.norm(dim=-1).sort().indices[:patch_size].sort().values
            entry['pocket_seq'] = ''.join([entry['target_seq'][i] for i in idx.tolist()])
            entry['pocket_coords'] = entry['target_coords'][idx]
            entry['pocket_atypes'] = entry['target_atypes'][idx]
            entry['pocket_dihedrals'] = entry['target_dihedrals'][idx]
            self.data.append(entry)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    @staticmethod
    def process(raw_data, patch_size):
        data = []
        for entry in tqdm(raw_data):
            entry['target_coords'] = torch.tensor(entry['target_coords']).float()
            entry['target_dihedrals'] = torch.zeros(len(entry['target_seq']), 6)
            entry['target_atypes'] = torch.tensor(
                [[ATOM_TYPES.index(a) for a in RES_ATOM14[ALPHABET.index(s)]] for s in entry['target_seq']]
            )
            mol = entry['binder_mol']
            conf = mol.GetConformer()
            coords = [conf.GetAtomPosition(i) for i,atom in enumerate(mol.GetAtoms())]
            entry['binder_coords'] = torch.tensor([[p.x, p.y, p.z] for p in coords]).float()
            # make pocket
            dist = entry['target_coords'][:, 1] - entry['binder_coords'].mean(dim=0, keepdims=True)
            entry['pocket_idx'] = idx = dist.norm(dim=-1).sort().indices[:patch_size].sort().values
            entry['pocket_seq'] = ''.join([entry['target_seq'][i] for i in idx.tolist()])
            entry['pocket_coords'] = entry['target_coords'][idx]
            entry['pocket_atypes'] = entry['target_atypes'][idx]
            entry['pocket_dihedrals'] = entry['target_dihedrals'][idx]
            data.append(entry)
        return data

    @staticmethod
    def make_bind_batch(batch, embedding, args):
        mols = [entry['binder_mol'] for entry in batch]
        N = max([mol.GetNumAtoms() for mol in mols])
        bind_X = torch.zeros(len(batch), N, 14, 3).cuda()
        bind_A = torch.zeros(len(batch), N, 14).cuda().long()
        tgt_X, tgt_S, tgt_A, tgt_V = featurize(batch, 'pocket')
        tgt_S = torch.zeros(tgt_S.size(0), tgt_S.size(1), args.esm_size).cuda()
        for i,b in enumerate(batch):
            L = b['binder_mol'].GetNumAtoms()
            bind_X[i,:L,1,:] = b['binder_coords']
            bind_A[i,:L,1] = 1
            L = len(b['pocket_seq'])
            tgt_S[i,:L] = embedding[b['target_seq']][b['pocket_idx']]
        return (bind_X, mols, bind_A, None), (tgt_X, tgt_S, tgt_A, tgt_V)


class PocketDataset():

    def __init__(self, data_path):
        self.data = []
        with open(data_path, 'rb') as f:
            data = pickle.load(f)

        for entry in tqdm(data):
            pocket = entry['target_site']
            pocket = pocket[np.isin(pocket.atom_name, ATOM_TYPES)]
            entry['target_coords'] = torch.tensor(pocket.coord).float()
            entry['target_atypes'] = torch.tensor([
                    ATOM_TYPES.index(a) for a in pocket.atom_name
            ])
            mol = entry['binder_mol']
            conf = mol.GetConformer()
            coords = [conf.GetAtomPosition(i) for i,atom in enumerate(mol.GetAtoms())]
            entry['binder_coords'] = torch.tensor([[p.x, p.y, p.z] for p in coords]).float()
            self.data.append(entry)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    @staticmethod
    def make_bind_batch(batch):
        mols = [entry['binder_mol'] for entry in batch]
        N = max([mol.GetNumAtoms() for mol in mols])
        M = max([len(entry['target_site']) for entry in batch])
        bind_X = torch.zeros(len(batch), N, 14, 3).cuda()
        bind_A = torch.zeros(len(batch), N, 14).cuda()
        tgt_X = torch.zeros(len(batch), M, 14, 3).cuda()
        tgt_S = torch.zeros(len(batch), M).cuda().long()
        tgt_A = torch.zeros(len(batch), M, 14).cuda()
        for i,b in enumerate(batch):
            L = b['binder_mol'].GetNumAtoms()
            bind_X[i,:L,1,:] = b['binder_coords']
            bind_A[i,:L,1] = 1
            L = len(b['target_atypes'])
            tgt_X[i,:L,1,:] = b['target_coords']
            tgt_S[i,:L] = b['target_atypes']
            tgt_A[i,:L,1] = 1
        return (bind_X, mols, bind_A, None), (tgt_X, tgt_S, tgt_A, None)
