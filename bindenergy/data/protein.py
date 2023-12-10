import torch
import torch.nn.functional as F
import numpy as np
import pickle
import random

from tqdm import tqdm
from copy import deepcopy
from bindenergy.data.loader import *
from bindenergy.data.constants import *
from bindenergy.utils.utils import cross_square_dist, compute_dihedrals, kabsch, rigid_transform


class ProteinDataset():

    def __init__(self, data_path, patch_size):
        with open(data_path, 'rb') as f:
            data = pickle.load(f)
            
        self.data = []
        for pdb, aseq, acoords, bseq, bcoords in tqdm(data):
            if len(aseq) == len(acoords) and len(bseq) == len(bcoords):
                entry = {'pdb': pdb}
                entry['binder_seq'] = bseq
                entry['binder_coords'] = torch.tensor(bcoords).float()
                entry['binder_dihedrals'] = torch.zeros(len(bseq), 6)
                entry['binder_atypes'] = torch.tensor(
                        [[ATOM_TYPES.index(a) for a in RES_ATOM14[ALPHABET.index(s)]] for s in entry['binder_seq']]
                )
                entry['target_seq'] = aseq
                entry['target_coords'] = torch.tensor(acoords).float()
                entry['target_dihedrals'] = torch.zeros(len(aseq), 6)
                entry['target_atypes'] = torch.tensor(
                        [[ATOM_TYPES.index(a) for a in RES_ATOM14[ALPHABET.index(s)]] for s in entry['target_seq']]
                )
                dist = entry['target_coords'][:, 1] - entry['binder_coords'][:, 1].mean(dim=0, keepdims=True)
                entry['target_idx'] = dist.norm(dim=-1).sort().indices[:patch_size].sort().values
                dist = entry['binder_coords'][:, 1] - entry['target_coords'][:, 1].mean(dim=0, keepdims=True)
                entry['binder_idx'] = dist.norm(dim=-1).sort().indices[:patch_size].sort().values
                ProteinDataset.make_site(entry, 'target')
                ProteinDataset.make_site(entry, 'binder')
                self.data.append(entry)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    @staticmethod
    def make_site(entry, field):
        idx = entry[f'{field}_idx']
        entry[f'{field}_full'] = entry[f'{field}_seq']
        entry[f'{field}_seq'] = ''.join([entry[f'{field}_seq'][i] for i in idx.tolist()])
        entry[f'{field}_coords'] = entry[f'{field}_coords'][idx]
        entry[f'{field}_atypes'] = entry[f'{field}_atypes'][idx]
        entry[f'{field}_dihedrals'] = entry[f'{field}_dihedrals'][idx]

    @staticmethod
    def make_local_batch(batch, embedding, args, binder, target):
        bind_X, bind_S, bind_A, _ = featurize(batch, binder)
        tgt_X, tgt_S, tgt_A, _ = featurize(batch, target)
        bind_S = torch.zeros(bind_S.size(0), bind_S.size(1), args.esm_size).cuda()
        tgt_S = torch.zeros(tgt_S.size(0), tgt_S.size(1), args.esm_size).cuda()
        for i,b in enumerate(batch):
            idx = b[f'{target}_idx']
            tgt_S[i,:len(idx)] = embedding[b[f'{target}_full']][idx]
            idx = b[f'{binder}_idx']
            bind_S[i,:len(idx)] = embedding[b[f'{binder}_full']][idx]
        return (bind_X, bind_S, bind_A, None), (tgt_X, tgt_S, tgt_A, None)


class PeptideDataset():

    def __init__(self, data_path, patch_size):
        with open(data_path, 'rb') as f:
            data = pickle.load(f)

        self.data = []
        for entry in tqdm(data):
            entry['binder_coords'] = torch.tensor(entry['binder_coords']).float()
            entry['binder_dihedrals'] = torch.zeros(len(entry['binder_seq']), 6)
            entry['binder_atypes'] = torch.tensor(
                    [[ATOM_TYPES.index(a) for a in RES_ATOM14[ALPHABET.index(s)]] for s in entry['binder_seq']]
            )
            entry['target_coords'] = torch.tensor(entry['target_coords']).float()
            entry['target_dihedrals'] = torch.zeros(len(entry['target_seq']), 6)
            entry['target_atypes'] = torch.tensor(
                    [[ATOM_TYPES.index(a) for a in RES_ATOM14[ALPHABET.index(s)]] for s in entry['target_seq']]
            )
            dist = entry['target_coords'][:, 1] - entry['binder_coords'][:, 1].mean(dim=0, keepdims=True)
            entry['target_idx'] = idx = dist.norm(dim=-1).sort().indices[:patch_size].sort().values
            entry['receptor_seq'] = entry['target_seq']
            entry['target_seq'] = ''.join([entry['target_seq'][i] for i in idx.tolist()])
            entry['target_coords'] = entry['target_coords'][idx]
            entry['target_atypes'] = entry['target_atypes'][idx]
            entry['target_dihedrals'] = entry['target_dihedrals'][idx]
            self.data.append(entry)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    @staticmethod
    def make_local_batch(batch, embedding, args):
        bind_X, bind_S, bind_A, _ = featurize(batch, 'binder')
        tgt_X, tgt_S, tgt_A, _ = featurize(batch, 'target')
        bind_S = torch.zeros(bind_S.size(0), bind_S.size(1), args.esm_size).cuda()
        tgt_S = torch.zeros(tgt_S.size(0), tgt_S.size(1), args.esm_size).cuda()
        for i,b in enumerate(batch):
            idx = b['target_idx']
            tgt_S[i,:len(idx)] = embedding[b['receptor_seq']][idx]
            L = len(b['binder_seq'])
            bind_S[i,:L] = embedding[b['binder_seq']]
        return (bind_X, bind_S, bind_A, None), (tgt_X, tgt_S, tgt_A, None)
