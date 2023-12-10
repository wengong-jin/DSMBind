import torch
import torch.nn.functional as F
import numpy as np
import json 
import random
from copy import deepcopy
from tqdm import tqdm

from bindenergy.data.constants import *
from bindenergy.data.loader import *
from bindenergy.utils.utils import cross_square_dist, compute_dihedrals


class AntibodyDataset():

    def __init__(self, data_path, cdr_type, epitope_size):
        self.data = []
        self.ab_map = {}
        self.epitope_size = epitope_size

        with open(data_path) as f:
            for line in tqdm(f.readlines()):
                entry = json.loads(line)
                entry['antibody_coords'] = torch.tensor(entry['antibody_coords'])
                entry['antibody_dihedrals'] = compute_dihedrals(entry['antibody_coords'][None,...])[0]
                entry['antibody_atypes'] = torch.tensor(
                        [[ATOM_TYPES.index(a) for a in RES_ATOM14[ALPHABET.index(s)]] for s in entry['antibody_seq']]
                )

                entry['antigen_coords'] = torch.tensor(entry['antigen_coords'])
                entry['antigen_dihedrals'] = compute_dihedrals(entry['antigen_coords'][None,...])[0]
                entry['antigen_atypes'] = torch.tensor(
                        [[ATOM_TYPES.index(a) for a in RES_ATOM14[ALPHABET.index(s)]] for s in entry['antigen_seq']]
                )

                entry['paratope_idx'] = idx = [i for i,v in enumerate(entry['antibody_cdr']) if v in cdr_type]
                entry['paratope_cdr'] = torch.tensor([int(entry['antibody_cdr'][i]) for i in idx])
                entry['paratope_seq'] = ''.join([entry['antibody_seq'][i] for i in idx])
                entry['paratope_coords'] = entry['antibody_coords'][idx]
                entry['paratope_atypes'] = entry['antibody_atypes'][idx]
                entry['paratope_dihedrals'] = entry['antibody_dihedrals'][idx]

                entry['scaffold_idx'] = idx = [i for i,v in enumerate(entry['antibody_cdr']) if v not in '0' + cdr_type]
                entry['scaffold_seq'] = ''.join([entry['antibody_seq'][i] for i in idx])
                entry['scaffold_coords'] = entry['antibody_coords'][idx]
                entry['scaffold_atypes'] = entry['antibody_atypes'][idx]
                entry['scaffold_dihedrals'] = entry['antibody_dihedrals'][idx]

                dist, _ = cross_square_dist(
                        entry['antigen_coords'][None,...],
                        entry['paratope_coords'][None,...],
                        entry['antigen_atypes'][None,:,1],
                        entry['paratope_atypes'][None,:,1],
                )
                entry['epitope_dist'] = dist = dist[0].sqrt().mean(dim=-1)
                entry['epitope_center'] = center = dist.argmin().item()
                AntibodyDataset.make_epitope(entry, [center], epitope_size)

                entry['target_seq'] = entry['scaffold_seq'] + entry['epitope_seq']
                entry['target_coords'] = torch.cat((entry['scaffold_coords'], entry['epitope_coords']), dim=0)
                entry['target_atypes'] = torch.cat((entry['scaffold_atypes'], entry['epitope_atypes']), dim=0)
                entry['target_dihedrals'] = torch.cat((entry['scaffold_dihedrals'], entry['epitope_dihedrals']), dim=0)
                self.ab_map[entry['antibody_seq']] = entry
                self.data.append(entry)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    @staticmethod
    def make_epitope(entry, center, epitope_size):
        dist, _ = cross_square_dist(
                entry['antigen_coords'][None,...],
                entry['antigen_coords'][None,center,...],
                entry['antigen_atypes'][None,:,1],
                entry['antigen_atypes'][None,center,1],
        )
        dist = dist[0].amin(dim=-1)
        K = min(len(dist), epitope_size)
        idx = dist.topk(k=K, largest=False).indices.sort().values

        entry['epitope_idx'] = idx
        entry['epitope_seq'] = ''.join([entry['antigen_seq'][i] for i in idx.tolist()])
        entry['epitope_coords'] = entry['antigen_coords'][idx]
        entry['epitope_atypes'] = entry['antigen_atypes'][idx]
        entry['epitope_dihedrals'] = entry['antigen_dihedrals'][idx]
        return entry

    @staticmethod
    def make_local_batch(batch, embedding, args):
        cdr_X, cdr_S, cdr_A, cdr_V = featurize(batch, 'paratope')
        tgt_X, tgt_S, tgt_A, tgt_V = featurize(batch, 'epitope')
        cdr_S = torch.zeros(cdr_S.size(0), cdr_S.size(1), args.bert_size).cuda()
        tgt_S = torch.zeros(tgt_S.size(0), tgt_S.size(1), args.esm_size).cuda()
        for i,b in enumerate(batch):
            L = len(b['epitope_seq'])
            tgt_S[i,:L] = embedding[b['antigen_seq']][b['epitope_idx']]
            L = len(b['paratope_seq'])
            cdr_S[i,:L] = embedding[b['antibody_seq']][b['paratope_idx']]
        return (cdr_X, cdr_S, cdr_A, cdr_V), (tgt_X, tgt_S, tgt_A, tgt_V)

    @staticmethod
    def make_contact_batch(batch, embedding, args):
        cdr_X, cdr_S, cdr_A, cdr_V = featurize(batch, 'paratope')
        tgt_X, tgt_S, tgt_A, tgt_V = featurize(batch, 'antigen')
        labels = torch.zeros_like(tgt_S).float()
        cdr_S = torch.zeros(cdr_S.size(0), cdr_S.size(1), args.bert_size).cuda()
        tgt_S = torch.zeros(tgt_S.size(0), tgt_S.size(1), args.esm_size).cuda()
        for i,b in enumerate(batch):
            L = len(b['antigen_seq'])
            tgt_S[i,:L] = embedding[b['antigen_seq']]
            th = b['epitope_dist'][b['epitope_center']]
            labels[i,:L] = ((b['epitope_dist'] - th) < args.tolerance).float()
            L = len(b['paratope_seq'])
            cdr_S[i,:L] = embedding[b['antibody_seq']][b['paratope_idx']]
        return (cdr_X, cdr_S, cdr_A, cdr_V), (tgt_X, tgt_S, tgt_A, tgt_V), labels

    @staticmethod
    def make_global_batch(batch, embedding, args):
        cdr_X, cdr_S, cdr_A, cdr_V = featurize(batch, 'paratope')
        tgt_X, tgt_S, tgt_A, tgt_V = featurize(batch, 'antigen')
        labels = torch.zeros_like(tgt_S).float()
        cdr_S = torch.zeros(cdr_S.size(0), cdr_S.size(1), args.bert_size).cuda()
        tgt_S = torch.zeros(tgt_S.size(0), tgt_S.size(1), args.esm_size).cuda()
        for i,b in enumerate(batch):
            L = len(b['antigen_seq'])
            tgt_S[i,:L] = embedding[b['antigen_seq']]
            labels[i,b['epitope_idx']] = 1
            L = len(b['paratope_seq'])
            cdr_S[i,:L] = embedding[b['antibody_seq']][b['paratope_idx']]
        return (cdr_X, cdr_S, cdr_A, cdr_V), (tgt_X, tgt_S, tgt_A, tgt_V), labels

    @staticmethod
    def make_decoy_batch(batch, K=0):
        new_batch = []
        for ab in batch:
            new_batch.append(ab)
            for _ in range(K):
                ab = {k:v for k,v in ab.items()}  # shallow copy
                ab['paratope_seq'] = ''.join([random.choice(ALPHABET[1:]) for _ in ab['paratope_seq']])
                ab['paratope_atypes'] = torch.tensor(
                        [[ATOM_TYPES.index(a) for a in RES_ATOM14[ALPHABET.index(s)]] for s in ab['paratope_seq']]
                )
                new_batch.append(ab)
        cdr_X, cdr_aa, cdr_A, _ = featurize(new_batch, 'paratope')
        tgt_X, tgt_aa, tgt_A, _ = featurize(new_batch, 'epitope')
        cdr_S = F.one_hot(cdr_aa, num_classes=len(ALPHABET)).float()
        tgt_S = F.one_hot(tgt_aa, num_classes=len(ALPHABET)).float()
        cdr_S[:, :, 0] = 0
        tgt_S[:, :, 0] = 0
        return (cdr_X, cdr_S, cdr_A, cdr_aa), (tgt_X, tgt_S, tgt_A, tgt_aa)

