import torch
import torch.nn as nn
import torch.optim as optim

import json
import csv
import math, random, sys
import numpy as np
import argparse
import os
import glob

from bindenergy import *
from tqdm import tqdm, trange
from sklearn.metrics import roc_auc_score

torch.set_num_threads(8)
natom = {aa : len([t for t in RES_ATOM14[i] if t != '']) for i,aa in enumerate(ALPHABET)}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_path', required=True)
    parser.add_argument('--ref_path', required=True)
    parser.add_argument('--dsmbind_model', required=True)
    parser.add_argument('--cdr1_model', required=True)
    parser.add_argument('--cdr2_model', required=True)
    parser.add_argument('--cdr3_model', required=True)
    parser.add_argument('--sc_model', required=True)
    parser.add_argument('--cdr', default='123')
    parser.add_argument('--epitope_size', type=int, default=50)
    parser.add_argument('--seed', type=int, default=7)
    args = parser.parse_args()

    sc_ckpt, _, sc_args = torch.load(args.sc_model)
    sc_docker = SideChainDocker(sc_args).cuda()
    sc_docker.load_state_dict(sc_ckpt)
    sc_docker.eval()

    cdr1_ckpt, _, cdr1_args = torch.load(args.cdr1_model)
    cdr1_docker = FADocker(cdr1_args).cuda()
    cdr1_docker.load_state_dict(cdr1_ckpt)
    cdr1_docker.eval()

    cdr2_ckpt, _, cdr2_args = torch.load(args.cdr2_model)
    cdr2_docker = FADocker(cdr2_args).cuda()
    cdr2_docker.load_state_dict(cdr2_ckpt)
    cdr2_docker.eval()

    cdr3_ckpt, _, cdr3_args = torch.load(args.cdr3_model)
    cdr3_docker = FADocker(cdr3_args).cuda()
    cdr3_docker.load_state_dict(cdr3_ckpt)
    cdr3_docker.eval()

    model_ckpt, _, model_args = torch.load(args.dsmbind_model)
    model = AllAtomEnergyModel(model_args).cuda()
    model.load_state_dict(model_ckpt)
    model.eval()

    esm_model, alphabet = esm.pretrained.esm2_t36_3B_UR50D()
    batch_converter = alphabet.get_batch_converter()
    esm_model = esm_model.cuda()
    esm_model.eval()

    reference = AntibodyDataset(args.ref_path, args.cdr, args.epitope_size).data[0]

    # Antigen Embedding
    seq = reference['antigen_seq']
    batch_labels, batch_strs, batch_tokens = batch_converter([(seq, seq)])
    batch_tokens = batch_tokens.cuda()
    results = esm_model(batch_tokens, repr_layers=[36], return_contacts=False)
    embedding = {
        seq : results["representations"][36][0, 1:-1]
    }

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    with open(args.test_path) as f:
        test_data = [json.loads(line) for line in f]

    preds, labels = [], []
    for idx, entry in enumerate(tqdm(test_data)):
        ab = {k:v for k,v in reference.items()}  # shallow copy
        ab['antibody_seq'] = entry['antibody_seq']
        ab['label'] = entry['label']
        old_cdr = ab['antibody_cdr']
        new_cdr = entry['antibody_cdr']
        for i in '123':
            l, r = old_cdr.index(i), old_cdr.rindex(i)
            n = new_cdr.count(i)
            ab[f'cdr{i}_coords'] = ab['antibody_coords'][l:r+1].mean(dim=0).repeat(n, 1, 1)
            l, r = new_cdr.index(i), new_cdr.rindex(i)
            ab[f'cdr{i}_seq'] = ab['antibody_seq'][l:r+1]
            ab[f'cdr{i}_atypes'] = torch.tensor(
                    [[ATOM_TYPES.index(a) for a in RES_ATOM14[ALPHABET.index(s)]] for s in ab[f'cdr{i}_seq']]
            )
            ab[f'cdr{i}_dihedrals'] = torch.zeros(n, 6)

        with torch.no_grad():
            binder, target = AntibodyDataset.make_batch_noembed([ab], 'cdr1', 'scaffold')
            cdr1_X = cdr1_docker.predict(binder, target)
            binder, target = AntibodyDataset.make_batch_noembed([ab], 'cdr2', 'scaffold')
            cdr2_X = cdr2_docker.predict(binder, target)
            binder, target = AntibodyDataset.make_batch_noembed([ab], 'cdr3', 'scaffold')
            cdr3_X = cdr3_docker.predict(binder, target)

        ab['antibody_cdr'] = new_cdr
        ab['paratope_idx'] = [i for i,v in enumerate(ab['antibody_cdr']) if v in '123']
        ab['paratope_seq'] = ''.join([ab['antibody_seq'][i] for i in ab['paratope_idx']])
        ab['paratope_atypes'] = torch.tensor(
                [[ATOM_TYPES.index(a) for a in RES_ATOM14[ALPHABET.index(s)]] for s in ab['paratope_seq']]
        )
        ab['paratope_coords']= torch.cat([cdr1_X[0], cdr2_X[0], cdr3_X[0]], dim=0)
        ab['paratope_dihedrals']= torch.zeros(len(ab['paratope_seq']), 6)

        with torch.no_grad():
            binder, target = AntibodyDataset.make_batch_noembed([ab], 'paratope', 'epitope')
            bind_X = sc_docker.predict(binder, target)
            ab['paratope_coords'] = bind_X[0].cpu().detach()
            # ESM Embedding
            batch_labels, batch_strs, batch_tokens = batch_converter([(ab['antibody_seq'], ab['antibody_seq'])])
            batch_tokens = batch_tokens.cuda()
            results = esm_model(batch_tokens, repr_layers=[36], return_contacts=False)
            embedding[ab['antibody_seq']] = results["representations"][36][0, 1:-1]
            # DSMBind score
            binder, target = AntibodyDataset.make_local_batch([ab], embedding, model_args)
            n = sum([natom[aa] for aa,cc in zip(ab['antibody_seq'], ab['antibody_cdr']) if cc in '123'])
            score = model.predict(binder, target) / n
            preds.append(score.item())
            labels.append(ab['label'])

    print(roc_auc_score(labels, preds))