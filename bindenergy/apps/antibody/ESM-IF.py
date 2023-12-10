# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
# Scores sequences based on a given structure.
#
# usage:
# score_log_likelihoods.py [-h] [--outpath OUTPATH] [--chain CHAIN] pdbfile seqfile

import argparse
import numpy as np
import torch
import scipy
import torch.nn as nn
import torch.nn.functional as F

import esm
import esm.inverse_folding

from bindenergy import *
from bindenergy.apps.antibody.semi_train import build_data
from tqdm import tqdm, trange
from sklearn.metrics import roc_auc_score

torch.set_num_threads(8)

def single_eval(model, args):
    reference = AntibodyDataset(args.ref_path, args.cdr, args.epitope_size).data[0]
    data = build_data(args.data_path, reference)

    ref_pdb = reference['pdb']
    X = reference['antibody_coords'].cpu().numpy()
    Y = reference['antigen_coords'].cpu().numpy()

    array2 = print_pdb(Y, reference['antigen_seq'], 'A')
    array1 = print_pdb(X, reference['antibody_seq'], 'B')
    array = struc.array(array2 + array1)
    save_structure(f'{ref_pdb}.pdb', array)

    structure = esm.inverse_folding.util.load_structure(f'{ref_pdb}.pdb')
    native_coords, native_seqs = esm.inverse_folding.multichain_util.extract_coords_from_complex(structure)

    pred, label = [], []
    for ab in tqdm(data):
        coords = native_coords if args.native else {
            "A": native_coords["A"],
            "B": ab['antibody_coords'][:, :3, :].numpy()  # N, CA, C
        }
        ll, _ = esm.inverse_folding.multichain_util.score_sequence_in_complex(
                model, alphabet, coords, 'B', ab['antibody_seq']
        )
        pred.append(ll.item())
        label.append(ab['affinity'])

    print('Corr =', scipy.stats.spearmanr(pred, label)[0])


def multi_eval(model, args):
    data = AntibodyDataset(args.data_path, args.cdr, args.epitope_size).data
    pred, label = [], []
    for entry in tqdm(data):
        ref_pdb = entry['pdb']
        X = entry['antibody_coords'].cpu().numpy()
        Y = entry['antigen_coords'].cpu().numpy()

        array1 = print_pdb(X, entry['antibody_seq'], 'B')
        array2 = print_pdb(Y, entry['antigen_seq'], 'A')
        array = struc.array(array2 + array1)
        save_structure(f'{ref_pdb}.pdb', array)

        structure = esm.inverse_folding.util.load_structure(f'{ref_pdb}.pdb')
        native_coords, native_seqs = esm.inverse_folding.multichain_util.extract_coords_from_complex(structure)

        coords = native_coords if args.native else {
            "A": native_coords["A"],
            "B": entry['antibody_coords'][:, :3, :].numpy()  # N, CA, C
        }
        ll, _ = esm.inverse_folding.multichain_util.score_sequence_in_complex(
                model, alphabet, coords, 'B', entry['antibody_seq']
        )
        pred.append(ll.item())
        label.append(entry['affinity'])

    print('Corr =', scipy.stats.spearmanr(pred, label)[0])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', required=True)
    parser.add_argument('--ref_path', default=None)
    parser.add_argument('--cdr', default='123456')
    parser.add_argument('--seed', type=int, default=7)
    parser.add_argument('--epitope_size', type=int, default=50)
    parser.add_argument("--native", action="store_true", default=False)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    model, alphabet = esm.pretrained.esm_if1_gvp4_t16_142M_UR50()
    model = model.cuda()
    model = model.eval()

    if args.ref_path is None:
        multi_eval(model, args)
    else:
        single_eval(model, args)