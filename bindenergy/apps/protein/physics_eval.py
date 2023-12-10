import sys
import numpy as np
import json
import scipy
import argparse
from collections import defaultdict
from tqdm import tqdm
from multiprocessing import Pool
from pyfoldx.structure import Structure
from bindenergy import *


def make_mutation(tup):
    pdb, pdb_string, mutation, true_ddg = tup
    protein = Structure(code="", from_string=pdb_string)
    pred_ddg, _, _ = protein.mutate(mutation + ';', verbose=False)
    return pdb, mutation, float(pred_ddg['total'].values[-1]), true_ddg


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', default='data/skempi/data.pkl')
    parser.add_argument('--patch_size', type=int, default=50)
    args = parser.parse_args()
 
    test_data = ProteinDataset(args.data_path, args.patch_size).data
    mut_data = [entry for entry in test_data if len(entry['pdb'][1]) > 0]

    obj_list = []
    for entry in tqdm(mut_data):
        pdb, mutation, ddg = entry['pdb']
        with open(f"ckpts/foldX/skempi/{pdb[:4]}.pdb") as f:
            pdb_string = f.readlines()
        obj_list.append((pdb, pdb_string, mutation, ddg))

    with Pool(32) as pool:
        out = pool.map(make_mutation, obj_list)
    
    all_pred = defaultdict(list)
    all_label = defaultdict(list)
    pred, label = [], []
    for pdb, mutation, pred_ddg, true_ddg in out:
        all_pred[pdb].append(pred_ddg)
        all_label[pdb].append(true_ddg)
        pred.append(pred_ddg)
        label.append(true_ddg)
        print(pdb, mutation, pred_ddg, true_ddg)

    R2 = [scipy.stats.pearsonr(all_pred[k], all_label[k])[0] for k in all_pred.keys() if len(all_pred[k]) >= 5]
    print('Pearson', scipy.stats.pearsonr(pred, label)[0], np.mean(R2))

    R2 = [scipy.stats.spearmanr(all_pred[k], all_label[k])[0] for k in all_pred.keys() if len(all_pred[k]) >= 5]
    print('Spearman', scipy.stats.spearmanr(pred, label)[0], np.mean(R2))
