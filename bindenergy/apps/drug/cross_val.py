import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler

import json
import glob
import math, random, sys
import numpy as np
import scipy
import argparse
import os

from bindenergy import *
from tqdm import tqdm, trange
from collections import defaultdict

torch.set_num_threads(8)

def pdbbind_evaluate(model, data, args):
    model.eval()
    score = []
    label = []
    with torch.no_grad():
        for entry in tqdm(data):
            binder, target = PocketDataset.make_bind_batch([entry])
            pred = model.predict(binder, target)
            score.append(pred.item())
            label.append(-1.0 * entry['affinity'])
    return scipy.stats.pearsonr(score, label)[0]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--val_path', default='data/drug/pocket/diffdock_equibind.pkl')
    parser.add_argument('--test_path', default='data/drug/pocket/diffdock_casf16.pkl')
    parser.add_argument('--save_dir', default='ckpts/tmp')
    parser.add_argument('--epoch', type=int, default=5)
    args = parser.parse_args()

    val_data = PocketDataset(args.val_path)
    test_data = PocketDataset(args.test_path)
    test_smiles = set([Chem.MolToSmiles(entry['binder_mol']) for entry in test_data.data])
    val_data.data = [entry for entry in val_data.data if Chem.MolToSmiles(entry['binder_mol']) not in test_smiles]

    best_corr = 0
    for fn in glob.glob(f"{args.save_dir}*/model.ckpt.*"):
        if int(fn.split('.')[-1]) < args.epoch:
            model_ckpt, _, model_args = torch.load(fn)
            model = DrugEnergyModel(model_args).cuda()
            model.load_state_dict(model_ckpt)
            val_corr = pdbbind_evaluate(model, val_data.data, args)
            if val_corr > best_corr:
                test_corr = pdbbind_evaluate(model, test_data.data, args)
                best_corr = val_corr
                print(fn, val_corr, test_corr)

    print('Final test R:', test_corr)
