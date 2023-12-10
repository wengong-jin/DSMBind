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
import pickle

from bindenergy import *
from tqdm import tqdm, trange
from collections import defaultdict

torch.set_num_threads(8)

def evaluate(model, data, embedding, args):
    model.eval()
    pred, label = [], []
    for ab in tqdm(data):
        binder, target = AntibodyDataset.make_local_batch([ab], embedding, args)
        score = model.predict(binder, target)
        pred.append(score.item())
        label.append(-1.0 * ab['affinity'])
    return scipy.stats.pearsonr(pred, label)[0]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--val_path', default='data/sabdab/affinity/csm_data.jsonl')
    parser.add_argument('--test_path', default='data/sabdab/affinity/sabdab_data.jsonl')
    parser.add_argument('--save_dir', default='ckpts/tmp')
    parser.add_argument('--embedding', default=None)
    parser.add_argument('--cdr', default='123456')
    parser.add_argument('--epitope_size', type=int, default=50)
    args = parser.parse_args()

    val_data = AntibodyDataset(args.val_path, args.cdr, args.epitope_size)
    test_data = AntibodyDataset(args.test_path, args.cdr, args.epitope_size)
    embedding = torch.load(args.embedding)

    test_seq = {entry['paratope_seq'] for entry in test_data.data}
    val_data = [entry for entry in val_data.data if entry['paratope_seq'] not in test_seq]

    best_corr = 0
    for fn in glob.glob(f"{args.save_dir}*/model.ckpt.*"):
        model_ckpt, opt_ckpt, model_args = torch.load(fn)
        model = FAEnergyModel(model_args).cuda()
        model.load_state_dict(model_ckpt)
        val_corr = evaluate(model, val_data, embedding, model_args)
        print(f'{fn}: Corr = {val_corr:.4f}')
        if val_corr > best_corr:
            best_corr = val_corr
            test_corr = evaluate(model, test_data.data, embedding, model_args)
            print(f'Test Corr = {test_corr:.4f}')
