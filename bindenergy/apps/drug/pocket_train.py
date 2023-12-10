import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler

import json
import csv
import math, random, sys
import numpy as np
import scipy
import argparse
import os

from bindenergy import *
from tqdm import tqdm, trange
from collections import defaultdict

torch.set_num_threads(8)


def fep_evaluate(model, data, args):
    model.eval()
    score = defaultdict(list)
    label = defaultdict(list)
    with torch.no_grad():
        for entry in tqdm(data):
            pdb = entry['pdb']
            binder, target = PocketDataset.make_bind_batch([entry])
            pred = model.predict(binder, target)
            score[pdb].append(pred.item())
            label[pdb].append(-1.0 * entry['affinity'])
    R2 = [scipy.stats.spearmanr(score[pdb], label[pdb])[0] for pdb in score.keys()]
    return np.mean(R2)


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


def build_model(args):
    if args.dist:
        return DrugDistModel(args).cuda()
    elif args.decoy:
        return DrugDecoyModel(args).cuda()
    else:
        return DrugEnergyModel(args).cuda()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_path', default='data/drug/pocket/refine.pkl')
    parser.add_argument('--val_path', default='data/drug/pocket/test_equibind.pkl')
    parser.add_argument('--test_path', default='data/drug/pocket/test_casf16.pkl')
    parser.add_argument('--save_dir', default='ckpts/tmp')
    parser.add_argument('--load_model', default=None)

    parser.add_argument('--dist', action='store_true', default=False)
    parser.add_argument('--decoy', action='store_true', default=False)

    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--hidden_size', type=int, default=256)
    parser.add_argument('--esm_size', type=int, default=2560)
    parser.add_argument('--depth', type=int, default=2)
    parser.add_argument('--mpn_depth', type=int, default=3)
    parser.add_argument('--vocab_size', type=int, default=len(ATOM_TYPES))
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--threshold', type=float, default=10.0)

    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--seed', type=int, default=7)
    parser.add_argument('--anneal_rate', type=float, default=0.95)
    parser.add_argument('--clip_norm', type=float, default=1.0)

    args = parser.parse_args()
    args.bert_size = args.hidden_size
    print(args)

    os.makedirs(args.save_dir, exist_ok=True)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    train_data = PocketDataset(args.train_path)
    val_data = PocketDataset(args.val_path)
    test_data = PocketDataset(args.test_path)
    print('Training/Val data:', len(train_data), len(val_data))

    test_smiles = set([Chem.MolToSmiles(entry['binder_mol']) for entry in test_data.data])
    train_data.data = [entry for entry in train_data.data if Chem.MolToSmiles(entry['binder_mol']) not in test_smiles]
    val_data.data = [entry for entry in val_data.data if Chem.MolToSmiles(entry['binder_mol']) not in test_smiles]
    print('After filtering:', len(train_data), len(val_data))

    model = build_model(args)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = lr_scheduler.ExponentialLR(optimizer, args.anneal_rate)
    if args.load_model:
        model_ckpt, opt_ckpt, model_args = torch.load(args.load_model)
        model = build_model(args)
        model.load_state_dict(model_ckpt)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        optimizer.load_state_dict(opt_ckpt)
        scheduler = lr_scheduler.ExponentialLR(optimizer, args.anneal_rate)

    best_corr = 0
    for e in range(args.epochs):
        model.train()
        random.shuffle(train_data.data)
        for i in trange(0, len(train_data.data), args.batch_size):
            optimizer.zero_grad()
            batch = train_data.data[i : i + args.batch_size]
            binder, target = PocketDataset.make_bind_batch(batch)
            loss = model(binder, target)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), args.clip_norm)
            optimizer.step()

        val_corr = pdbbind_evaluate(model, val_data.data, args)
        ckpt = (model.state_dict(), optimizer.state_dict(), args)
        torch.save(ckpt, os.path.join(args.save_dir, f"model.ckpt.{e}"))
        print(f'Epoch {e}, Corr = {val_corr:.4f}')

        scheduler.step()
        if val_corr > best_corr:
            best_corr = val_corr
            torch.save(ckpt, os.path.join(args.save_dir, f"model.best"))

    if best_corr > 0:
        best_ckpt = os.path.join(args.save_dir, f"model.best")
        model.load_state_dict(torch.load(best_ckpt)[0])

    test_corr = pdbbind_evaluate(model, test_data.data, args)
    print(f'Test Corr = {test_corr:.4f}')
