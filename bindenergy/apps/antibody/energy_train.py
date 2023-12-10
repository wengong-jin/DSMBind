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


def ddg_evaluate(model, data, embedding, args):
    model.eval()
    all_pred = defaultdict(list)
    all_label = defaultdict(list)
    for ab in tqdm(data):
        binder, target = AntibodyDataset.make_local_batch([ab], embedding, args)
        score = model.predict(binder, target)
        all_pred[ab['pdb']].append(score.item())
        all_label[ab['pdb']].append(-1.0 * ab['affinity'])
    R2 = [scipy.stats.pearsonr(all_pred[k], all_label[k])[0] for k in all_pred.keys() if len(all_pred[k]) > 5]
    return np.mean(R2)


def build_model(args):
    if args.all_atom:
        return AllAtomEnergyModel(args).cuda()
    elif args.dist:
        return FADistModel(args).cuda()
    elif args.decoy:
        return FADecoyModel(args).cuda()
    else:
        return FAEnergyModel(args).cuda()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_path', default='data/sabdab/affinity/train_data.jsonl')
    parser.add_argument('--val_path', default='data/sabdab/affinity/csm_data.jsonl')
    parser.add_argument('--test_path', default='data/sabdab/affinity/sabdab_data.jsonl')
    parser.add_argument('--save_dir', default='ckpts/tmp')
    parser.add_argument('--load_model', default=None)
    parser.add_argument('--embedding', default=None)

    parser.add_argument('--cdr', default='123456')
    parser.add_argument('--all_atom', action='store_true', default=False)
    parser.add_argument('--dist', action='store_true', default=False)
    parser.add_argument('--decoy', action='store_true', default=False)
    parser.add_argument('--gaussian', action='store_true', default=False)
    parser.add_argument('--nosidechain', action='store_true', default=False)
    parser.add_argument('--nolm', action='store_true', default=False)

    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--hidden_size', type=int, default=256)
    parser.add_argument('--esm_size', type=int, default=2560)
    parser.add_argument('--bert_size', type=int, default=2560)
    parser.add_argument('--epitope_size', type=int, default=50)
    parser.add_argument('--depth', type=int, default=1)
    parser.add_argument('--vocab_size', type=int, default=len(ALPHABET) * 3)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--threshold', type=float, default=20.0)

    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--seed', type=int, default=7)
    parser.add_argument('--anneal_rate', type=float, default=0.95)
    parser.add_argument('--clip_norm', type=float, default=1.0)

    args = parser.parse_args()
    print(args)

    os.makedirs(args.save_dir, exist_ok=True)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    train_data = AntibodyDataset(args.train_path, args.cdr, args.epitope_size)
    val_data = AntibodyDataset(args.val_path, args.cdr, args.epitope_size)
    test_data = AntibodyDataset(args.test_path, args.cdr, args.epitope_size)

    if args.nolm:
        args.esm_size = args.bert_size = len(ALPHABET)
        embedding = {}
        for entry in train_data.data + val_data.data + test_data.data:
            for field in ['antibody_seq', 'antigen_seq']:
                seq = entry[field]
                embedding[seq] = torch.zeros(len(seq), args.esm_size)
                for i,aa in enumerate(seq):
                    embedding[seq][i, ALPHABET.index(aa)] = 1
    elif args.embedding is None:
        embedding = load_esm_embedding(train_data.data + val_data.data + test_data.data, ['antibody_seq', 'antigen_seq'])
        torch.save(embedding, 'embedding.ckpt')
    else:
        embedding = torch.load(args.embedding)

    test_pdb = set([entry['pdb'] for entry in test_data.data])
    val_data = [entry for entry in val_data.data if entry['pdb'] not in test_pdb]

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
            binder, target = AntibodyDataset.make_local_batch(batch, embedding, args)
            if args.nosidechain:
                loss = model(binder, target, use_sidechain=False)
            elif args.gaussian:
                loss = model.gaussian_forward(binder, target)
            else:
                loss = model(binder, target)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), args.clip_norm)
            optimizer.step()

        val_corr = evaluate(model, val_data, embedding, args)
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

    test_corr = evaluate(model, test_data.data, embedding, args)
    print(f'Test Corr = {test_corr:.4f}')
