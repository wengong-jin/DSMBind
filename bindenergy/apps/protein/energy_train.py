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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_path', default='data/protein/data.pkl')
    parser.add_argument('--test_path', default='data/skempi/data.pkl')
    parser.add_argument('--save_dir', default='ckpts/tmp')
    parser.add_argument('--load_model', default=None)
    parser.add_argument('--embedding', default=None)

    parser.add_argument('--all_atom', action='store_true', default=False)
    parser.add_argument('--decoy', action='store_true', default=False)
    parser.add_argument('--gaussian', action='store_true', default=False)
    parser.add_argument('--nosidechain', action='store_true', default=False)
    parser.add_argument('--nolm', action='store_true', default=False)

    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--hidden_size', type=int, default=256)
    parser.add_argument('--esm_size', type=int, default=2560)
    parser.add_argument('--bert_size', type=int, default=2560)
    parser.add_argument('--patch_size', type=int, default=50)
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

    train_data = ProteinDataset(args.train_path, args.patch_size)
    test_data = ProteinDataset(args.test_path, args.patch_size)

    if args.nolm:
        args.esm_size = args.bert_size = len(ALPHABET)
        embedding = {}
        for entry in train_data.data + test_data.data:
            for field in ['binder_full', 'target_full']:
                seq = entry[field]
                embedding[seq] = torch.zeros(len(seq), args.esm_size)
                for i,aa in enumerate(seq):
                    embedding[seq][i, ALPHABET.index(aa)] = 1
    elif args.embedding is None:
        embedding = load_esm_embedding(train_data.data + test_data.data, ['binder_full', 'target_full'])
        torch.save(embedding, 'embedding.ckpt')
    else:
        embedding = torch.load(args.embedding)

    model = FADecoyModel(args).cuda() if args.decoy else AllAtomEnergyModel(args).cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = lr_scheduler.ExponentialLR(optimizer, args.anneal_rate)
    if args.load_model:
        model_ckpt, opt_ckpt, model_args = torch.load(args.load_model)
        model = FADecoyModel(args).cuda() if args.decoy else AllAtomEnergyModel(args).cuda()
        model.load_state_dict(model_ckpt)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        scheduler = lr_scheduler.ExponentialLR(optimizer, args.anneal_rate)

    step = 0
    for e in range(args.epochs):
        random.shuffle(train_data.data)
        for i in trange(0, len(train_data.data), args.batch_size):
            model.train()
            optimizer.zero_grad()
            batch = train_data.data[i : i + args.batch_size]
            bfield, tfield = ('binder', 'target') if random.random() > 0.5 else ('target', 'binder')
            binder, target = ProteinDataset.make_local_batch(batch, embedding, args, bfield, tfield)
            if args.nosidechain:
                loss = model(binder, target, use_sidechain=False)
            elif args.gaussian:
                loss = model.gaussian_forward(binder, target)
            else:
                loss = model(binder, target)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), args.clip_norm)
            optimizer.step()

            step = step + args.batch_size
            if step > 0 and step % 100 == 0:
                ckpt = (model.state_dict(), optimizer.state_dict(), args)
                torch.save(ckpt, os.path.join(args.save_dir, f"model.ckpt.{step}"))

        scheduler.step()
