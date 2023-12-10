import json 
import random
import torch
import numpy as np
from bindenergy.data.constants import *


class ComplexLoader():

    def __init__(self, dataset, batch_tokens, field='epitope_seq'):
        self.dataset = dataset
        self.size = len(dataset)
        self.lengths = [len(dataset[i][field]) for i in range(self.size)]
        self.batch_tokens = batch_tokens
        sorted_ix = np.argsort(self.lengths)

        # Cluster into batches of similar sizes
        clusters, batch = [], []
        for ix in sorted_ix:
            size = self.lengths[ix]
            batch.append(ix)
            if size * (len(batch) + 1) > self.batch_tokens:
                clusters.append(batch)
                batch = []

        self.clusters = clusters
        if len(batch) > 0:
            clusters.append(batch)

    def __len__(self):
        return len(self.clusters)

    def __iter__(self):
        np.random.shuffle(self.clusters)
        for b_idx in self.clusters:
            batch = [self.dataset[i] for i in b_idx]
            yield batch


def featurize(batch, name, vocab=ALPHABET):
    B = len(batch)
    L_max = max([len(b[name + "_seq"]) for b in batch])
    X = torch.zeros([B, L_max, 14, 3])
    S = torch.zeros([B, L_max]).long()
    A = torch.zeros([B, L_max, 14]).long()
    V = torch.zeros([B, L_max, 6])

    # Build the batch
    for i, b in enumerate(batch):
        l = len(b[name + '_seq'])
        indices = torch.tensor([vocab.index(a) for a in b[name + '_seq']])
        S[i,:l] = indices
        X[i,:l] = b[name + '_coords']
        A[i,:l] = b[name + '_atypes']
        V[i,:l] = b[name + '_dihedrals']

    return X.cuda(), S.cuda(), A.cuda(), V.cuda()

