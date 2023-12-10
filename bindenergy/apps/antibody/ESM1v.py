# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import torch
import scipy
from tqdm import tqdm
from typing import List, Tuple

from esm import Alphabet, pretrained
from bindenergy import *
from sklearn.metrics import roc_auc_score
from bindenergy.apps.antibody.semi_train import build_data

torch.set_num_threads(8)

def evaluate(eval_data, model_idx):
    model, alphabet = pretrained.load_model_and_alphabet(f"esm1v_t33_650M_UR90S_{model_idx}")
    batch_converter = alphabet.get_batch_converter()
    model = model.cuda()
    model.eval()

    pred, label = [], []
    for ab in tqdm(eval_data):
        seq = ab['antibody_seq'] + ab['antigen_seq']
        batch_labels, batch_strs, batch_tokens = batch_converter([("seq", seq)])
        # compute probabilities at each CDR position
        log_prob = 0
        for i,v in enumerate(ab['antibody_cdr']):
            if v in args.cdr:
                batch_tokens_masked = batch_tokens.clone()
                batch_tokens_masked[0, i] = alphabet.mask_idx
                with torch.no_grad():
                    token_probs = torch.log_softmax(model(batch_tokens_masked.cuda())["logits"], dim=-1)
                    log_prob += token_probs[0, i, alphabet.get_idx(seq[i])].item()
        pred.append(log_prob)
        label.append(ab['affinity'])
    return scipy.stats.spearmanr(label, pred)[0]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', required=True)
    parser.add_argument('--ref_path', default=None)
    parser.add_argument('--cdr', default='123456')
    parser.add_argument('--seed', type=int, default=7)
    parser.add_argument('--epitope_size', type=int, default=50)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    if args.ref_path is None:
        data = AntibodyDataset(args.data_path, args.cdr, args.epitope_size).data
    else:
        reference = AntibodyDataset(args.ref_path, args.cdr, args.epitope_size).data[0]
        data = build_data(args.data_path, reference)

    for model_idx in range(1, 6):
        corr = evaluate(data, model_idx)
        print(f"esm1v_t33_650M_UR90S_{model_idx}: Corr = {corr:.4f}")
