import torch
import torch.nn as nn
import torch.optim as optim

import json
import csv
import math, random, sys
import numpy as np
import argparse
import os
import scipy

from io import StringIO
from bindenergy import *
from bindenergy.apps.antibody.HER2_train import build_data
from tqdm import tqdm, trange
from multiprocessing import Pool
from sklearn.metrics import roc_auc_score

import biotite.sequence.align as align
import biotite.structure as struc
from biotite.sequence import ProteinSequence
from biotite.structure import AtomArray, Atom
from biotite.structure.io.pdb import PDBFile
from pyfoldx.structure import Structure

import pyrosetta
from pyrosetta import *
from pyrosetta.rosetta.protocols.analysis import InterfaceAnalyzerMover
from pyrosetta.rosetta.core.import_pose import pose_from_pdbstring

pyrosetta.init(' '.join([
    '-mute', 'all',
    '-use_input_sc',
    '-ignore_unrecognized_res',
    '-ignore_zero_occupancy', 'false',
    '-load_PDB_components', 'false',
    '-relax:default_repeats', '2',
    '-no_fconfig',
]))

def pyrosetta_interface_energy(pdb_str, interface):
    pose = Pose()
    pose_from_pdbstring(pose, pdb_str)
    mover = InterfaceAnalyzerMover(interface)
    mover.set_pack_separated(True)
    mover.apply(pose)
    return pose.scores['dG_separated']


def compute_energy(s):
    (pdb_id, idx, label), s = s[0], s[1:]
    obj = Structure("pred", from_string=s)
    obj = obj.repair(other_parameters={"repair_Interface": "ONLY"}, verbose=False)
    foldE = obj.getInterfaceEnergy(verbose=False)['Interaction Energy'].values[-1]
    roseE = pyrosetta_interface_energy('\n'.join(obj.toPdb()), "A_B")
    return float(foldE), float(roseE)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', required=True)
    parser.add_argument('--ref_path', default=None)
    parser.add_argument('--cdr', default='123456')
    parser.add_argument('--seed', type=int, default=7)
    parser.add_argument('--epitope_size', type=int, default=50)
    args = parser.parse_args()
    print(args)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    if args.ref_path is None:
        data = AntibodyDataset(args.data_path, args.cdr, args.epitope_size).data
        pdb_id = "None"
    else:
        reference = AntibodyDataset(args.ref_path, args.cdr, args.epitope_size).data[0]
        data = build_data(args.data_path, reference)
        pdb_id = reference['pdb']

    obj_list = []
    for idx,ab in enumerate(tqdm(data)):
        label = ab['affinity']
        ag_array = print_pdb(ab['antigen_coords'].numpy(), ab['antigen_seq'], 'A')
        ab_array = print_pdb(ab['antibody_coords'].numpy(), ab['antibody_seq'], 'B')
        array = struc.array(ag_array + ab_array)
        stream = StringIO()
        f = PDBFile()
        f.set_structure(array)
        f.write(stream)
        obj_list.append([(pdb_id, idx, label)] + stream.getvalue().split('\n'))
        stream.close()

    label, pred = [], []
    with Pool(100) as pool:
        pred = pool.map(compute_energy, obj_list)
        foldX, roseX = zip(*pred)
        label = [ab['affinity'] for ab in data]

    foldX_corr = scipy.stats.spearmanr(foldX, label)[0]
    rosetta_corr = scipy.stats.spearmanr(roseX, label)[0]
    print(args.ref_path, 'foldX', foldX_corr, 'rosetta', rosetta_corr)
