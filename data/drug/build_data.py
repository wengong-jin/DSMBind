import sys
import os
import glob
import numpy as np
import pickle
from prody import *
from rdkit import Chem
from rdkit.Chem import PandasTools
from sidechainnet.utils.measure import get_seq_coords_and_angles
from tqdm import tqdm
from multiprocessing import Pool

def work(fn):
    pdb = fn.strip('/').split('/')[-1]
    hchain = parsePDB(f'{fn}/{pdb}_protein.pdb', model=1)

    _, hcoords, hseq, _, _ = get_seq_coords_and_angles(hchain)
    hcoords = hcoords.reshape((len(hseq), 14, 3))
    df = PandasTools.LoadSDF(f'{fn}/{pdb}_protein.sdf', molColName='Molecule', includeFingerprints=False)
    mol = df['Molecule'][0]
    return {
        "pdb": pdb, "binder_mol": mol,
        "target_seq": hseq, "target_coords": hcoords,
    }


if __name__ == "__main__":
    data = []
    for fn in tqdm(sorted(glob.glob('CASF-2016/coreset/*'))):
        pdb = fn.strip('/').split('/')[-1]
        if len(pdb) != 4: continue
        try:
            entry = work(fn)
            data.append(entry)
        except:
            continue

    print(len(data))
    with open('planet_casf16.pkl', 'wb') as f:
        pickle.dump(data, f)
