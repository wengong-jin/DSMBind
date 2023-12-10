import csv
import sys
import os
import numpy as np
import json
import math
import pickle
from multiprocessing import Pool
from collections import defaultdict
from tqdm import tqdm
from prody import *
from sidechainnet.utils.measure import get_seq_coords_and_angles


def process(tup):
    pdb, achain, bchain = tup
    _, acoords, aseq, _, _ = get_seq_coords_and_angles(achain)
    _, bcoords, bseq, _, _ = get_seq_coords_and_angles(bchain)
    acoords = acoords.reshape((-1,14,3))
    bcoords = bcoords.reshape((-1,14,3))
    return (pdb, aseq, acoords, bseq, bcoords)


if __name__ == "__main__":
    data = []
    visited = set()
    ab_map = defaultdict(list)
    with open("../sabdab/summary.tsv") as f:
        reader = csv.DictReader(f, delimiter='\t')
        for row in reader:
            if row['antigen_chain'] == 'NA': continue
            if row['Hchain'] == 'NA': continue
            if row['Lchain'] == 'NA': row['Lchain'] = ''
            ab_chain = row['Hchain'] + row['Lchain']
            ag_chain = row['antigen_chain']
            ab_map[row['pdb']].append((ab_chain, ag_chain))

    with open("../tcrdab/summary.tsv") as f:
        reader = csv.DictReader(f, delimiter='\t')
        for row in reader:
            if row['Bchain'] == 'NA': continue
            if row['Achain'] == 'NA': continue
            if row['mhc_chain1'] == 'NA': continue
            if row['mhc_chain2'] == 'NA': continue
            if row['antigen_chain'] == 'NA': row['antigen_chain'] = ''
            ab_chain = row['Bchain'] + row['Achain']
            ag_chain = row['mhc_chain1'] + row['antigen_chain'] + row['mhc_chain2']
            ab_map[row['pdb']].append((ab_chain, ag_chain))

    ddg_map = defaultdict(list)
    with open("skempi_v2.csv") as f:
        reader = csv.DictReader(f, delimiter=';')
        for row in tqdm(reader):
            pdb, achain, bchain = row['#Pdb'].split('_')
            if pdb.lower() in ab_map:
                for x, y in ab_map[pdb.lower()]:
                    if set(achain + bchain) <= set(x + y):
                        achain, bchain = x, y

            if row['#Pdb'] not in visited:
                xchain = parsePDB(f'clean/{pdb}.pdb', model=1, chain=achain)
                ychain = parsePDB(f'clean/{pdb}.pdb', model=1, chain=bchain)
                data.append(
                        ((row['#Pdb'], "", 0), xchain, ychain)
                )
                visited.add(row['#Pdb'])

            try:
                aff_mt = float(row['Affinity_mut_parsed'])
                aff_wt = float(row['Affinity_wt_parsed'])
                ddg = math.log(aff_mt, 10) - math.log(aff_wt, 10)
            except:
                continue

            mutation = row['Mutation(s)_cleaned']
            if os.path.exists(f'clean/{pdb}_{mutation}.pdb'):
                achain = parsePDB(f'clean/{pdb}_{mutation}.pdb', model=1, chain=achain)
                bchain = parsePDB(f'clean/{pdb}_{mutation}.pdb', model=1, chain=bchain)
                data.append(
                        ((row['#Pdb'], mutation, ddg), achain, bchain)
                )

    with Pool(120) as pool:
        data = pool.map(process, data)
    
    new_data = []
    visited = set()
    for (pdb, aseq, acoords, bseq, bcoords) in data:
        if (pdb[0], aseq, bseq) not in visited:
            visited.add((pdb[0], aseq, bseq))
            new_data.append((pdb, aseq, acoords, bseq, bcoords))

    print('Deduplication:', len(new_data))
    with open("data.pkl", 'wb') as f:
        pickle.dump(new_data, f)

