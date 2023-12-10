import csv
import sys
import numpy as np
import json
from prody import *
from tqdm import tqdm
from sidechainnet.utils.measure import get_seq_coords_and_angles


def tocdr(resseq):
    if 27 <= resseq <= 38:
        return '1'
    elif 56 <= resseq <= 65:
        return '2'
    elif 105 <= resseq <= 117:
        return '3'
    else:
        return '0'


if __name__ == "__main__":
    with open('summary.tsv') as f:
        reader = csv.DictReader(f, delimiter='\t')
        hmap = {}
        for row in tqdm(reader, total=14000):
            if row['Hchain'] == 'NA' or len(row['Hchain']) == 0:
                continue
            try:
                pdb_id = row['pdb']
                hchain = parsePDB(f'imgt/{pdb_id}.pdb', model=1, chain=row['Hchain'])
                _, hcoords, hseq, _, _ = get_seq_coords_and_angles(hchain)
                hcoords = hcoords.reshape((-1,14,3))
                hcdr = ''.join([tocdr(res.getResnum()) for res in hchain.iterResidues()])
                L = hcdr.rindex('3') + 11
                hseq, hcdr, hcoords = hseq[:L], hcdr[:L], hcoords[:L]
                hcoords = eval(np.array2string(hcoords, separator=',', threshold=np.inf, precision=3, suppress_small=True))

                if len(row['Lchain']) > 0 and row['Lchain'] != 'NA':
                    lchain = parsePDB(f'imgt/{pdb_id}.pdb', model=1, chain=row['Lchain'])
                    _, lcoords, lseq, _, _ = get_seq_coords_and_angles(lchain)
                    lcoords = lcoords.reshape((-1,14,3))
                    lcdr = ''.join([tocdr(res.getResnum()) for res in lchain.iterResidues()])
                    L = lcdr.rindex('3') + 10
                    lseq, lcdr, lcoords = lseq[:L], lcdr[:L], lcoords[:L]
                    lcoords = eval(np.array2string(lcoords, separator=',', threshold=np.inf, precision=3, suppress_small=True))
                else:
                    lseq = lcdr = lcoords = None
            except Exception as e:
                print(pdb_id, row['Hchain'], row['Lchain'], file=sys.stderr)
                continue

            if lseq is None:
                s = json.dumps({
                    "pdb": pdb_id + "_" + row['Hchain'],
                    "antibody_seq": hseq,
                    "antibody_cdr": hcdr,
                    "antibody_coords": hcoords,
                })
            else:
                s = json.dumps({
                    "pdb": pdb_id + "_" + row['Hchain'] + row['Lchain'],
                    "antibody_seq": hseq + lseq,
                    "antibody_cdr": hcdr + lcdr.replace('1','4').replace('2','5').replace('3','6'),
                    "antibody_coords": hcoords + lcoords,
                })
            print(s)
