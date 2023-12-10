import biotite.structure as struc
from biotite.structure import AtomArray, Atom
from biotite.structure.io import save_structure
from bindenergy.data.constants import *
from tqdm import tqdm, trange
import torch
import esm


def print_pdb(coord, seq, chain, indices=None):
    array = []
    for i in range(coord.shape[0]):
        idx = indices[i] + 1 if indices else i + 1
        aaname = seq[i]
        aid = ALPHABET.index(aaname)
        aaname = RESTYPE_1to3[aaname]
        for j,atom in enumerate(RES_ATOM14[aid]):
            if atom != '' and (coord[i, j] ** 2).sum() > 1e-4:
                atom = Atom(coord[i, j], chain_id=chain, res_id=idx, atom_name=atom, res_name=aaname, element=atom[0])
                array.append(atom)
    return array


def print_ca_pdb(coord, seq, chain, indices=None):
    array = []
    for i in range(coord.shape[0]):
        idx = indices[i] + 1 if indices else i + 1
        aaname = seq[i]
        aaname = RESTYPE_1to3[aaname]
        atom = Atom(coord[i, 1], chain_id=chain, res_id=idx, atom_name="CA", res_name=aaname, element='C')
        array.append(atom)
    return array


def load_esm_embedding(data, fields):
    model, alphabet = esm.pretrained.esm2_t36_3B_UR50D()
    batch_converter = alphabet.get_batch_converter()
    model = model.cuda()
    model.eval()
    embedding = {}
    with torch.no_grad():
        for f in fields:
            seqs = [d[f] for d in data if d[f] not in embedding]
            for s in tqdm(sorted(set(seqs))):
                batch_labels, batch_strs, batch_tokens = batch_converter([(s, s)])
                batch_tokens = batch_tokens.cuda()
                results = model(batch_tokens, repr_layers=[36], return_contacts=False)
                embedding[s] = results["representations"][36][0, 1:len(s)+1].cpu()
    return embedding
