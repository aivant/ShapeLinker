import random
from rdkit import Chem
from rdkit.Chem import AllChem
from pathlib import Path
from itertools import chain
from tqdm.notebook import tqdm
import copy
from rdkit.Chem import rdMolTransforms
from rdkit.Chem import TorsionFingerprints
import numpy as np



def enumerateTorsions(mol, ring=False):
    conf = mol.GetConformer(0)
    torsion_list = []
    for non_ring_idx in TorsionFingerprints.CalculateTorsionLists(mol)[0]:
        for a,b,c,d in non_ring_idx[0]:
            torsion_list.append((a, b, c, d, rdMolTransforms.GetDihedralDeg(conf, a, b, c, d)))
    if ring:
        for non_ring_idx in TorsionFingerprints.CalculateTorsionLists(mol)[1]:
            for a,b,c,d in non_ring_idx[0]:
                torsion_list.append((a, b, c, d, rdMolTransforms.GetDihedralDeg(conf, a, b, c, d)))
    return torsion_list


def add_noise_to_torsion_angles(mol, deg=1, ring=False):
    mp2 = AllChem.MMFFGetMoleculeProperties(mol, mmffVariant="MMFF94s")
    ff2 = AllChem.MMFFGetMoleculeForceField(mol, mp2)
    for a,b,c,d, angle in enumerateTorsions(mol):
        noise = (np.random.random() * 2 - 1) * deg
        ff2.MMFFAddTorsionConstraint(a,b,c,d, False, noise + angle - .1, noise + angle + .1, 10000.0)
    ff2.Minimize()
    return mol