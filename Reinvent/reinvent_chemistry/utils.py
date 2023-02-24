import numpy as np


def get_indices_of_unique_smiles(smiles: [str]) -> np.array:
    """Returns an np.array of indices corresponding to the first entries in a list of smiles strings"""
    _, idxs = np.unique(smiles, return_index=True)
    sorted_indices = np.sort(idxs)
    return sorted_indices