from typing import List, Union, Dict, Tuple, Set
import copy

import pandas as pd
import rdkit
from rdkit import Chem
from rdkit.Chem import AllChem, rdRGroupDecomposition, rdmolops
from rdkit.Chem.MolStandardize import rdMolStandardize
from copy import deepcopy

def get_linker_frag(mol: Chem.Mol, frag1: Chem.Mol, frag2: Chem.Mol) -> Chem.Mol:
    '''
    Extract linker from a molecule.
    :param mol: RDKit molecule (protac)
    :param frag1: RDKit molecule (e.g anchor)
    :param frag2: RDKit molecule (e.g. warhead)
    :return: RDKit molecule (linker)
    '''
    def remove_warhead(s, kekulize=False):
        s_copy = copy.deepcopy(s)
        if kekulize:
            Chem.Kekulize(s_copy,clearAromaticFlags=True)
            Chem.Kekulize(frag1,clearAromaticFlags=True)
        return Chem.AllChem.DeleteSubstructs(s_copy, frag1)

    def remove_anchor(s, kekulize=False):
        if kekulize:
            Chem.Kekulize(frag2,clearAromaticFlags=True)
        return Chem.AllChem.DeleteSubstructs(s, frag2)

    linker_im = remove_warhead(mol)
    linker = remove_anchor(linker_im)
    try:
        Chem.Kekulize(linker)
        return linker
    except Chem.KekulizeException:
        linker_im = remove_warhead(mol, kekulize=True)
        linker = remove_anchor(linker_im, kekulize=True)
        return linker

def neutralizeRadicals(mol: Chem.Mol):
     for a in mol.GetAtoms():
         a.SetNumRadicalElectrons(0)

def get_nb_ringinfo(a, nb_ringsystem):
    '''
    Get indices of neighbors if they are in a ring (recursive)
    :param a: RDKit atom
    :param nb_ringsystem: list of indices of neighbors in a ring
    :return: list of indices of neighbors in a ring
    '''
    nbs = a.GetNeighbors()
    for nb in nbs:
        if nb.IsInRing() and nb.GetIdx() not in nb_ringsystem:
            ring_idx = nb.GetIdx()
            nb_ringsystem.append(ring_idx)
            nb_ringsystem=get_nb_ringinfo(nb, nb_ringsystem)
    return nb_ringsystem

def GetRingSystems(mol: Chem.Mol, includeSpiro: bool =False) -> List[Set[int]]:
    ri = mol.GetRingInfo()
    systems = []
    for ring in ri.AtomRings():
        ringAts = set(ring)
        nSystems = []
        for system in systems:
            nInCommon = len(ringAts.intersection(system))
            if nInCommon and (includeSpiro or nInCommon>1):
                ringAts = ringAts.union(system)
            else:
                nSystems.append(system)
        nSystems.append(ringAts)
        systems = nSystems
    return systems

def remove_atom_indices(mol: Chem.Mol, atom_indics: List[int]) -> Chem.Mol:
    mol_red = Chem.RWMol(mol)    
    for idx in atom_indics:
        mol_red.RemoveAtom(idx)
    mol_red = mol_red.GetMol()
    return mol_red

def replace_atom_indices(mol: Chem.Mol, atom_indics: List[int], atom_num: int = 1) -> Chem.Mol:
    mol_red = Chem.RWMol(mol)    
    for idx in atom_indics:
        mol_red.ReplaceAtom(idx, Chem.Atom(atom_num))
    mol_red = mol_red.GetMol()
    return mol_red

def remove_ring_attachment(mol: Chem.Mol, frag: Chem.Mol) -> Chem.Mol:
    '''
    Modifies anchor or warhead by removing the atom neighboring the exit vector.
    If this atom is in a ring, the whole ring is removed.
    Goal: Being able to extract linker plus attaching structure (for shape alignment)
    :param mol: PROTAC RDKit mol
    :param frag: anchor or warhead RDKit mol
    :return: modified anchor or warhead RDKit mol
    '''
    groups, _ = rdRGroupDecomposition.RGroupDecompose([frag], [mol], asSmiles=True)
    frag_ev = Chem.MolFromSmiles(groups[0]['Core'])
    ring_systems = GetRingSystems(frag_ev)
    nb_ringsystem = []
    for a in frag_ev.GetAtoms():
        if a.GetSymbol() == '*':
            ev_idx = a.GetIdx()
            nb = a.GetNeighbors()
            assert len(nb) == 1
            nb_idx = nb[0].GetIdx()
            # check if any of neighbors are in a ring
            for ring in ring_systems:
                if nb_idx in ring:
                    nb_ringsystem = list(ring)
                    break  
            nb_ringsystem.append(ev_idx)
    # if only one atom (exit vector): remove also the neighbor
    ev_nb = None
    if len(nb_ringsystem) == 1:
        ev_nb = frag_ev.GetAtomWithIdx(nb_ringsystem[0]).GetNeighbors()[0].GetIdx()
        nb_ringsystem_minus = nb_ringsystem.copy()
        nb_ringsystem.append(ev_nb)
    nb_ringsystem.sort(reverse=True)
    frag_adapt = remove_atom_indices(frag_ev, nb_ringsystem)
    try:
        frag_adapt_im = copy.deepcopy(frag_adapt)
        Chem.Kekulize(frag_adapt_im)
    except Chem.KekulizeException:
        if ev_nb is not None and len(rdmolops.GetMolFrags(frag_adapt)) == 1:
            nb_ringsystem_minus.sort(reverse=True)
            frag_adapt = remove_atom_indices(frag_ev, nb_ringsystem_minus)
            frag_adapt = replace_atom_indices(frag_adapt, [ev_nb])
            frag_adapt = Chem.RemoveHs(frag_adapt)
    frags_split = rdmolops.GetMolFrags(frag_adapt)
    if len(frags_split) > 1:
        idx_to_remove = []
        for frag in frags_split:
            # remove fragments that are max 3 heavy atoms (side chains to ring)
            if len(frag) <= 3:
                idx_to_remove.append(frag[:])
        idx_to_remove = [item for sublist in idx_to_remove for item in sublist]
        idx_to_remove.sort(reverse=True)
        frag_adapt = remove_atom_indices(frag_adapt, idx_to_remove)
    try:
        frag_adapt_im = copy.deepcopy(frag_adapt)
        Chem.SanitizeMol(frag_adapt_im)
        return frag_adapt_im
    except:
        return frag_adapt

def get_exit_vec(mol, frag_atom_idxs):
    """
    Gets the exit vector of a frag in a mol, where exit vector is defined
    as the atom in a fragment with a bond to another atom
    that is outside of the fragment.
    
    :param mol: RDKit mol object.
    :param frag_atom_idxs: idxs of a fragment in mol.
    :return: Exit vector atom idx
    :return: Exit vector Atom object
    """

    exit_vecs = []
    for atom_idx in frag_atom_idxs:
        atom = mol.GetAtomWithIdx(atom_idx)
        nbrs = atom.GetNeighbors()
        for nbr in nbrs:
            nbr_idx = nbr.GetIdx()
            if nbr_idx not in frag_atom_idxs:
                bond = mol.GetBondBetweenAtoms(atom_idx, nbr_idx)
                bond_type = bond.GetBondType()
                exit_vecs.append((atom_idx, atom, bond_type))

    assert len(exit_vecs) == 1
    return exit_vecs[0][0], exit_vecs[0][1], exit_vecs[0][2]

def add_nbrs(atom, depth, max_depth, frag_idxs, keep, ringinfo):
    """
    Add neighbors of an atom to keep from being truncated.
    Recursively add an atom and its neighborhood to _keep_
    upto a certain depth away from the original atom --
    unless a neighbor is in a ring, then complete the ring
    
    :param atom: RDKit Atom in a mol
    :param depth: The distance from the original atom at the current step
    :param max_depth: How far away from the original atom to go into the frag
    :param keep: Idxs of atoms in the neighborhood to keep
    :param ringinfo: Object with info about the rings in mol
    """

    for nbr in atom.GetNeighbors():
        if nbr.GetIdx() in frag_idxs and not nbr.GetIdx() in keep:
            in_same_ring = ringinfo.AreAtomsInSameRing(nbr.GetIdx(), atom.GetIdx())
            if depth < max_depth:
                keep.add(nbr.GetIdx())
                add_nbrs(nbr, depth+1, max_depth, frag_idxs, keep, ringinfo)
            elif nbr.IsInRing() and in_same_ring:
                keep.add(nbr.GetIdx())
                add_nbrs(nbr, depth+1, max_depth, frag_idxs, keep, ringinfo)
                
def truncate_frags(mol, ringinfo,
                   wrh_idxs, wrh_exit_vec, wrh_exit_vec_id,
                   anc_idxs, anc_exit_vec, anc_exit_vec_id,
                   all_atom_idxs, linker_idxs):
    """
    Truncate PROTAC around the linker so that small fragments 
    around the attachment atoms are present.
    
    :param mol: RDKit Mol object containing an anchor, warhead, and linker
    :param ringinfo: Object with info about the rings in mol
    :param wrh_idxs: idxs of atoms in the warhead
    :param wrh_exit_ve: atom representing warhead exit vector
    :param wrh_exit_vec_id: idx of wrh_exit_vec in mol
    :param anc_idxs: idxs of atoms in the anchor
    :param anc_exit_vec: atom representing anchor exit vector
    :param anc_exit_vec_id: idx of anc_exit_vec in mol
    :param linker_idxs: idxs of atoms in linker
    """
    num_before = mol.GetNumAtoms()
    keep = set([wrh_exit_vec_id, anc_exit_vec_id]).union(linker_idxs)
    add_nbrs(wrh_exit_vec, 1, 2, wrh_idxs, keep, ringinfo)
    add_nbrs(anc_exit_vec, 1, 2, anc_idxs, keep, ringinfo)
    extra_atoms = list(all_atom_idxs.difference(keep))
    onestep_idx = []

    # replace atoms that are one bond away from kept structure with H
    extra_atoms_loop = extra_atoms.copy()
    for a_idx in extra_atoms_loop:
        for b_idx in keep:
            bond = mol.GetBondBetweenAtoms(a_idx, b_idx)
            if bond is not None:
                if bond.GetBondType() == Chem.rdchem.BondType.SINGLE:
                    onestep_idx.append(a_idx)
                else:
                    # keep atoms that are not connected by single bond 
                    extra_atoms.remove(a_idx)
    extra_atoms = list(set(extra_atoms).difference(set(onestep_idx)))
    extra_atoms.sort(reverse=True)
    for a in mol.GetAtoms():
        a.SetAtomMapNum(a.GetIdx())
    for extra_atom_idx in extra_atoms:
        mol.RemoveAtom(extra_atom_idx)
    for a in mol.GetAtoms():
        mapnum = a.GetAtomMapNum()
        if mapnum in onestep_idx:
            mol.ReplaceAtom(a.GetIdx(), Chem.Atom(1))
        a.SetAtomMapNum(0)
        
    assert mol.GetNumAtoms() < num_before

def get_frags(mol: Chem.Mol, wrh: Chem.Mol, anc: Chem.Mol, removeHs: bool = True):
    """
    Get truncated anchor and warhead fragments from a PROTAC as well as their exit vector atom idxs.
    
    :param mol: RDKit Mol object representing the PROTAC.
    :param frag_df: DataFrame containing SMILES representing the PROTAC, its anchor, and warhead.
    :param row_id: The idx of the row corresponding to mol in frag_df.
    
    :return: RDKit mol object representing the anchor and warhead fragments
    :return: Idx of the anchor's exit vector atom in the truncated mol.
    :return: Idx of the warhead's exit vector atom in the truncated mol.
    :return: Idx of the anchor's exit vector atom in the original mol.
    :return: Idx of the warhead's exit vector atom in the original mol.
    """
    
    mol = Chem.RWMol(mol)
        
    all_atom_ids = set(range(mol.GetNumAtoms()))
    wrh_atom_ids = set(mol.GetSubstructMatch(wrh))
    anc_atom_ids = set(mol.GetSubstructMatch(anc))
    linker_atom_ids = all_atom_ids.difference(wrh_atom_ids.union(anc_atom_ids))
    wrh_exit_vec_id, wrh_exit_vec, wrh_exit_bond = get_exit_vec(mol, wrh_atom_ids)
    anc_exit_vec_id, anc_exit_vec, anc_exit_bond = get_exit_vec(mol, anc_atom_ids)

    # Truncate
  
    ringinfo = mol.GetRingInfo()
    
    truncate_frags(mol, ringinfo,
                   wrh_atom_ids, wrh_exit_vec, wrh_exit_vec_id,
                   anc_atom_ids, anc_exit_vec, anc_exit_vec_id,
                   all_atom_ids, linker_atom_ids)
    
    #Get indices of exit vectors in truncated fragments
    wrh_trunc_exit_vec_id = wrh_exit_vec.GetIdx()
    anc_trunc_exit_vec_id = anc_exit_vec.GetIdx()

    if removeHs:
        mol = Chem.RemoveHs(mol)
    else:
        mol= mol.GetMol()

    return mol, anc_trunc_exit_vec_id, wrh_trunc_exit_vec_id, anc_exit_vec_id, wrh_exit_vec_id

def generate_ext_linker(
    protac: Union[str, Chem.Mol], 
    anchor: Union[str, Chem.Mol], 
    warhead: Union[str, Chem.Mol], 
    return_mol: bool = False, 
    removeHs: bool = True, 
    return_idx: bool = False,
    return_pd_series: bool =False
) -> Union[str, Chem.Mol]:
    """
    Extract linker including small fragments from both anchor and warhead.
    
    :param protac_smi: Smiles of the PROTAC
    :param anchor_smi: Smiles of the anchor
    :param warhead_smi: Smiles of the warhead
    :param return_mol: If True, return RDKit mol object of the extracted fragment. If False, return SMILES.
    :return: RDKit mol object of the extracted fragment
    """
    if type(protac) == str:
        protac = Chem.MolFromSmiles(protac)
    if type(anchor) == str:
        anchor = Chem.MolFromSmiles(anchor)
    if type(warhead) == str:
        warhead = Chem.MolFromSmiles(warhead)
    #Extract fragments from ligand
    frags, anc_trunc_exit_vec_id, wrh_trunc_exit_vec_id, anc_exit_vec_id, wrh_exit_vec_id = get_frags(protac, warhead, anchor, removeHs=removeHs)
    
    if return_mol and not return_idx:
        return frags
    elif return_mol and return_idx:
        return frags, anc_trunc_exit_vec_id, wrh_trunc_exit_vec_id, anc_exit_vec_id, wrh_exit_vec_id
    elif not return_mol and return_idx:
        return Chem.MolToSmiles(frags), anc_trunc_exit_vec_id, wrh_trunc_exit_vec_id, anc_exit_vec_id, wrh_exit_vec_id
    elif return_pd_series:
        return pd.Series([Chem.MolToSmiles(frags)])
    else:
        return Chem.MolToSmiles(frags)


def set_stereo2query(mol_input: Chem.Mol, query: Chem.Mol) -> Chem.Mol:
    mol = deepcopy(mol_input)
    p = Chem.AdjustQueryParameters.NoAdjustments()
    p.makeDummiesQueries = True
    query = Chem.AdjustQueryProperties(query, p)
    match_idx= mol_input.GetSubstructMatch(query)

    assert len(match_idx) == query.GetNumAtoms()

    # define atom chirality
    for i, idx in enumerate(match_idx):
        test_sym = mol.GetAtomWithIdx(idx).GetSymbol()
        stereo_sym = query.GetAtomWithIdx(i).GetSymbol()
        stereo = query.GetAtomWithIdx(i).GetChiralTag()
        if test_sym == stereo_sym and stereo != Chem.rdchem.ChiralType.CHI_UNSPECIFIED:
            mol.GetAtomWithIdx(idx).SetChiralTag(stereo)
        if test_sym != stereo_sym and stereo_sym != '*':
            print('mismatch', test_sym, stereo_sym, stereo)
    
    # define stereogenic bonds
    for bond in query.GetBonds():
        if bond.GetBondDir() != Chem.rdchem.BondDir.NONE:
            idx1 = bond.GetBeginAtomIdx()
            idx2 = bond.GetEndAtomIdx()
            bond_mol = mol.GetBondBetweenAtoms(match_idx[idx1], match_idx[idx2])
            if bond_mol is not None and bond_mol.GetBondDir() == Chem.rdchem.BondDir.NONE:
                bond_mol.SetBondDir(bond.GetBondDir())
    
    Chem.rdmolops.AssignStereochemistry(mol, cleanIt=True, force=True)
    return mol