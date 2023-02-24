from typing import List, Union, Dict, Tuple, Set
import os

import pandas as pd
import numpy as np
import rdkit
from rdkit import Chem
from rdkit.Chem import Mol, AllChem
import igraph
import networkx as nx
from networkx.classes.graph import Graph
from networkx.algorithms.shortest_paths.generic import shortest_path
from copy import deepcopy
from tqdm import tqdm
import copy
try:
    from pymol import cmd
except:
    pass
try:
    from rdkit.Chem import rdFMCS
except:
    pass

########################################################################################
## utils for data post-processing of DiffLinker

def add_h_pymol(filepath:str):
    """
    Use pymol to add Hs before loading molecules into RDKit.
    """
    cmd.reinitialize()
    cmd.load(filepath)
    cmd.h_add('(all)')
    cmd.save(filepath)

def add_conformers(folderpath: str):
    """
    RDKit's SDMolSupplier cannot infer stereochemistry from SDFs that
    have radicals and no hydrogens. Use pymol to fix this by adding Hs
    before loading molecules into RDKit.
    """
    print(len(os.listdir(folderpath)))
    for i, f_name in enumerate(os.listdir(folderpath)):
        f_path = os.path.join(folderpath, f_name)
        if "sdf" in f_path:
            add_h_pymol(f_path)

def align_ligand(folderpath: str, ref_anc_wrh: Mol):
    error = 0
    for file in os.listdir(folderpath):
        if 'input' not in file and 'sdf' in file:
            try:
                gen_mol = Chem.SDMolSupplier(os.path.join(folderpath, file))[0]
                for a in gen_mol.GetAtoms():
                    a.SetAtomMapNum(a.GetIdx()+1)
                atom_map = {a.GetAtomMapNum():a.GetIdx() for a in gen_mol.GetAtoms()}
                algned_gen_mol =AllChem.AssignBondOrdersFromTemplate(ref_anc_wrh, gen_mol)
                match_aligned = algned_gen_mol.GetSubstructMatch(ref_anc_wrh)
                for b in algned_gen_mol.GetBonds():
                    a1, a2 = b.GetBeginAtom(), b.GetEndAtom()
                    if a1.GetIdx() not in match_aligned or a2.GetIdx() not in match_aligned:
                        m1, m2 = a1.GetAtomMapNum(), a2.GetAtomMapNum()
                        b.SetBondType(gen_mol.GetBondBetweenAtoms(atom_map[m1], atom_map[m2]).GetBondType())
                for a in algned_gen_mol.GetAtoms():
                    a.SetAtomMapNum(0)
                for a in gen_mol.GetAtoms():
                    a.SetAtomMapNum(0)
                Chem.MolToMolFile(algned_gen_mol, os.path.join(folderpath, file))
                add_h_pymol(os.path.join(folderpath, file))
            except Exception as e:
                # print(e)
                error += 1
    print('Number of errors: ',error)

            
def mol_with_atom_index(mol: Mol) -> Mol:
    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(atom.GetIdx())
    return mol

def remove_index(mol: Mol) -> Union[Mol, str]:
    if type(mol) is str:
        is_smi = True
        mol = Chem.MolFromSmiles(mol)
    else:
        is_smi = False
    
    [a.SetAtomMapNum(0) for a in mol.GetAtoms()]
    
    if is_smi:
        smi = Chem.MolToSmiles(mol)
        return smi
    else:
        return mol

def get_frags_f_name(mol_f_name: str, PDB_id: str) -> str:
    """
    Get file name of fragments that generated a difflinker output.
    """
    
    title_init_idx = mol_f_name.index(PDB_id)
    len_idx = mol_f_name.index("_len")
    input_title_substr = mol_f_name[title_init_idx:len_idx]
    frags_f_name = "input_" + input_title_substr + ".sdf"
    
    return frags_f_name

def neutralizeRadicals(mol: Mol):
    for a in mol.GetAtoms():
        a.SetNumRadicalElectrons(0)   

def expand_frags(mol, warhead, wrh_frag, replacement_attch_map_wrh,
                      anchor, anc_frag, replacement_attch_map_anc):
    """
    Replace small fragments containing exit vectors with entire original warheads and anchors
    from which they were extracted.
    
    Args:
    mol - RDKit Mol containing the small fragments to make the replacements in.
    warhead - RDKit Mol representing the warhead
    wrh_frag - RDKit Mol representing the small fragment of the warhead in mol with a dummy attachment
    warhead_exit_idx - index of atom in warhead that binds to the linker
    anchor - RDKit Mol representing the anchor
    anc_frag - RDKit Mol representing the small fragment of the anchor in mol with a dummy attachment
    anchor_exit_idx - index of atom in anchor that binds to the linker
    
    Returns:
    RDKit mol representing a PROTAC with the entire anchor and warhead.
    """
    
    mol = Chem.MolFromSmiles(Chem.MolToSmiles(mol))
    wrh_frag = Chem.MolFromSmiles(Chem.MolToSmiles(wrh_frag))
    warhead = Chem.MolFromSmiles(Chem.MolToSmiles(warhead))
    replacement = warhead
    wrh_map = {a.GetAtomMapNum(): a.GetIdx() for a in replacement.GetAtoms()}
    replacement_attch_idx = wrh_map[replacement_attch_map_wrh]
    p = Chem.AdjustQueryParameters.NoAdjustments()
    p.makeDummiesQueries = True
    query_wrh = Chem.AdjustQueryProperties(wrh_frag, p)
    mol_with_warhead = Chem.rdmolops.ReplaceSubstructs(mol, query_wrh, replacement, replacementConnectionPoint=replacement_attch_idx)[0]
    replacement = anchor
    anc_map = {a.GetAtomMapNum(): a.GetIdx() for a in replacement.GetAtoms()}
    replacement_attch_idx = anc_map[replacement_attch_map_anc]
    query_anc = Chem.AdjustQueryProperties(anc_frag, p)
    complete_mol = Chem.rdmolops.ReplaceSubstructs(mol_with_warhead, query_anc,
                                                    replacement,
                                                    replacementConnectionPoint=replacement_attch_idx)[0]

    #The replacement will cut off attachment points
    resulting_frags = Chem.GetMolFrags(complete_mol, asMols = True)
    resulting_frags = sorted(resulting_frags, key = lambda mol: mol.GetNumAtoms())
    complete_mol = resulting_frags[-1]
    neutralizeRadicals(complete_mol)
    
    assert complete_mol.HasSubstructMatch(query_wrh), "Warhead not a substructure of expanded mol"
    assert complete_mol.HasSubstructMatch(query_anc), "Anchor not a substructure of expanded mol"
    assert complete_mol.HasSubstructMatch(mol), "Input mol not a substructure of expanded mol"
    
    return complete_mol

def get_attachment_type(mol, frag_atom_idxs):
    """
    Gets the attachment atom and bond type of a frag in a mol, where exit vector is defined
    as the atom in a fragment with a bond to another atom
    that is outside of the fragment, and that atom outside is the attachment atom.
    
    Args:
    mol - RDKit mol object.
    frag_atom_idxs - idxs of a fragment in mol.
    
    Returns:
    - Exit vector bond type to attachment
    - Attachment atom type
    """
    attachments = []
    for atom_idx in frag_atom_idxs:
        atom = mol.GetAtomWithIdx(atom_idx)
        nbrs = atom.GetNeighbors()
        for nbr in nbrs:
            nbr_idx = nbr.GetIdx()
            if nbr_idx not in frag_atom_idxs:
                bond = mol.GetBondBetweenAtoms(atom_idx, nbr_idx)
                bond_type = bond.GetBondType()
                # print(nbr_idx, atom_idx)
                attachments.append((bond_type, nbr.GetAtomicNum()))

    # print(attachments)
    assert len(attachments) == 1
    return attachments[0][0], attachments[0][1]

def get_superfrag_w_filled_attch(original_df, frag_id, mol, frag_type):
    """
    Given a truncated frag with a bond to a dummy atom, a larger "superfrag" containing the frag,
    and a mol containing the frag with a bond to an "attachment atom" corresponding to the dummy...
    replace the dummy in the superfrag with an atom of the same type as the attachment atom,
    and change the bond to the same type as the bond to the attachment atom.
    
    Args:
    original_df - Dataframe containing the warhead, anchor, warhead_ev, anchor_ev, wrh_trunc_w_attch_smiles, 
        anc_trunc_w_attch_smiles, anchors, and trunc_anchors.
    frag_id - ID of the reference ligand
    mol - Mol object containing the fragment
    frag_type - The type of fragment, either "wrh" or "anc"
    
    Returns:
    - Superfrag with the attachment atom type & bond type frag has in mol
    - The truncated frag with an attachment to a dummy atom
    """
    # print("checking", frag_type)
    frag_type_big = "warhead" if frag_type == "wrh" else "anchor"
    replacement_map = 2 if frag_type == "wrh" else 1

    trunc_w_attch_smi = original_df.loc[frag_id][f"{frag_type}_trunc_w_attch_smiles"]
    trunc_w_dummy_attch = Chem.MolFromSmiles(trunc_w_attch_smi)
    neutralizeRadicals(trunc_w_dummy_attch)
    for a in trunc_w_dummy_attch.GetAtoms():
        if a.GetSymbol() == '*':
            dummy_idx = a.GetIdx()
            break
    # find neighbors of dummy atom
    nbrs_trunc_w_dummy_attch = trunc_w_dummy_attch.GetAtomWithIdx(dummy_idx).GetNeighbors()
    assert len(nbrs_trunc_w_dummy_attch) == 1, 'dummy atom has more than one neighbor'
    nbr_trunc_idx = nbrs_trunc_w_dummy_attch[0].GetIdx()
    trunc_smi = trunc_w_attch_smi.replace("*", "")
    trunc = Chem.MolFromSmiles(trunc_smi)
    neutralizeRadicals(trunc)
    trunc_atom_ids = set(mol.GetSubstructMatch(trunc))
    assert len(trunc_atom_ids) > 0 
    # print(Chem.MolToSmiles(mol), trunc_smi, trunc_atom_ids)
    trunc_attch_bond_type, trunc_attch_atom_type = get_attachment_type(mol, trunc_atom_ids)
    # superfrag_smi = original_df.loc[frag_id][frag_type_big]
    superfrag_w_dummy_attch_smi = original_df.loc[frag_id][f"ev_{frag_type_big}"]

    superfrag_w_dummy_attch = Chem.MolFromSmiles(superfrag_w_dummy_attch_smi)
    neutralizeRadicals(superfrag_w_dummy_attch)

    superfrag_w_attch = deepcopy(superfrag_w_dummy_attch)
    superfrag_map = {a.GetAtomMapNum(): a.GetIdx() for a in superfrag_w_attch.GetAtoms()}
    replacement_idx = superfrag_map[replacement_map]
    superfrag_w_attch.GetAtomWithIdx(replacement_idx).SetAtomicNum(trunc_attch_atom_type)
    # get neighbor of replacement atom
    replacement_atom = superfrag_w_attch.GetAtomWithIdx(replacement_idx)
    nbrs = replacement_atom.GetNeighbors()
    assert len(nbrs) == 1
    nbr_idx = nbrs[0].GetIdx()
    superfrag_w_attch.GetBondBetweenAtoms(
        nbr_idx, replacement_idx).SetBondType(
        trunc_attch_bond_type)
    neutralizeRadicals(superfrag_w_attch)
    trunc_w_dummy_attch.GetBondBetweenAtoms(
        dummy_idx, nbr_trunc_idx).SetBondType(
        trunc_attch_bond_type)
    neutralizeRadicals(trunc_w_dummy_attch)
    
    return superfrag_w_attch, trunc_w_dummy_attch

def get_generated_data_df(original_df, results_root, PDB_id):
    """
    Get a dataframe containing info about generated molecules including
    the file names of the corresponding inputs, fragments, linker lengths,
    and corresponding PDB IDs.
    
    Args:
    original_df - Dataframe containing the warhead, anchor, warhead_ev, anchor_ev, wrh_trunc_w_attch_smiles, anc_trunc_w_attch_smiles, anchors, and trunc_anchors.
    """
    
    gen_data = {"ID": [],
                "reference": [],
                "lig_id": [],
                "protac_smiles": [],
                "anchor_smiles": [],
                "warhead_smiles": [],
                "anchor_ev": [],
                "warhead_ev": [],
                "gen_smiles": [],
                "gen_filename": [],
                "frags": [],
                "wrh_trunc_w_attch": [],
                "anc_trunc_w_attch": [],
                "linker_len": [],
                }

    mols = []
    all_files = os.listdir(results_root)
    files = [f for f in all_files if f.split(".")[-1] == "sdf" and f.split("_")[0] != "input"]
    n_total = len(files)
    n_invalid = 0
    n_valid = 0
    for i, f_name in tqdm(enumerate(files), total=len(files)):
        # if n_valid >= 3:
        #     break
            
        split = f_name.split("_")
        is_output = split[0] == "output"
        if is_output:
            lig_id = split[-3]
            linker_len = split[-1].split(".")[0][3:]
            try:
                mol = Chem.SDMolSupplier(f"{results_root}/{f_name}")[0]
                neutralizeRadicals(mol)
                if mol is not None:
                    # print("PROCESSING", i)
                    smi = Chem.MolToSmiles(mol)
                    mols.append(mol)
                    frags_f_name = get_frags_f_name(f_name, PDB_id)

                    frags = Chem.SDMolSupplier(f"{results_root}/{frags_f_name}", removeHs=False)[0]
                    neutralizeRadicals(frags)
                    frags_smiles = Chem.MolToSmiles(frags)
                    ev_warhead_w_attch, wrh_trunc_w_dummy_attch = get_superfrag_w_filled_attch(original_df, lig_id, mol, "wrh")
                    ev_anchor_w_attch, anc_trunc_w_dummy_attch = get_superfrag_w_filled_attch(original_df, lig_id, mol, "anc")
                    anc_attch_idx, wrh_attch_idx = 1, 2 # based on the map number in the ev smiles               
                    protac = expand_frags(mol, ev_warhead_w_attch, wrh_trunc_w_dummy_attch, wrh_attch_idx,
                                               ev_anchor_w_attch, anc_trunc_w_dummy_attch, anc_attch_idx)
                    remove_index(protac)

                    # print(f"\nPROTAC = '{protac_smiles}'")
                    Chem.rdmolops.AssignStereochemistryFrom3D(protac)
                    Chem.AssignAtomChiralTagsFromStructure(protac)
                    protac_with_stereo_smiles = Chem.MolToSmiles(protac, isomericSmiles=True)
                    assert '.' not in protac_with_stereo_smiles, 'Protac smiles is more than one molecule'

                    warhead_smi = original_df.loc[lig_id].warhead
                    anchor_smi = original_df.loc[lig_id].anchor
                    ev_warhead_smi = original_df.loc[lig_id].ev_warhead
                    ev_anchor_smi = original_df.loc[lig_id].ev_anchor
                    wrh_trunc_w_attch_smi = original_df.loc[lig_id].wrh_trunc_w_attch_smiles
                    anc_trunc_w_attch_smi = original_df.loc[lig_id].anc_trunc_w_attch_smiles

                    gen_data["ID"].append(f'{PDB_id}_difflinker_{i}')
                    gen_data["reference"].append(PDB_id)
                    gen_data["lig_id"].append(lig_id)
                    gen_data["protac_smiles"].append(protac_with_stereo_smiles)
                    gen_data["warhead_smiles"].append(warhead_smi)
                    gen_data["anchor_smiles"].append(anchor_smi)
                    gen_data["anchor_ev"].append(ev_anchor_smi)
                    gen_data["warhead_ev"].append(ev_warhead_smi)
                    gen_data["gen_smiles"].append(smi)
                    gen_data["gen_filename"].append(f_name)
                    gen_data["frags"].append(frags_smiles)
                    gen_data["wrh_trunc_w_attch"].append(wrh_trunc_w_attch_smi)
                    gen_data["anc_trunc_w_attch"].append(anc_trunc_w_attch_smi)
                    gen_data["linker_len"].append(linker_len)
            except Exception as e:
                # print("error", e)
                n_invalid += 1
                continue
            else:
                n_valid += 1


    validity = 1 - (n_invalid / n_total)
    print("validity", validity)
                
    gen_data_df = pd.DataFrame(gen_data)
    return gen_data_df

########################################################################################
## utils for post-processing
def get_linker_new(protac, warhead_smi, anchor_smi):
    warhead_mol = Chem.MolFromSmiles(warhead_smi)
    anchor_mol = Chem.MolFromSmiles(anchor_smi)
    def remove_warhead(s, kekulize=False):
        s_copy = copy.deepcopy(s)
        if kekulize:
            Chem.Kekulize(s_copy,clearAromaticFlags=True)
            Chem.Kekulize(warhead_mol,clearAromaticFlags=True)
        return Chem.AllChem.DeleteSubstructs(s_copy, warhead_mol)

    def remove_anchor(s, kekulize=False):
        if kekulize:
            Chem.Kekulize(anchor_mol,clearAromaticFlags=True)
        return Chem.AllChem.DeleteSubstructs(s, anchor_mol)

    linker_im = remove_warhead(protac)
    linker = remove_anchor(linker_im)
    try:
        Chem.Kekulize(linker)
        return linker
    except Chem.KekulizeException:
        linker_im = remove_warhead(protac, kekulize=True)
        linker = remove_anchor(linker_im, kekulize=True)
        return linker

########################################################################################
## utils for metrics
def check_linker_branch(linker: Mol, linker_indices: List[int]) -> Union[None, bool]:
    # C-C and C-N bonds are considered as branches
    removable_bond_types = [
        [6, 6, Chem.rdchem.BondType.SINGLE],
        [5, 6, Chem.rdchem.BondType.SINGLE],
    ]

    mnum_idx_dict = {}
    for atom in linker.GetAtoms():
        mnum_idx_dict[atom.GetAtomMapNum()] = atom.GetIdx()

    branch_exists = False
    for atom in linker.GetAtoms():
        # Scan non-ring atoms on the shortest path. If it's connected
        # to off-path atoms through C-C or C-N bonds, marked as branched
        # linkers. If it's connected to off-path atoms only through C=O
        # bonds, the linker is considered unbranched.
        if not atom.IsInRing():
            elem1 = atom.GetAtomicNum()
            for nei in atom.GetNeighbors():
                if nei.GetAtomMapNum() not in linker_indices:
                    if not nei.IsInRing():
                        elem2 = nei.GetAtomicNum()
                        bond = linker.GetBondBetweenAtoms(
                            mnum_idx_dict[atom.GetAtomMapNum()],
                            mnum_idx_dict[nei.GetAtomMapNum()],
                        )
                        bond_type = sorted([elem1, elem2])
                        bond_type = bond_type + [bond.GetBondType()]
                        if bond_type in removable_bond_types:
                            branch_exists = True

    return branch_exists

def mol_with_atom_index(mol: Mol) -> Mol:
    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(atom.GetIdx())
    return mol

def mol_remove_atom_index(mol: Mol) -> Mol:
    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(0)
    return mol

def mol2graph(mol: Mol) -> Graph:
    atoms_info = [
        (atom.GetIdx(), atom.GetAtomicNum(), atom.GetSymbol())
        for atom in mol.GetAtoms()
    ]
    bonds_info = [
        (
            bond.GetBeginAtomIdx(),
            bond.GetEndAtomIdx(),
            bond.GetBondType(),
            bond.GetBondTypeAsDouble(),
        )
        for bond in mol.GetBonds()
    ]
    graph = igraph.Graph()
    for atom_info in atoms_info:
        graph.add_vertex(
            atom_info[0], AtomicNum=atom_info[1], AtomicSymbole=atom_info[2]
        )
    for bond_info in bonds_info:
        graph.add_edge(
            bond_info[0],
            bond_info[1],
            BondType=bond_info[2],
            BondTypeAsDouble=bond_info[3],
        )
    graph_nx = nx.Graph(graph.get_edgelist())
    return graph_nx

def intercom_shortest_path(
        graph: Graph, matched_atoms_in_mol: List[List[int]]
) -> List[int]:
    all_paths = []
    all_path_lengths = []
    for i in matched_atoms_in_mol[0]:
        for j in matched_atoms_in_mol[1]:
            shortest = shortest_path(graph, i, j)
            all_paths.append(shortest)
            all_path_lengths.append(len(shortest))
    the_path = all_paths[int(np.argmin(all_path_lengths))]
    return the_path

def is_good_linker(protac: Mol, ev_smiles: str, anc_smiles: str) -> bool:
    whr_patt = Chem.MolFromSmiles(
        ev_smiles.replace("([*:1])", "").replace("([*:2])", "").replace('[*:1]', '').replace('[*:2]', '').replace('*', '')
    )
    anc_patt = Chem.MolFromSmiles(
        anc_smiles.replace("([*:1])", "").replace("([*:2])", "").replace('[*:1]', '').replace('[*:2]', '').replace('*', '')
    )

    # anchor atom indices
    mcs = rdFMCS.FindMCS([protac, anc_patt], timeout=3)
    mcs = Chem.MolFromSmarts(mcs.smartsString)

    # anchor exit atom index in protac
    try:
        protac_anc_indices = protac.GetSubstructMatches(mcs)[0]
    except Exception:
        protac_anc_indices = protac.GetSubstructMatches(
            Chem.MolFromSmiles(Chem.MolToSmiles(mcs))
        )[0]

    # whr atom indeixes
    mcs = rdFMCS.FindMCS([protac, whr_patt], timeout=3)
    mcs = Chem.MolFromSmarts(mcs.smartsString)

    # warhead exit atom index in protac
    try:
        protac_whr_indices = protac.GetSubstructMatches(mcs)[0]
    except Exception:
        protac_whr_indices = protac.GetSubstructMatches(
            Chem.MolFromSmiles(Chem.MolToSmiles(mcs))
        )[0]

    # linker atom indices
    protac_graph = mol2graph(protac)
    linker_indices = intercom_shortest_path(
        protac_graph, [list(protac_whr_indices), list(protac_anc_indices)]
    )
    # escape if no linker atoms found
    if len(linker_indices) < 3:
        return False

    # Extract a linker fragment
    mol_with_atom_index(protac)
    em = Chem.EditableMol(protac)
    em.RemoveBond(linker_indices[0], linker_indices[1])
    em.RemoveBond(linker_indices[-2], linker_indices[-1])
    frags = Chem.GetMolFrags(em.GetMol(), asMols=True, sanitizeFrags=False)
    atom_id_in_frags = [[atom.GetAtomMapNum() for atom in f.GetAtoms()] for f in frags]
    linker = [
        frags[i] for i in range(len(frags)) if linker_indices[1] in atom_id_in_frags[i]
    ][0]

    # check if there are extra branches
    has_branch = check_linker_branch(linker, linker_indices)

    mol_remove_atom_index(protac)

    return has_branch