from collections import defaultdict
from typing import List, Set

from rdkit import Chem
from rdkit.Chem import Mol, EditableMol, GetMolFrags


class BondBreaker:
    """
    breaks and identify bonds in labeled molecules / smiles
    """
    def __init__(self):
        self._mol_atom_map_number = 'molAtomMapNumber'

    def labeled_mol_into_fragment_mols(self, labeled_mol: Mol) -> List[Mol]:
        e_mol = EditableMol(labeled_mol)
        for atom_pair in self.get_bond_atoms_idx_pairs(labeled_mol):
            e_mol.RemoveBond(*atom_pair)
        mol_fragments = GetMolFrags(e_mol.GetMol(), asMols=True, sanitizeFrags=False)
        return mol_fragments

    def get_linker_fragment(self, labeled_mol: Mol, get_input_frags: bool = False):
        """
        Returns the mol of the linker (labeled), where the linker is the only fragment with two attachment points
        returns None if no linker is found
        """
        fragment_mol_list = self.labeled_mol_into_fragment_mols(labeled_mol)
        linker = None
        for fragment in fragment_mol_list:
            labeled_atom_dict = self.get_labeled_atom_dict(fragment)
            if len(labeled_atom_dict) == 2:
                linker = fragment
        if get_input_frags:
            input_frags = [frag for frag in fragment_mol_list if frag != linker]
            return input_frags
        else:
            return linker

    def get_bond_atoms_idx_pairs(self, labeled_mol: Mol):
        labeled_atom_dict = self.get_labeled_atom_dict(labeled_mol)
        bond_atoms_idx_list = [value for value in dict(labeled_atom_dict).values()]
        return bond_atoms_idx_list

    def get_labeled_atom_dict(self, labeled_mol: Mol):
        bonds = defaultdict(list)
        for atom in labeled_mol.GetAtoms():
            if atom.HasProp(self._mol_atom_map_number):
                bonds[atom.GetProp(self._mol_atom_map_number)].append(atom.GetIdx())
        bond_dict = dict(sorted(bonds.items()))
        return bond_dict

    def generate_ext_linker(self, 
        mol: Chem.Mol, 
        wrh: Chem.Mol, 
        anc: Chem.Mol, 
        removeHs: bool = True
    ) -> Chem.Mol:
        """
        Get truncated anchor and warhead fragments including the linker.
        
        :param mol: RDKit Mol object representing the PROTAC.
        :param wrh: RDKit Mol object representing the warhead.
        :param anc: RDKit Mol object representing the anchor.
        :param removeHs: If True, remove hydrogens from the fragments.
        
        :return: RDKit mol object representing the anchor and warhead fragments
        """
        
        mol = Chem.RWMol(mol)
            
        all_atom_ids = set(range(mol.GetNumAtoms()))
        wrh_atom_ids = set(mol.GetSubstructMatch(wrh))
        anc_atom_ids = set(mol.GetSubstructMatch(anc))
        linker_atom_ids = all_atom_ids.difference(wrh_atom_ids.union(anc_atom_ids))
        wrh_exit_vec_id, wrh_exit_vec = self.get_exit_vec(mol, wrh_atom_ids)
        anc_exit_vec_id, anc_exit_vec = self.get_exit_vec(mol, anc_atom_ids)

        if wrh_exit_vec_id is None or anc_exit_vec_id is None:
            return None
        # Truncate
        ringinfo = self.GetRingSystems(mol)
        
        self.truncate_frags(mol, ringinfo,
                    wrh_atom_ids, wrh_exit_vec, wrh_exit_vec_id,
                    anc_atom_ids, anc_exit_vec, anc_exit_vec_id,
                    all_atom_ids, linker_atom_ids)

        if removeHs:
            mol = Chem.RemoveHs(mol)
        else:
            mol= mol.GetMol()

        return mol
    
    def get_exit_vec(self, mol, frag_atom_idxs):
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

        if len(exit_vecs) == 1:
            return exit_vecs[0][0], exit_vecs[0][1]
        else: 
            return None, None

    def add_nbrs(self, atom, depth, max_depth, frag_idxs, keep, ringinfo):
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
                nbr_idx = nbr.GetIdx()
                a_idx = atom.GetIdx()
                # check if in same set of ringinfo
                for ring in ringinfo:
                    if a_idx in ring and nbr_idx in ring:
                        in_same_ring = True
                        break
                    else:
                        in_same_ring = False
                if depth < max_depth:
                    keep.add(nbr.GetIdx())
                    self.add_nbrs(nbr, depth+1, max_depth, frag_idxs, keep, ringinfo)
                elif nbr.IsInRing() and in_same_ring:
                    keep.add(nbr.GetIdx())
                    self.add_nbrs(nbr, depth+1, max_depth, frag_idxs, keep, ringinfo)
                
    def truncate_frags(self, mol, ringinfo,
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
        self.add_nbrs(wrh_exit_vec, 1, 2, wrh_idxs, keep, ringinfo)
        self.add_nbrs(anc_exit_vec, 1, 2, anc_idxs, keep, ringinfo)
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
        for a in mol.GetAtoms(): # not in same loop because kernel dies
            a.SetAtomMapNum(0)
            
        assert mol.GetNumAtoms() < num_before

    def GetRingSystems(self, mol: Chem.Mol, includeSpiro: bool =False) -> List[Set[int]]:
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
