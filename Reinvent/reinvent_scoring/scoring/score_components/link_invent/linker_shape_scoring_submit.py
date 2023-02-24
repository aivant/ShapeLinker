'''
Script for accessing the surface alignment model rom the command line for the use in REINVENT with RL.
Takes SMILES from the console, aligns to a query molecule and saves the aligned poses.
'''

import os
import sys

import numpy as np
import copy
import torch
import argparse
from tqdm import tqdm
from rdkit import Chem
from rdkit.Chem import MCS
from unidip import UniDip
import unidip.dip as dip

class LinkerShapeScoringSubmit:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.output_path = os.path.join(self.args.output_folder, f'poses_step_{self.args.step}')

    def main(self):
        sys.path.append(self.args.alignment_repo_path)
        from structural.molecule import MoleculeInfo

        os.makedirs(self.output_path, exist_ok=True)

        query_path = os.path.join(self.args.output_folder, 'query.mol')
        
        if not os.path.isfile(query_path):
            if self.args.query_type == 'sdf':
                self.query_mol = MoleculeInfo.from_sdf(self.args.query_file)
            elif self.args.query_type == 'mol2':
                self.query_mol = MoleculeInfo.from_molblock(self.args.query_file)
            elif self.args.query_type == 'smiles':
                self.query_mol = MoleculeInfo.from_smiles(self.args.query_file)
            self.query_mol.write_to_file(os.path.join(self.args.output_folder, 'query.mol'))
        else:
            query_molblock = Chem.MolToMolBlock(Chem.MolFromMolFile(query_path))
            self.query_mol = MoleculeInfo.from_molblock(query_molblock)
        
        smiles = self._smiles_from_console(self.args.smiles_cmd)
        
        self.model = torch.load(self.args.model_path)
        self.model = self.model.to(self.device)
        self.model.eval()

        if not os.path.isfile(os.path.join(self.args.output_folder, 'query.mol')):
            self.query_mol.write_to_file(os.path.join(self.args.output_folder, 'query.mol'))
        
        self.query_rdkit_mol = Chem.MolFromMolFile(query_path)

        smiles_dict = {}
        score_rmsd_dict = {}
        for i, smile in enumerate(smiles):
            smiles_dict[i] = smile
            score_rmsd_dict[i] = (np.nan, np.nan)
        
        goal_nonflip = 0.9
        max_iter = 5
        self.completed = 0
        total = len(smiles)

        with torch.no_grad():
            for i in sorted(smiles_dict.keys()):
                score, rmsd = self._get_score(smiles_dict[i], i)
                score_rmsd_dict[i] = (score, rmsd)
            iterations = 1
            rmsds_sorted = np.msort(np.array([i[1] for i in score_rmsd_dict.values()]))
            rmsds_sorted = rmsds_sorted[~np.isnan(rmsds_sorted)]
            uniformity_check = dip.diptst(rmsds_sorted)
            if uniformity_check[1] is not None and uniformity_check[1] < 0.05 and self.args.correct_flipping:
                intervals = UniDip(rmsds_sorted, mrg_dst=1).run()
                try:
                    self.split_point = (rmsds_sorted[intervals[0][1]] + rmsds_sorted[intervals[-1][0]]) / 2
                except:
                    self.split_point = (rmsds_sorted[intervals[0][0]] + rmsds_sorted[intervals[0][1]]) / 2
                self._get_num_completed(score_rmsd_dict)
                while self.completed/total < goal_nonflip and iterations <= max_iter:
                        for i in self.indices_fail:
                            score, rmsd = self._get_score(smiles_dict[i], i)
                            score_rmsd_dict[i] = (score, rmsd)
                        iterations += 1
                        self._get_num_completed(score_rmsd_dict)
                    
        scores = [score_rmsd_dict[i][0] for i in sorted(score_rmsd_dict.keys())]

        return scores

    def _get_num_completed(self, score_rmsd_dict: dict):
        self.indices_pass = [i for i, j in score_rmsd_dict.items() if j[1] < self.split_point]
        self.indices_fail = [i for i, j in score_rmsd_dict.items() if j[1] >= self.split_point]
        # add indices with nan rmsd to indices_fail
        self.indices_fail += [i for i, j in score_rmsd_dict.items() if np.isnan(j[1])]
        self.completed = len(self.indices_pass)

    def _smiles_from_console(self, console: str) -> list:
        smiles = console.split(';')
        return smiles

    def _get_score(self, smile, i):
        pose_file = os.path.join(self.output_path, f'pose_{i}_{smile}.mol')
        try:
            alignment = self.query_mol.align_to_multiconformer_smiles_fast2(smile, model=self.model, number_of_conformers=self.args.num_conformers, 
                                                                    device=self.device, addhs_in_post=False, es_weight=self.args.es_weight)
            score = alignment.combined_distance
            mol_pose = alignment.molecule_2
            mol_pose.write_to_file(pose_file)
            if self.args.correct_flipping:
                mol_pose_rdkit = Chem.MolFromMolFile(pose_file)
                rmsd = self.calc_mcs_rmsd(mol_pose_rdkit, self.query_rdkit_mol)
            else:
                rmsd = np.nan
        except:
            try:
                alignment = self.query_mol.align_to_multiconformer_smiles_fast2(smile, model=self.model, number_of_conformers=self.args.num_conformers, 
                                                                        device=self.device, addhs_in_post=True, es_weight=self.args.es_weight)
                score = alignment.combined_distance
                mol_pose = alignment.molecule_2
                mol_pose.write_to_file(pose_file)
                if self.args.correct_flipping:
                    mol_pose_rdkit = Chem.MolFromMolFile(pose_file)
                    rmsd = self.calc_mcs_rmsd(mol_pose_rdkit, self.query_rdkit_mol)
                else:
                    rmsd = np.nan
            except:
                score = 99 # when conformer optimization still fails, assign unreasonably high score     
                rmsd = np.nan
        return score, rmsd

    def is_nb_in_ring(self, ringinfo, atom_next_idx, mcs_match, only_mol, mol):
        if ringinfo.AtomMembers(atom_next_idx):
                mcs_match.remove(atom_next_idx)
                only_mol.append(atom_next_idx)
                return self.find_next_atom(mol, mcs_match, only_mol, ringinfo)
        else:
            return mcs_match, only_mol
            
    def find_next_atom(self, mol, mcs_match, only_mol, ringinfo):
        atom_next_idx = None
        for bond in mol.GetBonds():
            a1, a2 = [bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()]
            if a1 in mcs_match and a2 in only_mol:
                atom_next_idx = a1
                mcs_match, only_mol = self.is_nb_in_ring(ringinfo, atom_next_idx, mcs_match, only_mol, mol)
            elif a2 in mcs_match and a1 in only_mol:
                atom_next_idx = a2
                mcs_match, only_mol = self.is_nb_in_ring(ringinfo, atom_next_idx, mcs_match, only_mol, mol)
        return mcs_match, only_mol

    def adapt_mcs(self, mol1, mol2, ringinfo1, ringinfo2):
        mcs = MCS.FindMCS([mol1, mol2], completeRingsOnly=True, ringMatchesRingOnly=True, matchValences=True)
        mcs_mol = Chem.MolFromSmarts(mcs.smarts)
        mcs_match1 = mol1.GetSubstructMatch(mcs_mol)
        mcs_match2 = mol2.GetSubstructMatch(mcs_mol)
        mol1_all_atoms = [a.GetIdx() for a in mol1.GetAtoms()]
        mol2_all_atoms = [a.GetIdx() for a in mol2.GetAtoms()]
        only_mol1 = [a for a in mol1_all_atoms if a not in mcs_match1]
        only_mol2 = [a for a in mol2_all_atoms if a not in mcs_match2]

        mcs_match1_adapt, only_mol1 = self.find_next_atom(mol1, list(mcs_match1), only_mol1, ringinfo1)
        # find indices between mcs2_match and mol2_all_atoms
        mcs_match2_adapt, only_mol2 = self.find_next_atom(mol2, list(mcs_match2), only_mol2, ringinfo2)

        if len(mcs_match1_adapt) > len(mcs_match2_adapt):
            mcs_mol_adapt = self.remove_atoms(mol2, only_mol2)
            mcs_match1_adapt = mol1.GetSubstructMatch(mcs_mol_adapt)
        elif len(mcs_match1_adapt) < len(mcs_match2_adapt):
            mcs_mol_adapt = self.remove_atoms(mol1, only_mol1)
            mcs_match2_adapt = mol2.GetSubstructMatch(mcs_mol_adapt)
        else:
            mcs_mol_adapt = self.remove_atoms(mol1, only_mol1) 
        return mcs_mol_adapt

    def get_substruct_frags(self, mol1, mol2):
        ringinfo1  = mol1.GetRingInfo()
        ringinfo2  = mol2.GetRingInfo()
        mcs_a_mol = self.adapt_mcs(mol1, mol2, ringinfo1, ringinfo2)

        return mcs_a_mol

    def remove_atoms(self, mol, indices):
        remove_atoms = sorted(indices, reverse=True)
        # Chem.Kekulize(mol)
        mol_red = Chem.RWMol(mol)    
        for idx in remove_atoms:
            mol_red.RemoveAtom(idx)
        mol_red = mol_red.GetMol()
        return mol_red

    def get_frags(self, mol, indices):
        all_atoms = [a.GetIdx() for a in mol.GetAtoms()]
        remove_idx = [a for a in all_atoms if a not in indices]
        mol_red = self.remove_atoms(mol, remove_idx)
        return mol_red

    def get_matching_indices(self, mol1, mol2, combo):
        '''
        Get the indices of the atoms in mol1 that match the atoms in mol2
        '''
        # get matching atoms
        for a in mol1.GetAtoms():
            a.SetAtomMapNum(a.GetIdx()+1)
        for a in mol2.GetAtoms():
            a.SetAtomMapNum(a.GetIdx()+1)
        matches1 = mol1.GetSubstructMatches(combo)
        matches2 = mol2.GetSubstructMatches(combo)
        both_matches = []
        for idx1 in matches1:
            mol1_copy = copy.deepcopy(mol1)
            frag_mol1 = self.get_frags(mol1_copy, idx1)
            for idx2 in matches2:
                mol2_copy = copy.deepcopy(mol2)
                frag_mol2 = self.get_frags(mol2_copy, idx2)
                if frag_mol2.HasSubstructMatch(frag_mol1) and frag_mol1.HasSubstructMatch(frag_mol2):
                    both_matches.append(self.mapnum_2_indices(frag_mol1, 
                                                        frag_mol2))
        return both_matches
    
    def mapnum_2_indices(self, mol1, mol2):
        mapnum_to_idx = {}
        indices1_ori = [a.GetAtomMapNum()-1 for a in mol1.GetAtoms()]
        for a in mol2.GetAtoms():
            mapnum_to_idx[a.GetIdx()] = a.GetAtomMapNum()-1
        matches = mol2.GetSubstructMatch(mol1)
        # get the map numbers of the atoms in the linker
        indices2_ori = [mapnum_to_idx[idx] for idx in matches]
        return indices1_ori, indices2_ori

    def get_coords(self, mol, att_points):
        '''
        Returns the coordinates of the attachment points
        '''
        coords = []
        for idx in att_points:
            coords.append(mol.GetConformer().GetAtomPosition(idx))
        return coords

    def calc_rmsd_all(self, coords1, coords2):
        '''
        Calculates the RMSD between two sets of coordinates. 
        Need to be in matching order.
        '''
        rmsd = 0
        for i in range(len(coords1)):
            rmsd += (coords1[i].x - coords2[i].x)**2 + (coords1[i].y - coords2[i].y)**2 + (coords1[i].z - coords2[i].z)**2
        rmsd = np.sqrt(rmsd/len(coords1))
        
        return rmsd

    def calc_mcs_rmsd(self, mol1, mol2):
        mcs = self.get_substruct_frags(mol1, mol2)
        matching_indices = self.get_matching_indices(mol1, mol2, mcs)
        rmsds = []
        for match in matching_indices:
            coords1 = self.get_coords(mol1, match[0])
            coords2 = self.get_coords(mol2, match[1])
            rmsds.append(self.calc_rmsd_all(coords1, coords2))
        return min(rmsds)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='shape alignment for scoring')
    parser.add_argument('--model_path', type=str, help='path to model')
    parser.add_argument('--query_file', type=str, help='path to query file or SMILES of query')
    parser.add_argument('--query_type', type=str, help='type of query file (sdf, mol2, smiles)')
    parser.add_argument('--smiles_cmd', type=str, help='SMILES from command line separated by ;', required=True)
    parser.add_argument('--num_conformers', type=int, help='Number of conformers to generate for each SMILES', default=4)
    parser.add_argument('--output_folder', type=str, help='Folder to save output files')
    parser.add_argument('--step', type=int, help='Step of the RL run.')
    parser.add_argument('--alignment_repo_path', type=str, help='Path to the alignment repo')
    parser.add_argument('--es_weight', type=float, help='Weight of electrostatics in the combined score', default=0.0)
    parser.add_argument('--correct_flipping', action='store_true', help='Resample for correct orientation of alignment.', default=False)

    args = parser.parse_args()
    shape_scoring_submit = LinkerShapeScoringSubmit(args)
    scores = shape_scoring_submit.main()
    for score in scores:
        print(score, end="\n")
