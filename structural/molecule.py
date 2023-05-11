from structural.dmasif_pcg.geometry_processing import ransac_registration, ElasticDistortion, \
    atoms_to_points_normals, curvatures
import torch
import typing as ty
from dataclasses import dataclass
import numpy as np
from torch.nn.utils.rnn import pad_sequence
from pathlib import Path
from structural.dmasif_pcg.mol_model import AtomNet_MP
import open3d as o3d
from pytorch3d.loss import chamfer_distance

from rdkit import Chem
from rdkit.Chem import AllChem
from copy import deepcopy
from pytorch3d import transforms
import prody as pd
from pytorch3d import transforms
import pandas
from rdkit.Chem.rdMolInterchange import JSONToMols

ELE2NUM = {"C": 0, "C1": 0, "H": 1, "O": 2, "N": 3, "S": 4, "SE": 5, "other": 6, "filler": 7}


def batches_to_variable_length(sequence, batches):
    variable_sequences = []
    for i in range(int(batches.max().item()) + 1):
        variable_sequences.append(sequence[batches == i])
    return variable_sequences


def padded_with_zeromask(point_features, surface_batches):
    variable_sized_sequences = batches_to_variable_length(point_features, surface_batches)
    return pad_sequence(variable_sized_sequences, batch_first=True)


@dataclass
class AtomInfo:
    coordinates: ty.Tuple[float, float, float]
    atom_type: str
    attributes: ty.List[int]

    @classmethod
    def from_atom_line(cls, line: str):
        content = line.split()
        coords = tuple([float(x) for x in content[:3]])
        atom_type = content[3]
        attributes = [int(x) for x in content[4:]]
        return cls(coords, atom_type, attributes)

    @property
    def _str(self) -> str:
        coord_segment = ""
        for c in self.coordinates:
            if c < 0:
                coord_segment += "    " + "{:.4f}".format(c)[:6]
            else:
                coord_segment += "    " + "{:.4f}".format(c)[:6]

        return coord_segment + " " + self.atom_type + "   " + "  ".join([str(x) for x in self.attributes])

    @property
    def rdstr(self) -> str:
        coord_segment = ""
        for c in self.coordinates:
            str_coord = "{:.4f}".format(c)
            gapsize = 6 - len(str_coord) + 4
            coord_segment += " "*gapsize + str_coord
        if len(self.atom_type) == 1:
            return coord_segment + " " + self.atom_type + "   " + "  ".join([str(x) for x in self.attributes])
        else:
            return coord_segment + " " + self.atom_type + "  " + "  ".join([str(x) for x in self.attributes])


@dataclass
class BondInfo:
    bond: str

    @classmethod
    def from_bond_line(cls, line: str):
        assert len(line) == 12
        return cls(line)

    @property
    def _str(self) -> str:
        return self.bond


@dataclass
class Alignment:
    molecule_1: ty.Union["MoleculeInfo", None] = None
    molecule_2: ty.Union["MoleculeInfo", None] = None
    surface_pointcloud_1: ty.Union[np.ndarray, None] = None
    surface_pointcloud_2: ty.Union[np.ndarray, None] = None
    chamfer_distance: ty.Union[float, None] = None
    electrostatics_score: ty.Union[float, None] = None
    combined_distance: ty.Union[float, None] = None



@dataclass
class MoleculeInfo:
    header: ty.Tuple[str, str, str]  # title, timestamp, comment
    attributes: str  # Counts line https://en.wikipedia.org/wiki/Chemical_table_file#Molfile
    atom_block: ty.List[AtomInfo]
    bond_block: ty.List[BondInfo]
    smiles: ty.Union[str, None] = None

    @classmethod
    def from_rdkit_mol(cls, rdkit_object):
        mol_block: str = Chem.MolToMolBlock(rdkit_object)
        return cls.from_molblock(mol_block, Chem.MolToSmiles(rdkit_object))

    @classmethod
    def from_molblock(cls, mol_block: str, smiles=None):
        mol_lines = mol_block.split("\n")
        header = ("RDKitBased", "3D", "")
        attributes = mol_lines[3].strip("\n")
        if len(attributes.split()[0]) == 6:
            first_attribute = attributes.split()[0][:3] + " " + attributes.split()[0][3:]
            attributes = first_attribute + attributes[6:]
        atoms = [AtomInfo.from_atom_line(x) for x in mol_lines if len(x.split()) == 16]
        bonds = [BondInfo.from_bond_line(x) for x in mol_lines if len(x) == 12]
        return cls(header, attributes, atoms, bonds, smiles=smiles)

    @classmethod
    def from_smiles(cls, smiles: str, optimize: bool = True, addhs_in_post: bool = False, add_hs=True, max_attempts=1000):
        m = Chem.MolFromSmiles(smiles)
        if addhs_in_post:
            pass
        else:
            if add_hs:
                m = Chem.AddHs(m, addCoords=True)
        AllChem.EmbedMolecule(
            m,
            maxAttempts=max_attempts,
            useRandomCoords=True
        )
        # AllChem.EmbedMultipleConfs(m, 1)

        AllChem.EmbedMultipleConfs(
            m,
            numConfs=1,
            maxAttempts=max_attempts,
            pruneRmsThresh=.1,
            useExpTorsionAnglePrefs=True,
            useBasicKnowledge=True,
            enforceChirality=True,
            numThreads=-1)

        if addhs_in_post and add_hs:
            m = Chem.AddHs(m, addCoords=True)
        mol_block: str = Chem.MolToMolBlock(m)
        return cls.from_molblock(mol_block, smiles)

    @classmethod
    def from_sdf(cls, sdf_file_path, add_hs=True):
        with Chem.SDMolSupplier(sdf_file_path) as suppl:
            for mol in suppl:
                if mol is None:
                    continue
                mol.GetNumAtoms()
                break
        # for mol in m.GetConformers():
            # print(mol)
        if add_hs:
            mol = Chem.AddHs(mol, addCoords=True)
        mol_block: str = Chem.MolToMolBlock(mol)
        smiles = Chem.MolToSmiles(mol)
        return cls.from_molblock(mol_block, smiles)

    @property
    def molfile(self):
        lines = []
        lines.append("1")
        lines.append("  " + "\t".join(self.header))
        lines.append("")
        lines.append(self.attributes)
        for a in self.atom_block:
            lines.append(a.rdstr)
        for b in self.bond_block:
            lines.append(b._str)
        lines.append("M  END")
        return ("\n".join(lines))

    def match_to_molecules(self, molecules: "Molecules", model: "PCRSingleMasked", resolution=.9, distance=.9, device="cuda"):
        batch_size = molecules.atom_batches.max() + 1
        query = Molecules.from_molecule_info([self], device=device, geometricus=False, random_rotation=False)
        query.update_surface_points_and_normals(resolution=resolution, distance=distance)
        surface_coords: torch.Tensor = query.surface_coordinates
        padded_surface_coords = padded_with_zeromask(surface_coords, query.surface_batches).reshape((-1, 3)).repeat(batch_size, 1, 1)
        smallest_batch_x = min([query.surface_batches[query.surface_batches == i].shape[0] for i in
                                range(int(query.surface_batches.max() + 1))])
        padded_surface_coords = padded_surface_coords[:, :smallest_batch_x, :].cuda()

        molecules.update_surface_points_and_normals(resolution=resolution, distance=distance)
        surface_coords2: torch.Tensor = molecules.surface_coordinates
        current_lengths = [molecules.surface_batches[molecules.surface_batches == i].shape[0] for i in
                           range(int(molecules.surface_batches.max() + 1))]
        padded_surface_coords2 = padded_with_zeromask(surface_coords2, molecules.surface_batches).cuda()
        result = model.coarse(padded_surface_coords, padded_surface_coords2, current_lengths)
        distances = []
        for k in range(0, batch_size):
            distances.append((chamfer_distance(result[k].reshape(1, -1, 3),
                                            padded_surface_coords2[k].reshape(1, -1, 3)[:, :current_lengths[k], :]))[0].item())
        return np.min(distances)

    def align_to_molecules(self, molecules: "Molecules", model: "PCRSingleMasked", resolution=.9, distance=.9, device="cuda", return_point_clouds=False, old_scoring=True, rep=1, get_best_electrostatics=False):

        batch_size = molecules.atom_batches.max() + 1
        query = Molecules.from_molecule_info([self], device=device, geometricus=False, random_rotation=False)
        query.update_surface_points_and_normals(resolution=resolution, distance=distance)
        surface_coords: torch.Tensor = query.surface_coordinates
        padded_surface_coords = padded_with_zeromask(surface_coords, query.surface_batches).reshape((-1, 3)).repeat(batch_size, 1, 1)
        smallest_batch_x = min([query.surface_batches[query.surface_batches == i].shape[0] for i in
                                range(int(query.surface_batches.max() + 1))])
        padded_surface_coords = padded_surface_coords[:, :smallest_batch_x, :].cuda()
        alignments: ty.List[Alignment] = []

        for _ in range(rep):
            molecules.update_surface_points_and_normals(resolution=resolution, distance=distance)
            surface_coords2: torch.Tensor = molecules.surface_coordinates
            current_lengths = [molecules.surface_batches[molecules.surface_batches == i].shape[0] for i in
                               range(int(molecules.surface_batches.max() + 1))]
            padded_surface_coords2 = padded_with_zeromask(surface_coords2, molecules.surface_batches).cuda()
            result = model.coarse(padded_surface_coords, padded_surface_coords2, current_lengths)


            rotations, translations, y_translate, centroids = model.coarse.get_rottrans(padded_surface_coords2, padded_surface_coords)
            rotations, translations, y_translate, centroids = rotations.cpu().detach().numpy(), translations.cpu().detach().numpy(), y_translate.cpu().detach().numpy(), centroids.cpu().detach().numpy()
            for k in range(0, batch_size):
                current_alignment = Alignment()
                current_alignment.chamfer_distance = (chamfer_distance(result[k].reshape(1, -1, 3), padded_surface_coords2[k].reshape(1, -1, 3)[:, :current_lengths[k], :]))[0].item()
                if return_point_clouds:
                    surface = padded_surface_coords2[k].reshape(1, -1, 3)[:, :current_lengths[k], :].reshape((-1, 3))
                    current_alignment.surface_pointcloud_2 = ((surface.cpu().detach().numpy() + centroids[k]) @ rotations[k]) + y_translate[k]
                current_alignment.molecule_2 = molecules.molecule_infos[k].transform_coords(translation_matrix=centroids[k]).transform_coords(rotation_matrix=rotations[k], translation_matrix=translations[k] + y_translate[k])
                alignments.append(current_alignment)
                e, _ = self.get_electrostatics_and_shapetanimoto_score(current_alignment.molecule_2)
                current_alignment.electrostatics_score = e

        best_score_idx = np.argmin([x.chamfer_distance for x in alignments])

        alignment = alignments[best_score_idx]
        alignment.molecule_1 = self
        if return_point_clouds:
            alignment.surface_pointcloud_1 = padded_surface_coords[0].cpu().detach().numpy()
        if old_scoring:
            if get_best_electrostatics:
                best_electrostatics = np.max([x.electrostatics_score for x in alignments])
                return -alignment.chamfer_distance + ((1 + best_electrostatics) ** 2), alignment.molecule_2
            return alignment.chamfer_distance, alignment.molecule_2
        return alignment

    def align_to_molecules2(self, molecules: "Molecules", model: "PCRSingleMasked", resolution=.9, distance=.9, device="cuda", return_point_clouds=False, rep=1, es_weight=1.):
        batch_size = molecules.atom_batches.max() + 1
        query = Molecules.from_molecule_info([self], device=device, geometricus=False, random_rotation=False)
        query.update_surface_points_and_normals(resolution=resolution, distance=distance)
        surface_coords: torch.Tensor = query.surface_coordinates
        padded_surface_coords = padded_with_zeromask(surface_coords, query.surface_batches).reshape((-1, 3)).repeat(batch_size, 1, 1)
        smallest_batch_x = min([query.surface_batches[query.surface_batches == i].shape[0] for i in
                                range(int(query.surface_batches.max() + 1))])
        padded_surface_coords = padded_surface_coords[:, :smallest_batch_x, :].cuda()
        alignments: ty.List[Alignment] = []

        for _ in range(rep):
            molecules.update_surface_points_and_normals(resolution=resolution, distance=distance)
            surface_coords2: torch.Tensor = molecules.surface_coordinates
            current_lengths = [molecules.surface_batches[molecules.surface_batches == i].shape[0] for i in
                               range(int(molecules.surface_batches.max() + 1))]
            padded_surface_coords2 = padded_with_zeromask(surface_coords2, molecules.surface_batches).cuda()
            result = model.coarse(padded_surface_coords, padded_surface_coords2, current_lengths)

            rotations, translations, y_translate, centroids = model.coarse.get_rottrans(padded_surface_coords, padded_surface_coords2, current_lengths)
            rotations, translations, y_translate, centroids = rotations.cpu().detach().numpy(), translations.cpu().detach().numpy(), y_translate.cpu().detach().numpy(), centroids.cpu().detach().numpy()
            for k in range(0, batch_size):
                current_alignment = Alignment()
                current_alignment.chamfer_distance = (chamfer_distance(result[k].reshape(1, -1, 3), padded_surface_coords2[k].reshape(1, -1, 3)[:, :current_lengths[k], :]))[0].item()
                if return_point_clouds:
                    surface = padded_surface_coords2[k].reshape(1, -1, 3)[:, :current_lengths[k], :].reshape((-1, 3)).cpu().detach().numpy()
                    current_alignment.surface_pointcloud_2 = ((surface - y_translate[k]) @ rotations[k].T) - centroids[k]
                current_alignment.molecule_2 = molecules.molecule_infos[k].transform_coords(translation_matrix=-translations[k]-y_translate[k]).transform_coords(rotation_matrix=rotations[k].T).transform_coords(translation_matrix=centroids[k])
                # current_alignment.molecule_2 = molecules.molecule_infos[k].transform_coords(translation_matrix=centroids[k]).transform_coords(rotation_matrix=rotations[k], translation_matrix=translations[k] + y_translate[k])
                alignments.append(current_alignment)
                e, _ = self.get_electrostatics_and_shapetanimoto_score(current_alignment.molecule_2)
                current_alignment.electrostatics_score = e
                current_alignment.combined_distance = (current_alignment.chamfer_distance + ((1 - current_alignment.electrostatics_score) * es_weight))

        best_score_idx = np.argmin([x.combined_distance for x in alignments])
        alignment = alignments[best_score_idx]
        alignment.electrostatics_score = np.max([x.electrostatics_score for x in alignments])
        alignment.molecule_1 = self
        if return_point_clouds:
            alignment.surface_pointcloud_1 = padded_surface_coords[0].cpu().detach().numpy()
        return alignment

    def get_electrostatics_and_shapetanimoto_score(self, molinfo: "MoleculeInfo"):
        from espsim import GetEspSim
        try:
            rdmol1 = Chem.MolFromMolBlock(self.molfile)
            rdmol2 = Chem.MolFromMolBlock(molinfo.molfile)

            rdmol1 = Chem.AddHs(rdmol1, addCoords=True)
            rdmol2 = Chem.AddHs(rdmol2, addCoords=True)
        except:
            from biotite.structure.io import mol
            self.write_to_file("temp.mol")
            m = mol.MOLFile()
            m.set_structure(mol.MOLFile.read("temp.mol").get_structure())
            rdmol1 = Chem.MolFromMolBlock("\n".join(m.lines))

            molinfo.write_to_file("temp.mol")
            m = mol.MOLFile()
            m.set_structure(mol.MOLFile.read("temp.mol").get_structure())
            rdmol2 = Chem.MolFromMolBlock("\n".join(m.lines))

            rdmol1 = Chem.AddHs(rdmol1, addCoords=True)
            rdmol2 = Chem.AddHs(rdmol2, addCoords=True)

        return GetEspSim(rdmol1, rdmol2, prbCid=0, refCid=0, partialCharges="gasteiger"), 1-AllChem.ShapeTanimotoDist(rdmol1,rdmol2,confId1=0,confId2=0)

    def align_to_multiconformer_smiles_fast2(self, smiles: str,
                                            model,
                                            resolution=.9,
                                            distance=.9,
                                            device="cuda",
                                            number_of_conformers=50,
                                            addhs_in_post=False,
                                            add_hs=True,
                                            return_point_clouds=False,
                                            rep=1, es_weight=3, max_conformer_attempts=10_000):
        smiles_molecules = Molecules.from_smiles_conformers(smiles, number_of_conformers, addhs_in_post=addhs_in_post, add_hs=add_hs, max_attempts=max_conformer_attempts)
        return self.align_to_molecules2(smiles_molecules, model, resolution=resolution, distance=distance, device=device, return_point_clouds=return_point_clouds, rep=rep, es_weight=es_weight)

    def align_to_multiconformer_smiles_fast(self, smiles: str,
                                            model,
                                            resolution=.9,
                                            distance=.9,
                                            device="cuda",
                                            number_of_conformers=50,
                                            addhs_in_post=False,
                                            add_hs=True,
                                            return_point_clouds=False,
                                            old_scoring=True,
                                            rep=1,
                                            get_best_electrostatics=False, max_conformer_attempts=10_000):
        smiles_molecules = Molecules.from_smiles_conformers(smiles, number_of_conformers, addhs_in_post=addhs_in_post, add_hs=add_hs, max_attempts=max_conformer_attempts)
        return self.align_to_molecules(smiles_molecules, model, resolution=resolution, distance=distance, device=device, return_point_clouds=return_point_clouds, old_scoring=old_scoring, rep=rep, get_best_electrostatics=get_best_electrostatics)

    def align_to_multiconformer_smiles(self, smiles: str, model, resolution=.9, distance=.9, device="cuda", number_of_conformers=50, limit=1.5):
        query = Molecules.from_molecule_info([self], device=device, geometricus=False, random_rotation=False)
        query.update_surface_points_and_normals(resolution=resolution, distance=distance)

        surface_coords: torch.Tensor = query.surface_coordinates
        current_lengths = [query.surface_batches[query.surface_batches == i].shape[0] for i in
                           range(int(query.surface_batches.max() + 1))]
        padded_surface_coords = padded_with_zeromask(surface_coords, query.surface_batches).cuda()

        chamfer_distances = []
        transformed = []

        for _ in range(number_of_conformers):
            smiles_molecules = Molecules.from_list_of_smiles([smiles], optimize=True)
            smiles_molecules.update_surface_points_and_normals(resolution=resolution, distance=distance)
            surface_coords2: torch.Tensor = smiles_molecules.surface_coordinates

            padded_surface_coords2 = padded_with_zeromask(surface_coords2, smiles_molecules.surface_batches).cuda()
            smiles_molecules.update_surface_points_and_normals(resolution=resolution, distance=distance)

            result = model.coarse(padded_surface_coords2, padded_surface_coords)
            rotations, translations, y_translate, centroids = model.coarse.get_rottrans(padded_surface_coords2, padded_surface_coords)
            rotations, translations, y_translate, centroids = rotations.cpu().detach().numpy(), translations.cpu().detach().numpy(), y_translate.cpu().detach().numpy(), centroids.cpu().detach().numpy()
            dist = (chamfer_distance(result[0].reshape(1, -1, 3),
                                     padded_surface_coords[0].reshape(1, -1, 3)))[0].item()
            chamfer_distances.append(dist)

            transformed.append(
                smiles_molecules.molecule_infos[0].transform_coords(translation_matrix=centroids[0]).transform_coords(rotation_matrix=rotations[0], translation_matrix=translations[0] + y_translate[0])
            )

            if dist < limit:
                break
        min_idx = np.argmin(chamfer_distances)
        return chamfer_distances[min_idx], transformed[min_idx]

    @staticmethod
    def get_smiles_batch(smiles, resolution=.9, distance=.9,
                         batch_size=32, device="cuda", add_hs=True, max_conformer_attempts=10000):
        molecules2: Molecules = Molecules.from_smiles_conformers(smiles, batch_size, add_hs=add_hs, max_attempts=max_conformer_attempts)
        molecules2.update_surface_points_and_normals(resolution=resolution, distance=distance)

        assert molecules2.surface_coordinates is not None
        surface_coords2: torch.Tensor = molecules2.surface_coordinates
        padded_surface_coords2 = padded_with_zeromask(surface_coords2, molecules2.surface_batches)

        translation_matrices = torch.randn((batch_size, 1, 3)).cuda() * 100
        rotation_matrices = transforms.random_rotations(batch_size).cuda()
        transformed_with_features = (padded_surface_coords2 @ rotation_matrices) + translation_matrices
        smallest_batch_y = min([molecules2.surface_batches[molecules2.surface_batches == i].shape[0] for i in
                                range(int(molecules2.surface_batches.max() + 1))])
        transformed_with_features = transformed_with_features[:, :smallest_batch_y,
        :]  # add something to adjust for 0s at the end
        return transformed_with_features.cuda()

    def get_self_batch(self, resolution=.9, distance=.9,
                       batch_size=32, device="cuda", generate_conformer=False, random_rotation=True) -> "Molecules":
        if not generate_conformer:
            query_mols = Molecules.from_molecule_info([self] * batch_size, random_rotation=random_rotation, device=device, geometricus=False)
        else:
            query_mols = Molecules.from_list_of_smiles([self.smiles] * batch_size, device=device, geometricus=False)
        query_mols.update_surface_points_and_normals(resolution=resolution, distance=distance)
        surface_coords: torch.Tensor = query_mols.surface_coordinates
        original_with_features = padded_with_zeromask(surface_coords, query_mols.surface_batches)
        smallest_batch_x = min([query_mols.surface_batches[query_mols.surface_batches == i].shape[0] for i in
                                range(int(query_mols.surface_batches.max() + 1))])
        original_with_features = original_with_features[:, :smallest_batch_x, :]
        return original_with_features.cuda()

    def get_chamfer_distance(self, other: "MoleculeInfo", resolution=.9, distance=.9, device="cuda", random_rotation=False):
        self_cloud = self.get_point_cloud(resolution=resolution, distance=distance, random_rotation=random_rotation, device=device)
        other_cloud = other.get_point_cloud(resolution=resolution, distance=distance, random_rotation=random_rotation, device=device)
        return chamfer_distance(self_cloud.reshape((1, -1, 3)), other_cloud.reshape((1, -1, 3)))[0].item()

    def get_self_molecule(self, resolution=.9, distance=.9, device="cuda", random_rotation=False):
        mol = Molecules.from_molecule_info([self], random_rotation=random_rotation, device=device, geometricus=False)
        mol.update_surface_points_and_normals(resolution=resolution, distance=distance)
        return mol

    def get_point_cloud(self, resolution=.9, distance=.9, device="cuda", random_rotation=False):
        return self.get_self_molecule(resolution=resolution, distance=distance, random_rotation=random_rotation, device=device).surface_coordinates

    def get_training_batch_against_smiles(self, smiles, resolution=.9, distance=.9,
                                         batch_size=32, device="cuda", generate_conformer=False):

        original_with_features = self.get_self_batch(resolution=resolution, distance=distance,
                                                     batch_size=batch_size, device=device, generate_conformer=generate_conformer)
        transformed_with_features = self.get_smiles_batch(smiles, resolution=resolution, distance=distance,
                                                          batch_size=batch_size, device=device)
        return original_with_features, transformed_with_features

    def get_training_batches(self, list_of_smiles, resolution=.9, distance=.9,
                             batch_size=32, batch_num=10, device="cuda", generate_conformer=False, add_hs=True,  max_conformer_attempts=10000):
        batches = []
        original_with_features = self.get_self_batch(resolution=resolution, distance=distance,
                                                     batch_size=batch_size, device=device, generate_conformer=generate_conformer)
        for i in range(len(list_of_smiles)):
            rotation_matrices = transforms.random_rotations(batch_size).cuda()
            translation_matrices = torch.randn((batch_size, 1, 3)).cuda() * 100

            features = self.get_smiles_batch(list_of_smiles[i], resolution=resolution,
                                             distance=distance, batch_size=batch_size,
                                             device=device, add_hs=add_hs, max_conformer_attempts=10000)
            for _ in range(batch_num):
                batches.append(
                    (original_with_features, (features @ rotation_matrices) + translation_matrices)
                )
        return batches

    def get_atom_coords(self):
        return np.array([x.coordinates for x in self.atom_block])

    def get_atom_types(self):
        return [x.atom_type for x in self.atom_block]

    def to_atomtype_coord_pairs(self):
        return self.get_atom_types(), self.get_atom_coords()

    def transform_coords(self, rotation_matrix=None, translation_matrix=None) -> "MoleculeInfo":

        new_class = MoleculeInfo(
            self.header, self.attributes, deepcopy(self.atom_block), deepcopy(self.bond_block)
        )

        coords = self.get_atom_coords()
        if rotation_matrix is not None:
            coords = coords @ rotation_matrix
        if translation_matrix is not None:
            coords += translation_matrix
        for i, atom in enumerate(new_class.atom_block):
            atom.coordinates = tuple(list(coords[i]))
        return new_class

    def write_to_file(self, filename: str):
        f = open(filename, "w")
        f.write(self.molfile)
        f.close()

    def to_graph(self):
        pass


def get_molecule_infos_from_smiles_with_batched_conformers(smiles, number_of_conformers=5, addhs_in_post: bool = False, add_hs=True, max_attempts=10_000):
    m = Chem.MolFromSmiles(smiles)
    Chem.SanitizeMol(m)
    if addhs_in_post:
        pass
    else:
        if add_hs:
            m = Chem.AddHs(m, addCoords=True)
    AllChem.EmbedMolecule(
        m,
        maxAttempts=max_attempts,
        useRandomCoords=False
    )
    AllChem.EmbedMultipleConfs(
        m,
        numConfs=number_of_conformers,
        maxAttempts=max_attempts,
        pruneRmsThresh=.1,
        useExpTorsionAnglePrefs=True,
        useBasicKnowledge=True,
        enforceChirality=True,
        numThreads=-1)

    if addhs_in_post and add_hs:
        m = Chem.AddHs(m, addCoords=True, )
    for i in range(number_of_conformers):
        try:
            mol_block: str = Chem.MolToMolBlock(m, confId=i)
            yield MoleculeInfo.from_molblock(mol_block, smiles)
        except:
            continue


@dataclass
class Molecules:
    atom_coordinates: torch.Tensor
    atom_batches: torch.Tensor
    atom_types: torch.Tensor
    surface_coordinates: ty.Union[None, torch.Tensor] = None
    surface_normals: ty.Union[None, torch.Tensor] = None
    surface_batches: ty.Union[None, torch.Tensor] = None
    features: ty.Union[None, torch.Tensor] = None
    embedding1: ty.Union[None, torch.Tensor] = None
    embedding2: ty.Union[None, torch.Tensor] = None
    interface_pred: ty.Union[None, torch.Tensor] = None
    device: str = "cuda"
    geometricus: bool = False
    surface_geometricus_embeddings: ty.Union[None, torch.Tensor] = None
    atom_geometricus_embeddings: ty.Union[None, torch.Tensor] = None
    molecule_infos: ty.Union[None, ty.List[MoleculeInfo]] = None


    @classmethod
    def from_rdkit_mols(cls, list_of_rdkit_mols,
                        device="cuda", geometricus: bool = False,
                        random_rotation: bool = False):
        batches = list()
        atom_types = list()
        coords = list()
        b_num = 0
        molecule_infos = []
        for mol in list_of_rdkit_mols:
            molecule_info = MoleculeInfo.from_rdkit_mol(mol)
            if random_rotation:
                rot = transforms.random_rotations(1)[0].cpu().detach().numpy()
                molecule_info = molecule_info.transform_coords(rotation_matrix=rot)
            atom_names, orig_coords = molecule_info.to_atomtype_coord_pairs()
            conformer = cls.mol_to_np(atom_names, orig_coords, center=False)
            current_coords = conformer["xyz"].astype(np.float32)
            current_atomtypes = conformer["types"].astype(np.float32)
            batch = np.zeros(len(current_coords), dtype=np.float32)
            batch[:] = b_num
            batches.append(torch.tensor(batch.astype(int), device=device))
            coords.append(torch.tensor(current_coords, device=device))
            atom_types.append(torch.tensor(current_atomtypes, device=device))
            molecule_infos.append(molecule_info)
            b_num += 1
        return cls(torch.vstack(coords), torch.concat(batches), torch.vstack(atom_types), geometricus=geometricus,
    molecule_infos=molecule_infos)

    @classmethod
    def from_conformer_dataframe(cls, dataframe, device="cuda", geometricus: bool = False,
                               random_rotation: bool = False):
        mols = []
        for conformer in dataframe.conformers:
            mols.append(JSONToMols(conformer)[0])
        return cls.from_rdkit_mols(mols, device=device, geometricus=geometricus, random_rotation=random_rotation)

    @classmethod
    def from_conformer_parquet(cls, parq_file, device="cuda", geometricus: bool = False,
                               random_rotation: bool = False):
        df = pandas.read_parquet(parq_file, engine="pyarrow")
        return cls.from_conformer_dataframe(df, device=device, geometricus=geometricus, random_rotation=random_rotation)

    @classmethod
    def from_molecule_info(cls, molecule_info_list: ty.List[MoleculeInfo], random_rotation=False, device="cuda", geometricus=False):
        batches = list()
        atom_types = list()
        coords = list()
        molecule_infos = list()
        b_num = 0

        for i, molecule_info in enumerate(molecule_info_list):
            if random_rotation:
                rot = transforms.random_rotations(1)[0].cpu().detach().numpy()
                molecule_info = molecule_info.transform_coords(rotation_matrix=rot)
            atom_names, orig_coords = molecule_info.to_atomtype_coord_pairs()
            conformer = cls.mol_to_np(atom_names, orig_coords, center=False)
            current_coords = conformer["xyz"].astype(np.float32)
            current_atomtypes = conformer["types"].astype(np.float32)
            batch = np.zeros(len(current_coords), dtype=np.float32)
            batch[:] = b_num
            batches.append(torch.tensor(batch.astype(int), device=device))
            coords.append(torch.tensor(current_coords, device=device))
            atom_types.append(torch.tensor(current_atomtypes, device=device))
            molecule_infos.append(molecule_info)
            b_num += 1

        return cls(torch.vstack(coords), torch.concat(batches), torch.vstack(atom_types), geometricus=geometricus,
                   molecule_infos=molecule_infos)

    @classmethod
    def from_coords(cls, lists_of_coords, lists_of_atomnames, device="cuda", geometricus=False):
        batches = list()
        atom_types = list()
        coords = list()

        for i, coord in enumerate(lists_of_coords):
            npys = cls.mol_to_np(lists_of_atomnames[i], coord, False)
            current_coords, current_atomtypes = npys["xyz"].astype(np.float32), npys["types"].astype(np.float32)
            batch = np.zeros(len(current_coords), dtype=int)
            batch[:] = i
            batches.append(torch.tensor(batch.astype(int), device=device))
            coords.append(torch.tensor(current_coords, device=device))
            atom_types.append(torch.tensor(current_atomtypes, device=device))

        return cls(torch.vstack(coords), torch.concat(batches), torch.vstack(atom_types), geometricus=geometricus)

    @classmethod
    def from_caveat_pdb(cls, pdb_paths, sub_cavity_id="A", atom_type="H", geometricus=False, device="cuda"):
        batches = list()
        atom_types = list()
        coords = list()

        for i, pdb_path in enumerate(pdb_paths):
            coord = pd.parsePDB(pdb_path).select(f"chain {sub_cavity_id} resname SUB").getCoords()
            atom_names = coord.shape[0] * [atom_type]
            npys = cls.mol_to_np(atom_names=atom_names, coords=coord, center=False)
            current_coords, current_atomtypes = npys["xyz"].astype(np.float32), npys["types"].astype(np.float32)
            batch = np.zeros(len(current_coords), dtype=int)
            batch[:] = i
            batches.append(torch.tensor(batch.astype(int), device=device))
            coords.append(torch.tensor(current_coords, device=device))
            atom_types.append(torch.tensor(current_atomtypes, device=device))

        return cls(torch.vstack(coords), torch.concat(batches), torch.vstack(atom_types), geometricus=geometricus)

    @classmethod
    def from_npy_pair(cls, atom_coord_and_type_pairs: ty.List[ty.Tuple[Path, Path]], device="cuda"):
        batches = list()
        atom_types = list()
        coords = list()

        for i, (coords_file, atoms_file) in enumerate(atom_coord_and_type_pairs):
            current_coords = np.load(coords_file).astype(np.float32)
            current_atomtypes = np.load(atoms_file).astype(np.float32)
            batch = np.zeros(len(current_coords), dtype=int)
            batch[:] = i
            batches.append(torch.tensor(batch.astype(int), device=device))
            coords.append(torch.tensor(current_coords, device=device))
            atom_types.append(torch.tensor(current_atomtypes, device=device))

        return cls(torch.vstack(coords), torch.concat(batches), torch.vstack(atom_types))

    @classmethod
    def from_smiles_conformers(cls, smiles, number_of_conformers: int = 1, max_attempts: int = 10_000,
                            device="cuda", geometricus: bool = False, random_rotation: bool = True, addhs_in_post: bool = False, add_hs=True):
        batches = list()
        atom_types = list()
        coords = list()
        b_num = 0
        molecule_infos = []
        mols = list(get_molecule_infos_from_smiles_with_batched_conformers(smiles, number_of_conformers, addhs_in_post, add_hs, max_attempts))
        for molecule_info in mols:
            if random_rotation:
                rot = transforms.random_rotations(1)[0].cpu().detach().numpy()
                molecule_info = molecule_info.transform_coords(rotation_matrix=rot)
            atom_names, orig_coords = molecule_info.to_atomtype_coord_pairs()
            conformer = cls.mol_to_np(atom_names, orig_coords, center=False)
            current_coords = conformer["xyz"].astype(np.float32)
            current_atomtypes = conformer["types"].astype(np.float32)
            batch = np.zeros(len(current_coords), dtype=np.float32)
            batch[:] = b_num
            batches.append(torch.tensor(batch.astype(int), device=device))
            coords.append(torch.tensor(current_coords, device=device))
            atom_types.append(torch.tensor(current_atomtypes, device=device))
            molecule_infos.append(molecule_info)
            b_num += 1
        return cls(torch.vstack(coords), torch.concat(batches), torch.vstack(atom_types), geometricus=geometricus,
                   molecule_infos=molecule_infos)

    @classmethod
    def from_list_of_smiles(cls, multi_smiles, number_of_conformers: int = 1, optimize: bool = True,
                            device="cuda", geometricus: bool = False, random_rotation: bool = True, addhs_in_post: bool = False, add_hs=True):
        batches = list()
        atom_types = list()
        coords = list()
        b_num = 0
        molecule_infos = []
        for smiles in multi_smiles:
            for _ in range(number_of_conformers):
                molecule_info = MoleculeInfo.from_smiles(smiles, optimize=optimize, addhs_in_post=addhs_in_post, add_hs=add_hs)
                if random_rotation:
                    rot = transforms.random_rotations(1)[0].cpu().detach().numpy()
                    molecule_info = molecule_info.transform_coords(rotation_matrix=rot)
                atom_names, orig_coords = molecule_info.to_atomtype_coord_pairs()
                conformer = cls.mol_to_np(atom_names, orig_coords, center=False)
                current_coords = conformer["xyz"].astype(np.float32)
                current_atomtypes = conformer["types"].astype(np.float32)
                batch = np.zeros(len(current_coords), dtype=np.float32)
                batch[:] = b_num
                batches.append(torch.tensor(batch.astype(int), device=device))
                coords.append(torch.tensor(current_coords, device=device))
                atom_types.append(torch.tensor(current_atomtypes, device=device))
                molecule_infos.append(molecule_info)
                b_num += 1
        return cls(torch.vstack(coords), torch.concat(batches), torch.vstack(atom_types), geometricus=geometricus,
    molecule_infos=molecule_infos)

    @classmethod
    def from_list_of_sdf_files(cls, sdf_files, number_of_conformers: int = 1,
                               device="cuda", geometricus: bool = False, random_rotation: bool = False, add_hs=True):
        batches = list()
        atom_types = list()
        coords = list()
        b_num = 0
        molecule_infos = []
        for sdf_file in sdf_files:
            for _ in range(number_of_conformers):
                molecule_info = MoleculeInfo.from_sdf(sdf_file, add_hs=add_hs)
                if random_rotation:
                    rot = transforms.random_rotations(1)[0].cpu().detach().numpy()
                    molecule_info = molecule_info.transform_coords(rotation_matrix=rot)
                atom_names, orig_coords = molecule_info.to_atomtype_coord_pairs()
                conformer = cls.mol_to_np(atom_names, orig_coords, center=False)
                current_coords = conformer["xyz"].astype(np.float32)
                current_atomtypes = conformer["types"].astype(np.float32)
                batch = np.zeros(len(current_coords), dtype=np.float32)
                batch[:] = b_num
                batches.append(torch.tensor(batch.astype(int), device=device))
                coords.append(torch.tensor(current_coords, device=device))
                atom_types.append(torch.tensor(current_atomtypes, device=device))
                molecule_infos.append(molecule_info)
                b_num += 1
        return cls(torch.vstack(coords), torch.concat(batches), torch.vstack(atom_types), geometricus=geometricus, molecule_infos=molecule_infos)

    @staticmethod
    def mol_to_np(atom_names, coords, center):
        types_array = np.zeros((len(atom_names), len(set(list(ELE2NUM.values())))))
        for i, name in enumerate(atom_names):
            if name in ELE2NUM:
                types_array[i, ELE2NUM[name]] = 1.
            else:
                types_array[i, ELE2NUM["other"]] = 1.
        if center:
            coords = coords - np.mean(coords, axis=0, keepdims=True)
        return {"xyz": coords, "types": types_array}

    def update_surface_points_and_normals(self, resolution: float = .5, sup_sampling: int = 100, nits: int = 6,
                                          distance: float = 1.05):
        self.surface_coordinates, self.surface_normals, self.surface_batches = atoms_to_points_normals(
            self.atom_coordinates,
            self.atom_batches,
            atomtypes=self.atom_types,
            resolution=resolution,
            sup_sampling=sup_sampling,
            nits=nits,
            distance=distance
        )

    def surface_points_and_normals_to_numpy(self, resolution: float = .9, sup_sampling: int = 100, nits: int = 6,
                                            distance: float = .9):
        self.update_surface_points_and_normals(resolution=resolution, sup_sampling=sup_sampling, nits=nits, distance=distance)
        return self.surface_coordinates.cpu().detach().numpy(), self.surface_normals.cpu().detach().numpy(), self.surface_batches.cpu().detach().numpy()

    def update_dmasif_features(self, atomnet: AtomNet_MP, curvature_scales: ty.List[int] = [1, 2, 4, 8, 10],
                               force_surface_gen: bool = False):
        curvatures = self.estimate_curvatures(curvature_scales)
        chem_features = self.get_chem_features(atomnet)
        self.features = curvatures  # torch.cat([curvatures, chem_features], dim=1).contiguous()

    def get_chem_features(self, atomnet_model: AtomNet_MP):
        assert self.surface_coordinates is not None
        return atomnet_model(
            self.surface_coordinates, self.atom_coordinates, self.atom_types, self.surface_batches, self.atom_batches
        )

    def estimate_curvatures(self, curvature_scales: float):
        assert self.surface_coordinates is not None
        return curvatures(
            self.surface_coordinates,
            triangles=None,
            normals=self.surface_normals,
            scales=curvature_scales,
            batch=self.surface_batches
        )

    def update_vanilla_features(
            self,
            atomnet_model: AtomNet_MP,
            curvature_scales: ty.List[float],
            resolution: float = .5,
            sup_sampling: int = 100,
            nits: int = 6,
            distance: float = 1.05,
            force_point_computation: bool = False):
        if (self.surface_coordinates is None) or force_point_computation:
            self.update_surface_points_and_normals(
                resolution, sup_sampling, nits, distance
            )
        chem_features = self.get_chem_features(atomnet_model)
        curvatures = self.estimate_curvatures(curvature_scales)
        self.features = torch.cat([curvatures, chem_features], dim=1).contiguous()

    def get_geom_and_chem_features(
            self,
            atomnet_model: AtomNet_MP,
            force: bool = False,
            radius: float = 8, ):
        assert self.surface_coordinates is not None
        if (self.surface_geometricus_embeddings is None) or force:
            self.update_geometricus_features(radius=radius)
        chem_features = self.get_chem_features(atomnet_model)
        # self.update_geometricus_features(radius=radius)
        return torch.cat([chem_features, self.surface_geometricus_embeddings], dim=1).contiguous()


    def get_geom_and_curve_features(
            self,
            curvature_scales: ty.List[float] = [1., 2., 4., 8., 10.],
            force: bool = False,
            radius: float = 8):
        assert self.surface_coordinates is not None
        if (self.surface_geometricus_embeddings is None) or force:
            self.update_geometricus_features(radius=radius)
        curve_features = self.estimate_curvatures(
            curvature_scales
        )
        return torch.cat([curve_features, self.surface_geometricus_embeddings], dim=1).contiguous()

    @property
    def padded_sorted_embedding1_with_zeromask(self):
        variable_sized_sequences = batches_to_variable_length(self.embedding1, self.surface_batches)
        return pad_sequence(variable_sized_sequences, batch_first=True)

    def to_cuda(self):
        self.atom_coordinates = self.atom_coordinates.cuda()
        self.atom_batches = self.atom_batches.cuda()
        self.atom_types = self.atom_types.cuda()
        if self.surface_geometricus_embeddings is not None:
            self.surface_geometricus_embeddings = self.surface_geometricus_embeddings.cuda()

    def update_geometricus_features(self, radius: float = 8.):
        self.surface_geometricus_embeddings = None

    def update_per_atom_invariants(self, radius: float = 10.):
        self.atom_geometricus_embeddings = None

    def get_per_atom_invariants(self, radius: float = 10.):
        if self.atom_geometricus_embeddings is None:
            self.update_per_atom_invariants(radius=radius)
        return self.atom_geometricus_embeddings

    def plot(self, molid, color=[1, 0, 0.706], return_as_obj=False, type_="surface"):
        if type_ == "surface":
            surface_vec = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(
                self.surface_coordinates[self.surface_batches == molid].cpu().detach().numpy().reshape((-1, 3))
            ))

        else:
            surface_vec = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(
                self.atom_coordinates[self.atom_batches == molid].cpu().detach().numpy().reshape((-1, 3))
            ))
        surface_vec.paint_uniform_color(color)

        surface = o3d.geometry.PointCloud(
            surface_vec
        )

        if not return_as_obj:
            o3d.visualization.draw_geometries([surface])
        else:
            return surface

    def query_surface_vs_atoms(self, k=3):  # this will obtain the indices for matching atoms to a surface point
        from pytorch3d.ops import knn_points, knn_gather
        padded_atoms = padded_with_zeromask(self.atom_coordinates, self.atom_batches)
        atom_lengths = torch.tensor(
            [self.atom_batches[self.atom_batches == i].shape[0] for i in range(self.atom_batches.max() + 1)]).cuda()
        padded_atomtypes = padded_with_zeromask(self.atom_types, self.atom_batches)

        padded_surface = padded_with_zeromask(self.surface_coordinates, self.surface_batches)
        surface_lengths = torch.tensor([self.surface_batches[self.surface_batches == i].shape[0] for i in
                                        range(self.surface_batches.max() + 1)]).cuda()
        _, idx, _ = knn_points(padded_surface, padded_atoms, surface_lengths, atom_lengths, K=k)
        return knn_gather(padded_atomtypes, idx, lengths=atom_lengths)

    def pull_atom_types_to_surface(
            self, ):  # this will use the query_surface_vs_atoms to add all matching atomtypes to the surface
        pass


def molecule_to_geomvariant_molecules(molecules: Molecules, molid: int, num_mols: int = 10,
                                      granularity: ty.List[int] = [8], magnitude: ty.List[int] = [50]):
    assert molid <= molecules.atom_batches.max()
    device = molecules.device
    surface_coords = molecules.surface_coordinates[molecules.surface_batches == molid].cpu().detach().numpy()
    atom_coords = molecules.atom_coordinates[molecules.atom_batches == molid].cpu().detach().numpy()
    atom_types = molecules.atom_types[molecules.atom_batches == molid].cpu().detach().numpy()

    batches = list()
    atom_types_all = [torch.tensor(atom_types.copy().astype(np.float32), device=device) for x in range(num_mols)]
    distort = ElasticDistortion(granularity=granularity, magnitude=magnitude)
    _, coords = distort.iterative(surface_coords, atom_coords, num_mols)
    coords = [torch.tensor(x.astype(np.float32), device=device) for x in coords]

    for i in range(num_mols):
        batch = np.zeros(len(coords[i]), dtype=int)
        batch[:] = i
        batches.append(torch.tensor(batch.astype(int), device=device))
    new_mols = Molecules(torch.vstack(coords), torch.concat(batches), torch.vstack(atom_types_all))
    new_mols.update_surface_points_and_normals(resolution=1.5, sup_sampling=200, distance=.9)
    return new_mols


@dataclass
class ContrastiveDataset:
    similar_1: Molecules
    similar_2: Molecules
    distant: Molecules
    geometric_distances_positive: ty.Union[torch.Tensor, None] = None
    geometric_distances_negative: ty.Union[torch.Tensor, None] = None

    @classmethod
    def from_smiles_atom_based(cls, molecules: Molecules, smiles_id: int, number_of_times=32, device="cuda",
                               distant_gran_mag=(10, 50), close_gran_mag=(10, 25)):
        similar_1 = molecule_to_geomvariant_molecules(molecules, smiles_id, num_mols=number_of_times, granularity=[1],
                                                      magnitude=[1])
        similar_2 = molecule_to_geomvariant_molecules(molecules, smiles_id, num_mols=number_of_times,
                                                      granularity=[close_gran_mag[0]], magnitude=[close_gran_mag[1]])
        distant = molecule_to_geomvariant_molecules(molecules, smiles_id, num_mols=number_of_times,
                                                    granularity=[distant_gran_mag[0]], magnitude=[distant_gran_mag[1]])
        return cls(similar_1, similar_2, distant)

    @classmethod
    def from_smiles_negative_unoptimized(
            cls, list_of_smiles, number_of_conformers, device="cuda",
            precompute_distances=False, resolution=1.5, distance=.9, sup_sampling=200,
            voxel_size=.6, var=1, geometricus_radius=8, granularity=[6, 7, 8], magnitude=[5, 5, 5]):

        var_resolution = np.random.choice(np.linspace(resolution - .1 * var, resolution + .1 * var, 20))
        var_sup_sampling = np.random.choice(np.linspace(sup_sampling - 10 * var, sup_sampling + 10 * var, 20))
        similar_1 = Molecules.from_list_of_smiles(list_of_smiles, number_of_conformers, optimize=True, device=device)
        similar_1.update_surface_points_and_normals(resolution=var_resolution, distance=distance,
                                                    sup_sampling=int(var_sup_sampling))
        similar_1.update_geometricus_features(radius=geometricus_radius)

        # create distorsion function
        distort = ElasticDistortion(granularity=granularity, magnitude=magnitude)

        # s
        similar_2 = Molecules.from_list_of_smiles(list_of_smiles, number_of_conformers, optimize=False, device=device)

        similar_2.atom_coordinates = similar_1.atom_coordinates
        similar_2.atom_types = similar_1.atom_types
        similar_2.atom_batches = similar_1.atom_batches

        var_resolution2 = np.random.choice(np.linspace(resolution - .1 * var, resolution + .1 * var, 20))
        var_sup_sampling2 = np.random.choice(np.linspace(sup_sampling - 10 * var, sup_sampling + 10 * var, 20))
        similar_2.update_surface_points_and_normals(resolution=var_resolution2, distance=distance,
                                                    sup_sampling=int(var_sup_sampling2))

        distorted_surface_coords = []
        distorted_atom_coords = []

        for i in range(similar_2.surface_batches.max() + 1):
            dist_surface, dist_atoms = distort(
                similar_2.surface_coordinates[similar_2.surface_batches == i].cpu().detach().numpy().reshape(
                    (-1, 3)).astype(np.float32),
                similar_2.atom_coordinates[similar_2.atom_batches == i].cpu().detach().numpy().reshape((-1, 3)).astype(
                    np.float32)
            )
            distorted_surface_coords.append(torch.tensor(dist_surface, device=similar_2.device))
            distorted_atom_coords.append(torch.tensor(dist_atoms, device=similar_2.device))
        similar_2.surface_coordinates = torch.vstack(distorted_surface_coords)
        similar_2.atom_coordinates = torch.vstack(distorted_atom_coords)

        similar_2.update_geometricus_features(radius=geometricus_radius)

        distant = Molecules.from_list_of_smiles(list_of_smiles, number_of_conformers, optimize=False, device=device)

        rand_suffle_idx = torch.randperm(distant.atom_types.shape[0])
        distant.update_surface_points_and_normals(resolution=var_resolution, distance=distance,
                                                  sup_sampling=int(var_sup_sampling))
        distant.update_geometricus_features(radius=geometricus_radius)
        distant.atom_types = distant.atom_types[rand_suffle_idx]

        dataset = cls(similar_1, similar_2, distant)
        if precompute_distances:
            dataset.update_ransac_distances(voxel_size=voxel_size)
        return dataset

    @property
    def data(self):
        return self.similar_1, self.similar_2, self.distant

    def update_ransac_distances(self, voxel_size=.6):
        positive_distances = []
        negative_distances = []
        for i in range(int(self.similar_1.surface_batches.max() + 1)):
            positive_distances.append(
                1 - ransac_registration(
                    self.similar_1.surface_coordinates[
                        self.similar_1.surface_batches == i].cpu().detach().numpy().reshape((-1, 3)),
                    self.similar_2.surface_coordinates[
                        self.similar_2.surface_batches == i].cpu().detach().numpy().reshape((-1, 3)),
                    voxel_size=voxel_size
                ))
            negative_distances.append(
                1 - ransac_registration(
                    self.similar_1.surface_coordinates[
                        self.similar_1.surface_batches == i].cpu().detach().numpy().reshape((-1, 3)),
                    self.distant.surface_coordinates[self.distant.surface_batches == i].cpu().detach().numpy().reshape(
                        (-1, 3)),
                    voxel_size=voxel_size
                ))
        self.geometric_distances_positive = torch.tensor(np.array(positive_distances).astype(np.float32),
                                                         device=self.similar_1.device)
        self.geometric_distances_negative = torch.tensor(np.array(negative_distances).astype(np.float32),
                                                         device=self.similar_1.device)


def get_geomvariant_batches(molecules: Molecules, number_of_samples: int, batch_no: int,
                            similar_gran_mag: ty.Tuple[int, int] = (8, 25),
                            distant_gran_mag: ty.Tuple[int, int] = (8, 50)):
    pass


def molecule_to_chemvariant_molecules(molecules: Molecules, molid: int, num_mols: int = 10):
    assert molid <= molecules.atom_batches.max()
    device = molecules.device
    atom_coords = molecules.atom_coordinates[molecules.atom_batches == molid].cpu().detach().numpy()
    atom_types = molecules.atom_types[molecules.atom_batches == molid].cpu().detach().numpy()

    batches = list()
    atom_types_all = list()
    coords = list()

    for i, ratio in enumerate(np.linspace(0., 1., num_mols)):
        if i == 0:
            current_atom_types = atom_types.copy()
        else:
            current_atom_types = atom_types.copy()
            permuted_idx_from = torch.randperm(int(current_atom_types.shape[0] * ratio))
            permuted_idx_to = torch.randperm(int(current_atom_types.shape[0] * ratio))
            current_atom_types[permuted_idx_to] = current_atom_types[permuted_idx_from]

        batch = np.zeros(len(current_atom_types), dtype=int)
        batch[:] = i
        batches.append(torch.tensor(batch.astype(int), device=device))
        coords.append(torch.tensor(atom_coords.astype(np.float32), device=device))
        atom_types_all.append(torch.tensor(current_atom_types.copy().astype(np.float32), device=device))

    new_mols = Molecules(torch.vstack(coords), torch.concat(batches), torch.vstack(atom_types_all))
    new_mols.update_surface_points_and_normals(resolution=1.5, sup_sampling=200, distance=.9)
    return new_mols


def crippen_score_from_aligned(mol1, mol2, iterations=0):
    return p2p_alignment(mol1, mol2, iterations=iterations)[0]


def p2p_alignment(mol_probe, mol_ref, iterations=50):
    crippenO3A = Chem.rdMolAlign.GetCrippenO3A(
        mol_probe, mol_ref, maxIters=iterations
    )
    crippenO3A.Align()
    rmsd = crippenO3A.Trans()[0]
    trans_mat = crippenO3A.Trans()[1]
    score = crippenO3A.Score()
    return [score, rmsd, trans_mat]

def crippen_multi(sdf_file, smiles, num_conf=50, add_hs=True, iterations=50, max_attempts=10_000):
    with Chem.SDMolSupplier(sdf_file) as suppl:
        for sdf_mol in suppl:
            if sdf_mol is None:
                continue
            sdf_mol.GetNumAtoms()
    if add_hs:
        sdf_mol = Chem.AddHs(sdf_mol, addCoords=True)
    for i in range(num_conf):
        smiles_mol = Chem.MolFromSmiles(smiles)

        if add_hs:
            smiles_mol = Chem.AddHs(smiles_mol, addCoords=True)

        AllChem.EmbedMolecule(
            smiles_mol,
            maxAttempts=max_attempts,
            useRandomCoords=True
        )
        # AllChem.EmbedMultipleConfs(m, 1)

        AllChem.EmbedMultipleConfs(
            smiles_mol,
            numConfs=1,
            maxAttempts=max_attempts,
            pruneRmsThresh=.1,
            useExpTorsionAnglePrefs=True,
            useBasicKnowledge=True,
            enforceChirality=True,
            numThreads=-1)

        (score, rmsd, trans) = p2p_alignment(smiles_mol, sdf_mol, iterations=iterations)
        Chem.rdMolTransforms.TransformConformer(smiles_mol.GetConformer(0), trans)
        molinfo = MoleculeInfo.from_molblock(Chem.MolToMolBlock(smiles_mol), smiles)
        yield (score, rmsd, trans), molinfo