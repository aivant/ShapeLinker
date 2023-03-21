from shape_alignment import models, molecule
import torch
from shape_alignment.molecule import Molecules, MoleculeInfo
from shape_alignment.loss import chamfer_distance as cmf
from tqdm.notebook import tqdm
from shape_alignment.dmasif.mol_model import get_atom_features
from rdkit import Chem
from rdkit import RDLogger
from shape_alignment.loss import chamfer_distance

lg = RDLogger.logger()
lg.setLevel(RDLogger.CRITICAL)
import numpy as np
from shape_alignment.models import PCRSingleMasked


def padded_with_zeromask(point_features, surface_batches):
    variable_sized_sequences = molecule.batches_to_variable_length(point_features, surface_batches)
    return molecule.pad_sequence(variable_sized_sequences, batch_first=True)


def get_diff_molecules_for_training_multibatch(smiles1, smiles2, resolution=.9, distance=.9, batch_size=32,
                                               batch_num=200):
    molecules: Molecules = Molecules.from_list_of_smiles([smiles1] * batch_size, 1, optimize=True)
    molecules.update_surface_points_and_normals(resolution=resolution, distance=distance)
    molecules.update_geometricus_features(radius=9)

    molecules2: Molecules = Molecules.from_list_of_smiles([smiles2] * batch_size, 1, optimize=True)
    molecules2.update_surface_points_and_normals(resolution=resolution, distance=distance)
    molecules2.update_geometricus_features(radius=9)

    assert molecules.surface_coordinates is not None
    surface_coords: torch.Tensor = molecules.surface_coordinates
    assert molecules.surface_geometricus_embeddings is not None
    rest_of_features1: torch.Tensor = molecules.surface_geometricus_embeddings
    molecules.update_geometricus_features(radius=7)
    rest_of_features2 = molecules.surface_geometricus_embeddings
    padded_surface_coords = padded_with_zeromask(surface_coords, molecules.surface_batches)
    padded_rest_of_features1 = padded_with_zeromask(rest_of_features1, molecules.surface_batches)
    padded_rest_of_features2 = padded_with_zeromask(rest_of_features2, molecules.surface_batches)

    assert molecules2.surface_coordinates is not None
    surface_coords2: torch.Tensor = molecules2.surface_coordinates
    assert molecules2.surface_geometricus_embeddings is not None
    rest_of_features1_2: torch.Tensor = molecules2.surface_geometricus_embeddings
    molecules2.update_geometricus_features(radius=7)
    rest_of_features2_2 = molecules2.surface_geometricus_embeddings
    padded_surface_coords2 = padded_with_zeromask(surface_coords2, molecules2.surface_batches)
    padded_rest_of_features1_2 = padded_with_zeromask(rest_of_features1_2, molecules2.surface_batches)
    padded_rest_of_features2_2 = padded_with_zeromask(rest_of_features2_2, molecules2.surface_batches)

    batches = []
    for i in range(batch_num):
        # translation_matrices = torch.randn((batch_size, 3)).cuda() * 10
        rotation_matrices = models.transforms.random_rotations(batch_size).cuda()
        padded_transformed_coords = (padded_surface_coords2 @ rotation_matrices)
        original_with_features = torch.cat((padded_surface_coords, padded_rest_of_features1, padded_rest_of_features2),
                                           dim=2)
        transformed_with_features = torch.cat(
            (padded_transformed_coords, padded_rest_of_features1_2, padded_rest_of_features2_2), dim=2)
        batches.append((original_with_features.cuda(), transformed_with_features.cuda(), rotation_matrices.cuda()))
    return batches


def get_diff_molecules_for_training_multibatch_coords(smiles1, smiles2, resolution=.9, distance=.9, batch_size=32,
                                                      batch_num=200):
    molecules: Molecules = Molecules.from_list_of_smiles([smiles1] * batch_size, 1, optimize=True)
    molecules.update_surface_points_and_normals(resolution=resolution, distance=distance)

    molecules2: Molecules = Molecules.from_list_of_smiles([smiles2] * batch_size, 1, optimize=True)
    molecules2.update_surface_points_and_normals(resolution=resolution, distance=distance)

    assert molecules.surface_coordinates is not None
    surface_coords: torch.Tensor = molecules.surface_coordinates
    rest_of_features2 = molecules.surface_geometricus_embeddings
    padded_surface_coords = padded_with_zeromask(surface_coords, molecules.surface_batches)

    assert molecules2.surface_coordinates is not None
    surface_coords2: torch.Tensor = molecules2.surface_coordinates
    padded_surface_coords2 = padded_with_zeromask(surface_coords2, molecules2.surface_batches)

    batches = []
    for i in range(batch_num):
        translation_matrices = torch.randn((batch_size, 1, 3)).cuda() * 100
        rotation_matrices = models.transforms.random_rotations(batch_size).cuda()
        padded_transformed_coords = (padded_surface_coords2 @ rotation_matrices) + translation_matrices
        original_with_features = padded_surface_coords
        transformed_with_features = padded_transformed_coords
        batches.append((original_with_features.cuda(), transformed_with_features.cuda(), rotation_matrices.cuda()))
    return batches


def get_diff_molecules_for_training_multibatch_atom_types(smiles1, smiles2, resolution=.9, distance=.9, batch_size=32,
                                                          batch_num=200, k=3):
    molecules: Molecules = Molecules.from_list_of_smiles([smiles1] * batch_size, 1, optimize=True)
    molecules.update_surface_points_and_normals(resolution=resolution, distance=distance)

    molecules2: Molecules = Molecules.from_list_of_smiles([smiles2] * batch_size, 1, optimize=True)
    molecules2.update_surface_points_and_normals(resolution=resolution, distance=distance)

    assert molecules.surface_coordinates is not None
    surface_coords: torch.Tensor = molecules.surface_coordinates
    padded_surface_coords = padded_with_zeromask(surface_coords, molecules.surface_batches)
    feats1 = molecules.query_surface_vs_atoms(k=k).mean(2)

    assert molecules2.surface_coordinates is not None
    surface_coords2: torch.Tensor = molecules2.surface_coordinates
    padded_surface_coords2 = padded_with_zeromask(surface_coords2, molecules2.surface_batches)
    feats2 = molecules2.query_surface_vs_atoms(k=k).mean(2)

    batches = []
    for i in range(batch_num):
        translation_matrices = torch.randn((batch_size, 1, 3)).cuda() * 100
        rotation_matrices = models.transforms.random_rotations(batch_size).cuda()
        padded_transformed_coords = (padded_surface_coords2 @ rotation_matrices) + translation_matrices
        original_with_features = torch.cat((padded_surface_coords, feats1), dim=2)
        transformed_with_features = torch.cat((padded_transformed_coords, feats2), dim=2)
        batches.append((original_with_features.cuda(), transformed_with_features.cuda(), rotation_matrices.cuda()))
    return batches


def get_diff_molecules_for_training_multibatch_with_atom_coords(smiles1, smiles2, resolution=.9, distance=.9,
                                                                batch_size=32, batch_num=200):
    molecules: Molecules = Molecules.from_list_of_smiles([smiles1] * batch_size, 1, optimize=True)
    molecules.update_surface_points_and_normals(resolution=resolution, distance=distance)

    molecules2: Molecules = Molecules.from_list_of_smiles([smiles2] * batch_size, 1, optimize=True)
    molecules2.update_surface_points_and_normals(resolution=resolution, distance=distance)

    assert molecules.surface_coordinates is not None
    surface_coords: torch.Tensor = molecules.surface_coordinates
    atom_coords = molecules.atom_coordinates
    padded_atom_coords = padded_with_zeromask(atom_coords, molecules.atom_batches)
    padded_surface_coords = padded_with_zeromask(surface_coords, molecules.surface_batches)

    assert molecules2.surface_coordinates is not None
    surface_coords2: torch.Tensor = molecules2.surface_coordinates
    atom_coords2 = molecules2.atom_coordinates
    padded_atom_coords2 = padded_with_zeromask(atom_coords2, molecules2.atom_batches)
    padded_surface_coords2 = padded_with_zeromask(surface_coords2, molecules2.surface_batches)

    batches = []
    for i in range(batch_num):
        translation_matrices = torch.randn((batch_size, 1, 3)).cuda() * 100
        rotation_matrices = models.transforms.random_rotations(batch_size).cuda()
        padded_transformed_coords = (padded_surface_coords2 @ rotation_matrices) + translation_matrices
        padded_transformed_atom_coords = (padded_atom_coords2 @ rotation_matrices) + translation_matrices
        original_with_features = padded_surface_coords
        transformed_with_features = padded_transformed_coords
        batches.append((original_with_features.cuda(), padded_atom_coords.cuda(), transformed_with_features.cuda(),
                        padded_transformed_atom_coords.cuda()))
    return batches


def rearrange_batches_and_apply_padding(batch_x, batch_y, batch_size):
    assert len(batch_x) == len(batch_x)
    assert batch_x[0].shape[0] == batch_y[0].shape[0]
    assert (len(batch_x) * batch_x[0].shape[0] % batch_size) == 0

    # first find the max of both batches
    batch_x_max = max([x.shape[1] for x in batch_x])
    batch_y_max = max([y.shape[1] for y in batch_y])

    # create empty
    numkeys = len(batch_x) * batch_x[0].shape[0]
    new_batch_x_stacked = torch.zeros((numkeys, batch_x_max, batch_x[0].shape[2]), dtype=torch.float32, device="cpu")
    new_batch_y_stacked = torch.zeros((numkeys, batch_y_max, batch_y[0].shape[2]), dtype=torch.float32, device="cpu")

    # fill in empty batch by batch with recording lengths
    lengths_x = torch.zeros(numkeys, dtype=torch.float32)
    lengths_y = torch.zeros(numkeys, dtype=torch.float32)
    old_batch_size = batch_x[0].shape[0]
    for i, (bx, by) in enumerate(zip(batch_x, batch_y)):
        new_batch_x_stacked[int(i * old_batch_size):int(i * old_batch_size + old_batch_size), :bx.shape[1], :] = bx[:]
        new_batch_y_stacked[int(i * old_batch_size):int(i * old_batch_size + old_batch_size), :by.shape[1], :] = by[:]
        lengths_x[int(i * old_batch_size):int(i * old_batch_size + old_batch_size)] = bx.shape[1]
        lengths_y[int(i * old_batch_size):int(i * old_batch_size + old_batch_size)] = by.shape[1]

    # shuffle and apply shuffled idx to lengths and the other batch
    indices = torch.randperm(new_batch_x_stacked.shape[0])
    new_batch_x_stacked = new_batch_x_stacked[indices]
    new_batch_y_stacked = new_batch_y_stacked[indices]
    lengths_x = lengths_x[indices]
    lengths_y = lengths_y[indices]

    # return in set batches
    new_batch_x = [new_batch_x_stacked[x: x + batch_size] for x in range(0, new_batch_x_stacked.shape[0], batch_size)]
    batch_length_x = [lengths_x[x: x + batch_size] for x in range(0, new_batch_x_stacked.shape[0], batch_size)]

    new_batch_y = [new_batch_y_stacked[x: x + batch_size] for x in range(0, new_batch_x_stacked.shape[0], batch_size)]
    batch_length_y = [lengths_y[x: x + batch_size] for x in range(0, new_batch_y_stacked.shape[0], batch_size)]

    return [(new_batch_x[i], new_batch_y[i], batch_length_x[i], batch_length_y[i]) for i in range(len(new_batch_x))]


def get_diff_single_sdf_vs_molecules_for_training_multibatch_coords(sdf_file_path, smiles2, resolution=.9, distance=.9,
                                                                    batch_size=32, batch_num=200, mol2=False):
    if mol2:
        molecules = Molecules.from_list_of_mol2([sdf_file_path] * batch_size)
    else:
        molecules: Molecules = Molecules.from_list_of_sdf_files([sdf_file_path] * batch_size)
    molecules.update_surface_points_and_normals(resolution=resolution, distance=distance)

    molecules2: Molecules = Molecules.from_list_of_smiles([smiles2] * batch_size, 1, optimize=True)
    molecules2.update_surface_points_and_normals(resolution=resolution, distance=distance)

    assert molecules.surface_coordinates is not None
    surface_coords: torch.Tensor = molecules.surface_coordinates
    rest_of_features2 = molecules.surface_geometricus_embeddings
    padded_surface_coords = padded_with_zeromask(surface_coords, molecules.surface_batches)
    smallest_batch_x = min([molecules.surface_batches[molecules.surface_batches == i].shape[0] for i in
                            range(int(molecules.surface_batches.max() + 1))])
    padded_surface_coords = padded_surface_coords[:, :smallest_batch_x, :]

    assert molecules2.surface_coordinates is not None
    surface_coords2: torch.Tensor = molecules2.surface_coordinates
    padded_surface_coords2 = padded_with_zeromask(surface_coords2, molecules2.surface_batches)

    batches = []
    for i in range(batch_num):
        translation_matrices = torch.randn((batch_size, 1, 3)).cuda() * 100
        rotation_matrices = models.transforms.random_rotations(batch_size).cuda()
        padded_transformed_coords = (padded_surface_coords2 @ rotation_matrices) + translation_matrices
        original_with_features = padded_surface_coords
        transformed_with_features = padded_transformed_coords
        smallest_batch_y = min([molecules2.surface_batches[molecules2.surface_batches == i].shape[0] for i in
                                range(int(molecules2.surface_batches.max() + 1))])
        transformed_with_features = transformed_with_features[:, :smallest_batch_y,
                                    :]  # add something to adjust for 0s at the end
        batches.append((original_with_features.cuda(), transformed_with_features.cuda(), rotation_matrices.cuda()))
    return batches





def get_diff_single_sdf_vs_sdf_for_training_multibatch_coords(sdf_file_path, sdf_file_path2, resolution=.9, distance=.9,
                                                                    batch_size=32, batch_num=200, mol2=False):
    if mol2:
        molecules = Molecules.from_list_of_mol2([sdf_file_path] * batch_size)
    else:
        molecules: Molecules = Molecules.from_list_of_sdf_files([sdf_file_path] * batch_size)
    molecules.update_surface_points_and_normals(resolution=resolution, distance=distance)

    molecules2: Molecules = Molecules.from_list_of_sdf_files([sdf_file_path2] * batch_size, 1, random_rotation=True)
    molecules2.update_surface_points_and_normals(resolution=resolution, distance=distance)

    assert molecules.surface_coordinates is not None
    surface_coords: torch.Tensor = molecules.surface_coordinates
    rest_of_features2 = molecules.surface_geometricus_embeddings
    padded_surface_coords = padded_with_zeromask(surface_coords, molecules.surface_batches)
    smallest_batch_x = min([molecules.surface_batches[molecules.surface_batches == i].shape[0] for i in
                            range(int(molecules.surface_batches.max() + 1))])
    padded_surface_coords = padded_surface_coords[:, :smallest_batch_x, :]

    assert molecules2.surface_coordinates is not None
    surface_coords2: torch.Tensor = molecules2.surface_coordinates
    padded_surface_coords2 = padded_with_zeromask(surface_coords2, molecules2.surface_batches)

    batches = []
    for i in range(batch_num):
        translation_matrices = torch.randn((batch_size, 1, 3)).cuda() * 100
        rotation_matrices = models.transforms.random_rotations(batch_size).cuda()
        padded_transformed_coords = (padded_surface_coords2 @ rotation_matrices) + translation_matrices
        original_with_features = padded_surface_coords
        transformed_with_features = padded_transformed_coords
        smallest_batch_y = min([molecules2.surface_batches[molecules2.surface_batches == i].shape[0] for i in
                                range(int(molecules2.surface_batches.max() + 1))])
        transformed_with_features = transformed_with_features[:, :smallest_batch_y,
        :]  # add something to adjust for 0s at the end
        batches.append((original_with_features.cuda(), transformed_with_features.cuda(), rotation_matrices.cuda()))
    return batches


def get_diff_single_smiles_vs_molecules_for_training_multibatch_coords(smiles, smiles2, resolution=.9, distance=.9,
                                                                       batch_size=32, batch_num=200, caveat=False, sub_cavity_id="A", atom_type="H"):
    if caveat:
        molecules = Molecules.from_caveat_pdb([smiles] * batch_size, sub_cavity_id=sub_cavity_id, atom_type=atom_type)
    else:
        molecules: Molecules = Molecules.from_list_of_smiles([smiles] * batch_size, optimize=True)
    molecules.update_surface_points_and_normals(resolution=resolution, distance=distance)

    molecules2: Molecules = Molecules.from_list_of_smiles([smiles2] * batch_size, optimize=True)
    molecules2.update_surface_points_and_normals(resolution=resolution, distance=distance)

    assert molecules.surface_coordinates is not None
    surface_coords: torch.Tensor = molecules.surface_coordinates
    rest_of_features2 = molecules.surface_geometricus_embeddings
    padded_surface_coords = padded_with_zeromask(surface_coords, molecules.surface_batches)
    smallest_batch_x = min([molecules.surface_batches[molecules.surface_batches == i].shape[0] for i in
                            range(int(molecules.surface_batches.max() + 1))])
    padded_surface_coords = padded_surface_coords[:, :smallest_batch_x, :]

    assert molecules2.surface_coordinates is not None
    surface_coords2: torch.Tensor = molecules2.surface_coordinates
    padded_surface_coords2 = padded_with_zeromask(surface_coords2, molecules2.surface_batches)

    batches = []
    mol_info_batches = []

    for i in range(batch_num):
        translation_matrices = torch.randn((batch_size, 1, 3)).cuda() * 100
        rotation_matrices = models.transforms.random_rotations(batch_size).cuda()
        padded_transformed_coords = padded_surface_coords2 #(padded_surface_coords2 @ rotation_matrices) + translation_matrices
        original_with_features = padded_surface_coords
        transformed_with_features = padded_transformed_coords
        smallest_batch_y = min([molecules2.surface_batches[molecules2.surface_batches == i].shape[0] for i in
                                range(int(molecules2.surface_batches.max() + 1))])
        transformed_with_features = transformed_with_features[:, :smallest_batch_y,
                                    :]  # add something to adjust for 0s at the end
        batches.append((original_with_features.cuda(), transformed_with_features.cuda(), rotation_matrices.cuda()))
        mol_info_batches.append((molecules.molecule_infos, molecules2.molecule_infos))
    return batches, mol_info_batches


def compare_to_other_mols_multiconf_mix_mols(sdf_file, other_smiles, model, smiles_per_batch=4, batch_size=128,
                                             resolution=.9, distance=.9, mol=False):
    # if not mol:
    #     sdf_mol = Molecules.from_list_of_sdf_files([sdf_file] * batch_size)
    # else:
    #     sdf_mol = Molecules.from_list_of_mol2([sdf_file] * batch_size)
    sdf_mol = Molecules.from_list_of_smiles([sdf_file] * batch_size)
    sdf_mol.update_surface_points_and_normals(resolution=resolution, distance=distance)

    surface_coords: torch.Tensor = sdf_mol.surface_coordinates
    padded_surface_coords = padded_with_zeromask(surface_coords, sdf_mol.surface_batches)
    smallest_batch_x = min([sdf_mol.surface_batches[sdf_mol.surface_batches == i].shape[0] for i in
                            range(int(sdf_mol.surface_batches.max() + 1))])
    padded_surface_coords = padded_surface_coords[:, :smallest_batch_x, :].cuda()

    chamfer_distances = []
    smiles_names = []
    for i in range(0, len(other_smiles), smiles_per_batch):
        current_smiles = other_smiles[i: i + smiles_per_batch]
        try:
            current_mols = Molecules.from_list_of_smiles(current_smiles, batch_size // smiles_per_batch, optimize=True)
        except (ValueError, RuntimeError):
            continue

        current_mols.update_surface_points_and_normals(resolution=resolution, distance=distance)
        surface_coords2: torch.Tensor = current_mols.surface_coordinates
        current_lengths = [current_mols.surface_batches[current_mols.surface_batches == i].shape[0] for i in
                           range(int(current_mols.surface_batches.max() + 1))]

        padded_surface_coords2 = padded_with_zeromask(surface_coords2, current_mols.surface_batches).cuda()
        # matrices = models.transforms.random_rotations(len(current_smiles) * (batch_size // smiles_per_batch)).cuda()
        # padded_surface_coords2 = padded_surface_coords2 @ matrices
        # smallest_batch_y = min([current_mols.surface_batches[current_mols.surface_batches == i].shape[0] for i in range(int(current_mols.surface_batches.max()+1))])
        # padded_surface_coords2 = padded_surface_coords2[:, :smallest_batch_y, :] # add something to adjust for 0s at the end

        current_mols.update_surface_points_and_normals(resolution=resolution, distance=distance)
        if padded_surface_coords2.shape[0] != batch_size:
            print("short")
            continue
        result = model.coarse(padded_surface_coords, padded_surface_coords2, current_lengths)
        smiles_idx = 0
        idx = 0
        for k in range(0, batch_size, batch_size // smiles_per_batch):
            current_batch_scores = []
            for j in range(k, k + batch_size // smiles_per_batch, 1):
                current_batch_scores.append((chamfer_distance(result[j].reshape(1, -1, 3),
                                                              current_mols.surface_coordinates[
                                                                  current_mols.surface_batches == j].reshape(1, -1, 3)[
                                                              :current_lengths[j]]))[0].item())
                idx += 1
            chamfer_distances.append(min(current_batch_scores))
            smiles_names.append(current_smiles[smiles_idx])
            smiles_idx += 1
    return chamfer_distances, smiles_names


def match_smiles_to_cavity(cavity_file, other_smiles, model: PCRSingleMasked,
                           batch_size=128, resolution=.9, distance=.9, sub_cavity_id="A", atom_type="H"):
    cavity = Molecules.from_caveat_pdb([cavity_file] * batch_size, sub_cavity_id=sub_cavity_id, atom_type=atom_type)
    cavity.update_surface_points_and_normals(resolution=resolution, distance=distance)

    surface_coords: torch.Tensor = cavity.surface_coordinates
    padded_surface_coords = padded_with_zeromask(surface_coords, cavity.surface_batches)
    smallest_batch_x = min([cavity.surface_batches[cavity.surface_batches == i].shape[0] for i in
                            range(int(cavity.surface_batches.max() + 1))])
    padded_surface_coords = padded_surface_coords[:, :smallest_batch_x, :].cuda()

    chamfer_distances = []
    smiles_names = []

    for i in tqdm(range(len(other_smiles))):
        current_smiles = [other_smiles[i]] * batch_size
        try:
            current_mols = Molecules.from_list_of_smiles(current_smiles, optimize=True)
        except (ValueError, RuntimeError):
            continue

        current_mols.update_surface_points_and_normals(resolution=resolution, distance=distance)
        surface_coords2: torch.Tensor = current_mols.surface_coordinates
        current_lengths = [current_mols.surface_batches[current_mols.surface_batches == i].shape[0] for i in
                           range(int(current_mols.surface_batches.max() + 1))]
        padded_surface_coords2 = padded_with_zeromask(surface_coords2, current_mols.surface_batches).cuda()

        if padded_surface_coords2.shape[0] != batch_size:
            print("short")
            continue

        result = model.coarse(padded_surface_coords, padded_surface_coords2, current_lengths)

        current_batch_scores = []
        for k in range(0, batch_size):
            current_batch_scores.append((chamfer_distance(result[k].reshape(1, -1, 3),
                                                          padded_surface_coords2[k].reshape(1, -1, 3)[:current_lengths[k]]))[0].item())
        argmin = np.argmin(current_batch_scores)
        chamfer_distances.append(current_batch_scores[argmin])
        smiles_names.append(current_smiles[0])
    return chamfer_distances, smiles_names

def match_query_to_db(molinfo: MoleculeInfo, other_molecules: Molecules, model: PCRSingleMasked, resolution=.9, distance=.9, device="cuda"):
    batch_size = other_molecules.atom_batches.max() + 1
    query = Molecules.from_molecule_info([molinfo] * batch_size, device=device, geometricus=False, random_rotation=False)
    query.update_surface_points_and_normals(resolution=resolution, distance=distance)

    surface_coords: torch.Tensor = query.surface_coordinates
    padded_surface_coords = padded_with_zeromask(surface_coords, query.surface_batches)
    smallest_batch_x = min([query.surface_batches[query.surface_batches == i].shape[0] for i in
                            range(int(query.surface_batches.max() + 1))])
    padded_surface_coords = padded_surface_coords[:, :smallest_batch_x, :].cuda()
    chamfer_distances = []
    other_molecules.update_surface_points_and_normals(resolution=resolution, distance=distance)
    surface_coords2: torch.Tensor = other_molecules.surface_coordinates
    current_lengths = [other_molecules.surface_batches[other_molecules.surface_batches == i].shape[0] for i in
                       range(int(other_molecules.surface_batches.max() + 1))]
    padded_surface_coords2 = padded_with_zeromask(surface_coords2, other_molecules.surface_batches).cuda()
    result = model.coarse(padded_surface_coords, padded_surface_coords2, current_lengths)
    current_batch_scores = []
    for k in range(0, batch_size):
        current_batch_scores.append((chamfer_distance(result[k].reshape(1, -1, 3),
                                                      padded_surface_coords2[k].reshape(1, -1, 3)[:current_lengths[k]]))[0].item())
    argmin = np.argmin(current_batch_scores)
    chamfer_distances.append(current_batch_scores[argmin])
    return chamfer_distances


def match_single_smiles_to_cavity(cavity_file, smiles, model: PCRSingleMasked, number_of_batches=1000,
                                  resolution=.9, distance=.9, sub_cavity_id="A", atom_type="H"):
    cavity_molecules = Molecules.from_caveat_pdb([cavity_file], sub_cavity_id=sub_cavity_id, atom_type=atom_type)
    cavity_molecules.update_surface_points_and_normals(resolution=resolution, distance=distance)

    surface_coords: torch.Tensor = cavity_molecules.surface_coordinates
    current_lengths = [cavity_molecules.surface_batches[cavity_molecules.surface_batches == i].shape[0] for i in
                       range(int(cavity_molecules.surface_batches.max() + 1))]
    padded_surface_coords = padded_with_zeromask(surface_coords, cavity_molecules.surface_batches).cuda()

    chamfer_distances = []
    transformed = []

    for i in range(number_of_batches):
        smiles_molecules = Molecules.from_list_of_smiles([smiles], optimize=True)
        smiles_molecules.update_surface_points_and_normals(resolution=resolution, distance=distance)
        surface_coords2: torch.Tensor = smiles_molecules.surface_coordinates


        padded_surface_coords2 = padded_with_zeromask(surface_coords2, smiles_molecules.surface_batches).cuda()
        smiles_molecules.update_surface_points_and_normals(resolution=resolution, distance=distance)

        result = model.coarse(padded_surface_coords2, padded_surface_coords, current_lengths)
        rotations, translations, y_translate, centroids = model.coarse.get_rottrans(padded_surface_coords2, padded_surface_coords)
        rotations, translations, y_translate, centroids = rotations.cpu().detach().numpy(), translations.cpu().detach().numpy(), y_translate.cpu().detach().numpy(), centroids.cpu().detach().numpy()

        chamfer_distances.append((chamfer_distance(result[0].reshape(1, -1, 3),
                                                   padded_surface_coords[0].reshape(1, -1, 3)))[0].item())

        transformed.append(
            smiles_molecules.molecule_infos[0].transform_coords(translation_matrix=centroids[0]).transform_coords(rotation_matrix=rotations[0], translation_matrix=translations[0] + y_translate[0])
        )

    return chamfer_distances, transformed


def match_single_smiles_to_molinfo():
    pass


def match_single_smiles_to_sdf(sdf_file, smiles, model: PCRSingleMasked, number_of_batches=1000,
                                  resolution=.9, distance=.9):
    cavity_molecules = Molecules.from_list_of_sdf_files([sdf_file])
    cavity_molecules.update_surface_points_and_normals(resolution=resolution, distance=distance)

    surface_coords: torch.Tensor = cavity_molecules.surface_coordinates
    current_lengths = [cavity_molecules.surface_batches[cavity_molecules.surface_batches == i].shape[0] for i in
                       range(int(cavity_molecules.surface_batches.max() + 1))]
    padded_surface_coords = padded_with_zeromask(surface_coords, cavity_molecules.surface_batches).cuda()

    chamfer_distances = []
    transformed = []

    for i in range(number_of_batches):
        smiles_molecules = Molecules.from_list_of_smiles([smiles], optimize=True)
        smiles_molecules.update_surface_points_and_normals(resolution=resolution, distance=distance)
        surface_coords2: torch.Tensor = smiles_molecules.surface_coordinates


        padded_surface_coords2 = padded_with_zeromask(surface_coords2, smiles_molecules.surface_batches).cuda()
        smiles_molecules.update_surface_points_and_normals(resolution=resolution, distance=distance)

        result = model.coarse(padded_surface_coords2, padded_surface_coords, current_lengths)
        rotations, translations, y_translate, centroids = model.coarse.get_rottrans(padded_surface_coords2, padded_surface_coords)
        rotations, translations, y_translate, centroids = rotations.cpu().detach().numpy(), translations.cpu().detach().numpy(), y_translate.cpu().detach().numpy(), centroids.cpu().detach().numpy()
        dist = (chamfer_distance(result[0].reshape(1, -1, 3),
                                 padded_surface_coords[0].reshape(1, -1, 3)))[0].item()
        chamfer_distances.append(dist)

        transformed.append(
            smiles_molecules.molecule_infos[0].transform_coords(translation_matrix=centroids[0]).transform_coords(rotation_matrix=rotations[0], translation_matrix=translations[0] + y_translate[0])
        )

        if dist < 1.:
            break

    return chamfer_distances, transformed

def match_single_sdf_to_sdf(sdf_file, sdf_file2, model: PCRSingleMasked, number_of_batches=1000,
                               resolution=.9, distance=.9):
    cavity_molecules = Molecules.from_list_of_sdf_files([sdf_file])
    cavity_molecules.update_surface_points_and_normals(resolution=resolution, distance=distance)

    surface_coords: torch.Tensor = cavity_molecules.surface_coordinates
    current_lengths = [cavity_molecules.surface_batches[cavity_molecules.surface_batches == i].shape[0] for i in
                       range(int(cavity_molecules.surface_batches.max() + 1))]
    padded_surface_coords = padded_with_zeromask(surface_coords, cavity_molecules.surface_batches).cuda()

    chamfer_distances = []
    transformed = []

    for i in range(number_of_batches):
        smiles_molecules = Molecules.from_list_of_sdf_files([sdf_file2], random_rotation=True)
        smiles_molecules.update_surface_points_and_normals(resolution=resolution, distance=distance)
        surface_coords2: torch.Tensor = smiles_molecules.surface_coordinates


        padded_surface_coords2 = padded_with_zeromask(surface_coords2, smiles_molecules.surface_batches).cuda()
        smiles_molecules.update_surface_points_and_normals(resolution=resolution, distance=distance)

        result = model.coarse(padded_surface_coords2, padded_surface_coords, current_lengths)
        rotations, translations, y_translate, centroids = model.coarse.get_rottrans(padded_surface_coords2, padded_surface_coords)
        rotations, translations, y_translate, centroids = rotations.cpu().detach().numpy(), translations.cpu().detach().numpy(), y_translate.cpu().detach().numpy(), centroids.cpu().detach().numpy()

        chamfer_distances.append((chamfer_distance(result[0].reshape(1, -1, 3),
                                                   padded_surface_coords[0].reshape(1, -1, 3)))[0].item())

        transformed.append(
            smiles_molecules.molecule_infos[0].transform_coords(translation_matrix=centroids[0]).transform_coords(rotation_matrix=rotations[0], translation_matrix=translations[0] + y_translate[0])
        )

    return chamfer_distances, transformed

def compare_smiles_to_other_mols_multiconf_mix_mols(smiles, other_smiles, model: PCRSingleMasked, smiles_per_batch=4,
                                                    batch_size=128, resolution=.9, distance=.9, caveat=False, sub_cavity_id="A", atom_type="H"):
    if caveat:
        sdf_mol = Molecules.from_caveat_pdb([smiles] * batch_size, sub_cavity_id=sub_cavity_id, atom_type=atom_type)
    else:
        sdf_mol = Molecules.from_list_of_smiles([smiles] * batch_size)
    sdf_mol.update_surface_points_and_normals(resolution=resolution, distance=distance)

    surface_coords: torch.Tensor = sdf_mol.surface_coordinates
    padded_surface_coords = padded_with_zeromask(surface_coords, sdf_mol.surface_batches)
    smallest_batch_x = min([sdf_mol.surface_batches[sdf_mol.surface_batches == i].shape[0] for i in
                            range(int(sdf_mol.surface_batches.max() + 1))])
    padded_surface_coords = padded_surface_coords[:, :smallest_batch_x, :].cuda()

    chamfer_distances = []
    smiles_names = []
    q_infos = []
    db_infos = []
    for i in range(0, len(other_smiles), smiles_per_batch):
        current_smiles = other_smiles[i: i + smiles_per_batch]
        try:
            current_mols = Molecules.from_list_of_smiles(current_smiles, batch_size // smiles_per_batch, optimize=True)
        except (ValueError, RuntimeError):
            continue

        current_mols.update_surface_points_and_normals(resolution=resolution, distance=distance)
        surface_coords2: torch.Tensor = current_mols.surface_coordinates
        current_lengths = [current_mols.surface_batches[current_mols.surface_batches == i].shape[0] for i in
                           range(int(current_mols.surface_batches.max() + 1))]

        padded_surface_coords2 = padded_with_zeromask(surface_coords2, current_mols.surface_batches).cuda()
        # matrices = models.transforms.random_rotations(len(current_smiles) * (batch_size // smiles_per_batch)).cuda()
        # padded_surface_coords2 = padded_surface_coords2 @ matrices

        current_mols.update_surface_points_and_normals(resolution=resolution, distance=distance)
        if padded_surface_coords2.shape[0] != batch_size:
            print("short")
            continue

        result = model.coarse(padded_surface_coords, padded_surface_coords2, current_lengths)
        rotations, translations, y_translate, centroids = model.coarse.get_rottrans(padded_surface_coords, padded_surface_coords2,
                                                                       current_lengths)
        rotations, translations, y_translate, centroids = rotations.cpu().detach().numpy(), translations.cpu().detach().numpy(), y_translate.cpu().detach().numpy(), centroids.cpu().detach().numpy()
        current_infos_db = current_mols.molecule_infos
        current_infos_qr = sdf_mol.molecule_infos
        smiles_idx = 0
        idx = 0

        for k in range(0, batch_size, batch_size // smiles_per_batch):
            current_batch_scores = []
            current_transformed_infos = []

            for qi, j in enumerate(range(k, k + batch_size // smiles_per_batch, 1)):
                current_batch_scores.append((chamfer_distance(result[j].reshape(1, -1, 3),
                                                              padded_surface_coords2[j].reshape(1, -1, 3)[:current_lengths[j]]))[0].item())
                if not caveat:
                    current_transformed_infos.append(
                        current_infos_qr[j].transform_coords(rotation_matrix=rotations[j])
                    )
                idx += 1

            argmin = np.argmin(current_batch_scores)
            chamfer_distances.append(current_batch_scores[argmin])
            if not caveat:
                q_infos.append(current_transformed_infos[argmin])
                db_infos.append(current_infos_db[argmin])
            smiles_names.append(current_smiles[smiles_idx])
            smiles_idx += 1
            print()
    return chamfer_distances, smiles_names, q_infos, db_infos


def compare_to_other_mols_multiconf(sdf_file, other_smiles, model, batch_size=32, resolution=.9, distance=.9,
                                    mol=False):
    # if not mol:
    #     sdf_mol = Molecules.from_list_of_sdf_files([sdf_file] * batch_size)
    # else:
    #     sdf_mol = Molecules.from_list_of_mol2([sdf_file] * batch_size)
    sdf_mol = Molecules.from_list_of_smiles([sdf_file] * batch_size, optimize=True)
    sdf_mol.update_surface_points_and_normals(resolution=resolution, distance=distance)

    surface_coords: torch.Tensor = sdf_mol.surface_coordinates
    rest_of_features2 = sdf_mol.surface_geometricus_embeddings
    padded_surface_coords = padded_with_zeromask(surface_coords, sdf_mol.surface_batches)
    smallest_batch_x = min([sdf_mol.surface_batches[sdf_mol.surface_batches == i].shape[0] for i in
                            range(int(sdf_mol.surface_batches.max() + 1))])
    padded_surface_coords = padded_surface_coords[:, :smallest_batch_x, :].cuda()


    chamfer_distances = []
    smiles_names = []
    q_infos = []
    db_infos = []
    for i in tqdm(range(len(other_smiles))):
        current_smiles = [other_smiles[i]]
        try:
            current_mols = Molecules.from_list_of_smiles(current_smiles, batch_size, optimize=True)
        except (ValueError, RuntimeError):
            continue
        current_mols.update_surface_points_and_normals(resolution=resolution, distance=distance)
        surface_coords2: torch.Tensor = current_mols.surface_coordinates

        padded_surface_coords2 = padded_with_zeromask(surface_coords2, current_mols.surface_batches).cuda()

        smallest_batch_y = min([current_mols.surface_batches[current_mols.surface_batches == i].shape[0] for i in
                                range(int(current_mols.surface_batches.max() + 1))])
        padded_surface_coords2 = padded_surface_coords2[:, :smallest_batch_y,
        :]  # add something to adjust for 0s at the end

        current_mols.update_surface_points_and_normals(resolution=resolution, distance=distance)
        if padded_surface_coords2.shape[0] != batch_size:
            print("short")
            continue
        result = model(padded_surface_coords, padded_surface_coords2)
        rotations, translations, centroids = model.coarse.get_rottrans(padded_surface_coords, padded_surface_coords2)
        rotations, translations, centroids = rotations.cpu().detach().numpy(), translations.cpu().detach().numpy(), centroids.cpu().detach().numpy()
        current_infos_db = current_mols.molecule_infos
        current_infos_qr = sdf_mol.molecule_infos

        current_batch_scores = []
        current_transformed_infos = []
        for j in range(batch_size):
            current_batch_scores.append((chamfer_distance(result[j].reshape(1, -1, 3), current_mols.surface_coordinates[
            current_mols.surface_batches == j].reshape(1, -1, 3)))[0].item())
            current_transformed_infos.append(
                current_infos_qr[j].transform_coords(rotations[j], translations[j] + centroids[j])
            )

        argmin = np.argmin(current_batch_scores)
        q_infos.append(current_transformed_infos[argmin])
        db_infos.append(current_infos_db[argmin])
        chamfer_distances.append(current_batch_scores[argmin])
        smiles_names.append(other_smiles[i])
    return chamfer_distances, smiles_names, q_infos, db_infos


def compare_to_other_mols_multiconf2(sdf_file, other_smiles, model, batch_size=32, resolution=.9, distance=.9):
    sdf_mol = Molecules.from_list_of_smiles([sdf_file] * batch_size, optimize=True)
    sdf_mol.update_surface_points_and_normals(resolution=resolution, distance=distance)

    surface_coords: torch.Tensor = sdf_mol.surface_coordinates
    padded_surface_coords = padded_with_zeromask(surface_coords, sdf_mol.surface_batches)
    smallest_batch_x = min([sdf_mol.surface_batches[sdf_mol.surface_batches == i].shape[0] for i in
                            range(int(sdf_mol.surface_batches.max() + 1))])
    padded_surface_coords = padded_surface_coords[:, :smallest_batch_x, :].cuda()


    chamfer_distances = []
    smiles_names = []
    for i in tqdm(range(len(other_smiles))):
        current_smiles = [other_smiles[i]]
        try:
            current_mols = Molecules.from_list_of_smiles(current_smiles, batch_size, optimize=True)
        except (ValueError, RuntimeError):
            continue
        current_mols.update_surface_points_and_normals(resolution=resolution, distance=distance)
        surface_coords2: torch.Tensor = current_mols.surface_coordinates

        padded_surface_coords2 = padded_with_zeromask(surface_coords2, current_mols.surface_batches).cuda()
        matrices = models.transforms.random_rotations(len(current_smiles) * batch_size).cuda()
        padded_surface_coords2 = padded_surface_coords2 @ matrices


        smallest_batch_y = min([current_mols.surface_batches[current_mols.surface_batches == i].shape[0] for i in
                                range(int(current_mols.surface_batches.max() + 1))])
        padded_surface_coords2 = padded_surface_coords2[:, :smallest_batch_y,
                                 :]  # add something to adjust for 0s at the end

        current_mols.update_surface_points_and_normals(resolution=resolution, distance=distance)
        if padded_surface_coords2.shape[0] != batch_size:
            print("short")
            continue
        result = model(padded_surface_coords, padded_surface_coords2)
        current_batch_scores = []
        for j in range(batch_size):
            current_batch_scores.append((chamfer_distance(result[j].reshape(1, -1, 3), current_mols.surface_coordinates[
                current_mols.surface_batches == j].reshape(1, -1, 3)))[0].item())
        chamfer_distances.append(min(current_batch_scores))
        smiles_names.append(other_smiles[i])
    return chamfer_distances, smiles_names


class NaiveDataLoader:
    def __init__(self, dataset):
        self.dataset = dataset
        self.batch_size = 32
        self.index = 0

    def __iter__(self):
        self.index = 0
        return self

    def __next__(self):
        if self.index >= len(self.dataset):
            # stop iteration once index is out of bounds
            raise StopIteration
        return self.get()

    def get(self):
        item = self.dataset[self.index]
        self.index += 1
        return item


def get_transformation_matrix_from_rot_trans(rotation_matrix, translation_matrix):
    transformation_matrix = np.identity(4)
    transformation_matrix[:3, :3] = rotation_matrix
    transformation_matrix[:3, 3] = translation_matrix
    return transformation_matrix


def get_transformation_matrix_from_rot(rotation_matrix):
    transformation_matrix = np.identity(4)
    transformation_matrix[:3, :3] = rotation_matrix
    return transformation_matrix


def get_transformation_matrix_from_trans(translation_matrix):
    transformation_matrix = np.identity(4)
    transformation_matrix[:3, 3] = translation_matrix
    return transformation_matrix
