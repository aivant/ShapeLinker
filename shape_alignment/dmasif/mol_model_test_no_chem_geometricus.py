import math
from pyexpat import features
import time
from turtle import distance
from matplotlib.pyplot import sca
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd.profiler as profiler
from torch.nn.utils.rnn import pad_packed_sequence, PackedSequence, pad_sequence
from pykeops.torch import LazyTensor
from shape_alignment.dmasif import data
from shape_alignment.dmasif.geometry_processing import points_to_invariants, ransac_registration, ElasticDistortion
import random
from perceiver_pytorch import PerceiverIO

from shape_alignment.dmasif.geometry_processing import (
    curvatures,
    mesh_normals_areas,
    tangent_vectors,
    atoms_to_points_normals,
)
from shape_alignment.dmasif.helper import soft_dimension, diagonal_ranges
from shape_alignment.dmasif.benchmark_models import DGCNN_seg, PointNet2_seg, dMaSIFConv_seg
from shape_alignment.dmasif.data_preprocessing.convert_smiles2npy import load_smiles_np
from shape_alignment.dmasif.data_preprocessing.convert_single_mol_sdf2npy import load_sdf_np, mol_to_np

from dataclasses import dataclass
import numpy as np
import typing as ty
from pathlib import Path


def knn_atoms(x, y, x_batch, y_batch, k):
    N, D = x.shape
    x_i = LazyTensor(x[:, None, :])
    y_j = LazyTensor(y[None, :, :])

    pairwise_distance_ij = ((x_i - y_j) ** 2).sum(-1)
    pairwise_distance_ij.ranges = diagonal_ranges(x_batch, y_batch)

    # N.B.: KeOps doesn't yet support backprop through Kmin reductions...
    # dists, idx = pairwise_distance_ij.Kmin_argKmin(K=k,axis=1)
    # So we have to re-compute the values ourselves:
    idx = pairwise_distance_ij.argKmin(K=k, axis=1)  # (N, K)
    x_ik = y[idx.view(-1)].view(N, k, D)
    dists = ((x[:, None, :] - x_ik) ** 2).sum(-1)

    return idx, dists


def get_atom_features(x, y, x_batch, y_batch, y_atomtype, k=16):

    idx, dists = knn_atoms(x, y, x_batch, y_batch, k=k)  # (num_points, k)
    num_points, _ = idx.size()

    idx = idx.view(-1)
    dists = 1 / dists.view(-1, 1)
    _, num_dims = y_atomtype.size()

    feature = y_atomtype[idx, :]
    feature = torch.cat([feature, dists], dim=1)
    feature = feature.view(num_points, k, num_dims + 1)

    return feature


class Atom_embedding(nn.Module):
    def __init__(self, atom_dims):
        super(Atom_embedding, self).__init__()
        self.D = atom_dims
        self.k = 16
        self.conv1 = nn.Linear(self.D + 1, self.D)
        self.conv2 = nn.Linear(self.D, self.D)
        self.conv3 = nn.Linear(2 * self.D, self.D)
        self.bn1 = nn.BatchNorm1d(self.D)
        self.bn2 = nn.BatchNorm1d(self.D)
        self.relu = nn.LeakyReLU(negative_slope=0.2)

    def forward(self, x, y, y_atomtypes, x_batch, y_batch):
        fx = get_atom_features(x, y, x_batch, y_batch, y_atomtypes, k=self.k)
        fx = self.conv1(fx)
        fx = fx.view(-1, self.D)
        fx = self.bn1(self.relu(fx))
        fx = fx.view(-1, self.k, self.D)
        fx1 = fx.sum(dim=1, keepdim=False)

        fx = self.conv2(fx)
        fx = fx.view(-1, self.D)
        fx = self.bn2(self.relu(fx))
        fx = fx.view(-1, self.k, self.D)
        fx2 = fx.sum(dim=1, keepdim=False)
        fx = torch.cat((fx1, fx2), dim=-1)
        fx = self.conv3(fx)

        return fx


class AtomNet(nn.Module):
    def __init__(self, atom_dims):
        super(AtomNet, self).__init__()

        self.transform_types = nn.Sequential(
            nn.Linear(atom_dims, atom_dims),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(atom_dims, atom_dims),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(atom_dims, atom_dims),
            nn.LeakyReLU(negative_slope=0.2),
        )
        self.embed = Atom_embedding(atom_dims)

    def forward(self, xyz, atom_xyz, atomtypes, batch, atom_batch):
        # Run a DGCNN on the available information:
        atomtypes = self.transform_types(atomtypes)
        return self.embed(xyz, atom_xyz, atomtypes, batch, atom_batch)

class Atom_embedding_MP(nn.Module):
    def __init__(self, atom_dims):
        super(Atom_embedding_MP, self).__init__()
        self.D = atom_dims
        self.k = 16
        self.n_layers = 3
        self.mlp = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(2 * self.D + 1, 2 * self.D + 1),
                    nn.LeakyReLU(negative_slope=0.2),
                    nn.Linear(2 * self.D + 1, self.D),
                )
                for i in range(self.n_layers)
            ]
        )
        self.norm = nn.ModuleList(
            [nn.GroupNorm(2, self.D) for i in range(self.n_layers)]
        )
        self.relu = nn.LeakyReLU(negative_slope=0.2)

    def forward(self, x, y, y_atomtypes, x_batch, y_batch):
        idx, dists = knn_atoms(x, y, x_batch, y_batch, k=self.k)  # N, 9, 7
        num_points = x.shape[0]
        num_dims = y_atomtypes.shape[-1]

        point_emb = torch.ones_like(x[:, 0])[:, None].repeat(1, num_dims)
        for i in range(self.n_layers):
            features = y_atomtypes[idx.reshape(-1), :]
            features = torch.cat([features, dists.reshape(-1, 1)], dim=1)
            features = features.view(num_points, self.k, num_dims + 1)
            features = torch.cat(
                [point_emb[:, None, :].repeat(1, self.k, 1), features], dim=-1
            )  # N, 8, 13

            messages = self.mlp[i](features)  # N,8,6
            messages = messages.sum(1)  # N,6
            point_emb = point_emb + self.relu(self.norm[i](messages))

        return point_emb

class Atom_Atom_embedding_MP(nn.Module):
    def __init__(self, atom_dims):
        super(Atom_Atom_embedding_MP, self).__init__()
        self.D = atom_dims
        self.k = 17
        self.n_layers = 3

        self.mlp = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(2 * self.D + 1, 2 * self.D + 1),
                    nn.LeakyReLU(negative_slope=0.2),
                    nn.Linear(2 * self.D + 1, self.D),
                )
                for i in range(self.n_layers)
            ]
        )

        self.norm = nn.ModuleList(
            [nn.GroupNorm(2, self.D) for i in range(self.n_layers)]
        )
        self.relu = nn.LeakyReLU(negative_slope=0.2)

    def forward(self, x, y, y_atomtypes, x_batch, y_batch):
        idx, dists = knn_atoms(x, y, x_batch, y_batch, k=self.k)  # N, 9, 7
        idx = idx[:, 1:]  # Remove self
        dists = dists[:, 1:]
        k = self.k - 1
        num_points = y_atomtypes.shape[0]

        out = y_atomtypes
        for i in range(self.n_layers):
            _, num_dims = out.size()
            features = out[idx.reshape(-1), :]
            features = torch.cat([features, dists.reshape(-1, 1)], dim=1)
            features = features.view(num_points, k, num_dims + 1)
            features = torch.cat(
                [out[:, None, :].repeat(1, k, 1), features], dim=-1
            )  # N, 8, 13

            messages = self.mlp[i](features)  # N,8,6
            messages = messages.sum(1)  # N,6
            out = out + self.relu(self.norm[i](messages))

        return out

class AtomNet_MP(nn.Module):
    def __init__(self, atom_dims):
        super(AtomNet_MP, self).__init__()

        self.transform_types = nn.Sequential(
            nn.Linear(atom_dims, atom_dims),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(atom_dims, atom_dims),
        )

        self.embed = Atom_embedding_MP(atom_dims)
        self.atom_atom = Atom_Atom_embedding_MP(atom_dims)

    def forward(self, xyz, atom_xyz, atomtypes, batch, atom_batch):
        # Run a DGCNN on the available information:
        atomtypes = self.transform_types(atomtypes)
        atomtypes = self.atom_atom(
            atom_xyz, atom_xyz, atomtypes, atom_batch, atom_batch
        )
        atomtypes = self.embed(xyz, atom_xyz, atomtypes, batch, atom_batch)
        return atomtypes


def combine_pair(P1, P2):
    P1P2 = {}
    for key in P1:
        v1 = P1[key]
        v2 = P2[key]
        if v1 is None:
            continue

        if key == "batch" or key == "batch_atoms":
            v1v2 = torch.cat([v1, v2 + v1[-1] + 1], dim=0)
        elif key == "triangles":
            # v1v2 = torch.cat([v1,v2],dim=1)
            continue
        else:
            v1v2 = torch.cat([v1, v2], dim=0)
        P1P2[key] = v1v2

    return P1P2


def split_pair(P1P2):
    batch_size = P1P2["batch_atoms"][-1] + 1
    p1_indices = P1P2["batch"] < batch_size // 2
    p2_indices = P1P2["batch"] >= batch_size // 2

    p1_atom_indices = P1P2["batch_atoms"] < batch_size // 2
    p2_atom_indices = P1P2["batch_atoms"] >= batch_size // 2

    P1 = {}
    P2 = {}
    for key in P1P2:
        v1v2 = P1P2[key]

        if (key == "rand_rot") or (key == "atom_center"):
            n = v1v2.shape[0] // 2
            P1[key] = v1v2[:n].view(-1, 3)
            P2[key] = v1v2[n:].view(-1, 3)
        elif "atom" in key:
            P1[key] = v1v2[p1_atom_indices]
            P2[key] = v1v2[p2_atom_indices]
        elif key == "triangles":
            continue
            # P1[key] = v1v2[:,p1_atom_indices]
            # P2[key] = v1v2[:,p2_atom_indices]
        else:
            P1[key] = v1v2[p1_indices]
            P2[key] = v1v2[p2_indices]

    P2["batch"] = P2["batch"] - batch_size + 1
    P2["batch_atoms"] = P2["batch_atoms"] - batch_size + 1

    return P1, P2



def project_iface_labels(P, threshold=2.0):

    queries = P["xyz"]
    batch_queries = P["batch"]
    source = P["mesh_xyz"]
    batch_source = P["mesh_batch"]
    labels = P["mesh_labels"]
    x_i = LazyTensor(queries[:, None, :])  # (N, 1, D)
    y_j = LazyTensor(source[None, :, :])  # (1, M, D)

    D_ij = ((x_i - y_j) ** 2).sum(-1).sqrt()  # (N, M)
    D_ij.ranges = diagonal_ranges(batch_queries, batch_source)
    nn_i = D_ij.argmin(dim=1).view(-1)  # (N,)
    nn_dist_i = (
        D_ij.min(dim=1).view(-1, 1) < threshold
    ).float()  # If chain is not connected because of missing densities MaSIF cut out a part of the protein
    query_labels = labels[nn_i] * nn_dist_i
    P["labels"] = query_labels


@dataclass
class Molecules:
    atom_coordinates: torch.TensorType
    atom_batches: torch.TensorType
    atom_types: torch.TensorType
    surface_coordinates: ty.Union[None, torch.TensorType] = None
    surface_normals: ty.Union[None, torch.TensorType] = None
    surface_batches: ty.Union[None, torch.TensorType] = None
    features: ty.Union[None, torch.TensorType] = None
    embedding1: ty.Union[None, torch.TensorType] = None
    embedding2: ty.Union[None, torch.TensorType] = None
    interface_pred: ty.Union[None, torch.TensorType] = None
    device: str = "cuda"
    geometricus: bool = False
    surface_geometricus_embeddings: ty.Union[None, torch.TensorType] = None

    @classmethod
    def from_coords(cls, lists_of_coords, lists_of_atomnames, device="cuda"):
        batches = list()
        atom_types = list()
        coords = list()

        for i, coord in enumerate(lists_of_coords):
            npys = mol_to_np(atom_names=lists_of_atomnames[i], coords=coord, center=False)
            current_coords, current_atomtypes = npys["xyz"].astype(np.float32), npys["types"].astype(np.float32)
            batch = np.zeros(len(current_coords), dtype=int)
            batch[:] = i
            batches.append(torch.tensor(batch.astype(int), device=device))
            coords.append(torch.tensor(current_coords, device=device))
            atom_types.append(torch.tensor(current_atomtypes, device=device))
        
        return cls(torch.vstack(coords), torch.concat(batches), torch.vstack(atom_types))


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
    def from_list_of_smiles(cls, multi_smiles, number_of_conformers, optimize=True, device="cuda", torsion_noise=0, geometricus=False):
        batches = list()
        atom_types = list()
        coords = list()
        b_num = 0
        for smiles in multi_smiles:
            for conformer in load_smiles_np(smiles, number_of_conformers, center=False, optimize=optimize, torsion_noise=torsion_noise):
                current_coords = conformer["xyz"].astype(np.float32)
                current_atomtypes = conformer["types"].astype(np.float32)
                batch = np.zeros(len(current_coords), dtype=np.float32)
                batch[:] = b_num
                batches.append(torch.tensor(batch.astype(int), device=device))
                coords.append(torch.tensor(current_coords, device=device))
                atom_types.append(torch.tensor(current_atomtypes, device=device))
                b_num += 1
        return cls(torch.vstack(coords), torch.concat(batches), torch.vstack(atom_types), geometricus=geometricus)
    
    @classmethod
    def from_list_of_sdf_files(cls, sdf_file_paths, device="cuda"):
        batches = list()
        atom_types = list()
        coords = list()
        b_num = 0
        for sdf_path in sdf_file_paths:
            for conformer in load_sdf_np(sdf_path, center=False):
                current_coords = conformer["xyz"].astype(np.float32)
                current_atomtypes = conformer["types"].astype(np.float32)
                batch = np.zeros(len(current_coords), dtype=np.float32)
                batch[:] = b_num
                batches.append(torch.tensor(batch.astype(int), device=device))
                coords.append(torch.tensor(current_coords, device=device))
                atom_types.append(torch.tensor(current_atomtypes, device=device))
                b_num += 1
        
        return cls(torch.vstack(coords), torch.concat(batches), torch.vstack(atom_types))

    def update_surface_points_and_normals(self, resolution: float = .5, sup_sampling: float = 100, nits: int = 6, distance: float = 1.05):
        self.surface_coordinates, self.surface_normals, self.surface_batches = atoms_to_points_normals(
            self.atom_coordinates, 
            self.atom_batches, 
            atomtypes=self.atom_types, 
            resolution=resolution,
            sup_sampling=sup_sampling, 
            nits=nits,
            distance=distance
        )

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

    def update_features(
        self, 
        atomnet_model: AtomNet_MP, 
        curvature_scales: ty.List[float], 
        resolution: float = .5, 
        sup_sampling: float = 100., 
        nits: int = 6,
        distance: float = 1.05,
        force: bool = False):
        if (self.surface_coordinates is None) or force:
            self.update_surface_points_and_normals(
                resolution, sup_sampling, nits, distance
            )
            self.update_geometricus_features()
        # chem_features = self.get_chem_features(atomnet_model)
        curvatures = self.estimate_curvatures(curvature_scales)
        
        if self.surface_geometricus_embeddings is not None:
            self.features = torch.cat([curvatures, self.surface_geometricus_embeddings], dim=1).contiguous()
        else:
            self.update_geometricus_features()
            self.features = torch.cat([curvatures, self.surface_geometricus_embeddings], dim=1).contiguous()
    
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
        geom_all = []
        for i in range(self.surface_batches.max() + 1):
            geom_all.append(
                torch.hstack((
                    torch.tensor(
                    points_to_invariants(self.surface_coordinates[self.surface_batches == i].cpu().detach().numpy().astype(np.float64), radius).astype(np.float32), 
                    device=self.device),
                    torch.tensor(
                    points_to_invariants(self.surface_normals[self.surface_batches == i].cpu().detach().numpy().astype(np.float64), radius).astype(np.float32), 
                    device=self.device)))
            )
        self.surface_geometricus_embeddings = torch.vstack(geom_all)



def batches_to_variable_length(sequence, batches):
    variable_sequences = []
    for i in range(int(batches.max().item()) + 1):
        variable_sequences.append(sequence[batches == i])
    return variable_sequences


def emit_molecules_for_contrastive_training(list_of_smiles, number_of_conformers, device="cuda"):
    similar1 = Molecules.from_list_of_smiles(list_of_smiles, number_of_conformers, optimize=True, device=device, geometricus=True)
    similar2 = Molecules.from_list_of_smiles(list_of_smiles, number_of_conformers, optimize=False, device=device, geometricus=True)
    distant = Molecules.from_list_of_smiles(random.sample(list_of_smiles, len(list_of_smiles)), number_of_conformers, optimize=False, device=device, geometricus=True)
    return similar1, similar2, distant


def emit_molecules_for_contrastive_training_with_torsion_noise(list_of_smiles, number_of_conformers, device="cuda", degree_noise=10, mult=2):
    similar1 = Molecules.from_list_of_smiles(list_of_smiles, number_of_conformers, optimize=True, device=device)
    similar2 = Molecules.from_list_of_smiles(list_of_smiles, number_of_conformers, optimize=True, device=device)
    distant = Molecules.from_list_of_smiles(list_of_smiles, number_of_conformers, optimize=False, device=device, torsion_noise=50)
    return similar1, similar2, distant


@dataclass
class ContrastiveDataset:
    similar_1: Molecules
    similar_2: Molecules
    distant: Molecules
    geometric_distances_positive: ty.Union[torch.Tensor, None] = None
    geometric_distances_negative: ty.Union[torch.Tensor, None] = None

    @classmethod
    def from_smiles(
        cls, list_of_smiles, number_of_conformers, device="cuda", degree_noise=50,
        precompute_distances=False, resolution=1.5, distance=.9, sup_sampling=200,
        voxel_size=.6, var=1, geometricus_radius=8, granularity=[2,5], magnitude=[4, 4]):
        
        distort = ElasticDistortion(granularity=granularity, magnitude=magnitude)
        
        var_resolution = np.random.choice(np.linspace(resolution-.1*var, resolution+.1*var, 20))
        var_sup_sampling = np.random.choice(np.linspace(sup_sampling-10*var, sup_sampling+10*var, 20))
        similar_1 = Molecules.from_list_of_smiles(list_of_smiles, number_of_conformers, optimize=True, device=device)
        similar_1.update_surface_points_and_normals(resolution=var_resolution, distance=distance, sup_sampling=int(var_sup_sampling))
        similar_1.update_geometricus_features(radius=geometricus_radius)
        
        similar_2 = Molecules.from_list_of_smiles(list_of_smiles, number_of_conformers, optimize=False, device=device)

        

        similar_2.atom_coordinates = similar_1.atom_coordinates
        similar_2.atom_types = similar_1.atom_types
        similar_2.atom_batches = similar_1.atom_batches

        

        var_resolution2 = np.random.choice(np.linspace(resolution-.1*var, resolution+.1*var, 20))
        var_sup_sampling2 = np.random.choice(np.linspace(sup_sampling-10*var, sup_sampling+10*var, 20))
        similar_2.update_surface_points_and_normals(resolution=var_resolution2, distance=distance, sup_sampling=int(var_sup_sampling2))

        distorted_coords = [torch.tensor(distort(similar_2.surface_coordinates[similar_2.surface_batches == i].cpu().detach().numpy().reshape((-1, 3))).astype(np.float32), device=similar_2.device) for i in range(similar_2.surface_batches.max()+1)]
        similar_2.surface_coordinates = torch.vstack(distorted_coords)

        similar_2.update_geometricus_features(radius=geometricus_radius)
        
        distant = Molecules.from_list_of_smiles(list_of_smiles, number_of_conformers, optimize=False, device=device)
        distant.update_surface_points_and_normals(resolution=var_resolution, distance=distance, sup_sampling=int(var_sup_sampling))
        distant.update_geometricus_features(radius=geometricus_radius)

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
                    self.similar_1.surface_coordinates[self.similar_1.surface_batches == i].cpu().detach().numpy().reshape((-1, 3)),
                    self.similar_2.surface_coordinates[self.similar_2.surface_batches == i].cpu().detach().numpy().reshape((-1, 3)),
                    voxel_size=voxel_size
                    ))
            negative_distances.append(
                1 - ransac_registration(
                    self.similar_1.surface_coordinates[self.similar_1.surface_batches == i].cpu().detach().numpy().reshape((-1, 3)),
                    self.distant.surface_coordinates[self.distant.surface_batches == i].cpu().detach().numpy().reshape((-1, 3)),
                    voxel_size=voxel_size
                    ))
        self.geometric_distances_positive = torch.tensor(np.array(positive_distances).astype(np.float32), device=self.similar_1.device)
        self.geometric_distances_negative = torch.tensor(np.array(negative_distances).astype(np.float32), device=self.similar_1.device)





class GeometricusPerc(torch.nn.Module):
    def __init__(self, input_dim, latent_dim, latent_scale):
        super(GeometricusPerc, self).__init__()
        self.perc = PerceiverIO(
                dim = input_dim,                    # dimension of sequence to be encoded
                queries_dim = 32,            # dimension of decoder queries
                logits_dim = 100,            # dimension of final logits
                depth = 6,                   # depth of net
                num_latents = latent_scale,           # number of latents, or induced set points, or centroids. different papers giving it different names
                latent_dim = latent_dim,            # latent dimension
                cross_heads = 1,             # number of heads for cross attention. paper said 1
                latent_heads = 8,            # number of heads for latent self attention, 8
                cross_dim_head = 64,         # number of dimensions per cross attention head
                latent_dim_head = 64,        # number of dimensions per latent self attention head
                weight_tie_layers = False    # whether to weight tie layers (optional, as indicated in the diagram)
            )
        self.relu = torch.nn.ReLU()
        self.bn = torch.nn.BatchNorm1d(latent_dim)
        
    def forward(self, molecules: Molecules):
        # geometricus as embedding
        
        sequences = batches_to_variable_length(molecules.surface_geometricus_embeddings, molecules.surface_batches)
        x = pad_sequence(sequences, batch_first=True)
        x = self.perc(x)
        return x.mean(1)



class dMaSIF_perc(torch.nn.Module):
    def __init__(
        self, 
        in_channels=26,
        orientation_units=16,
        emb_dims=8,
        post_units=8,
        curvature_scales=[1.0, 2.0, 3.0, 5.0, 10.0],
        atom_dims=8,
        dropout=.0,
        embedding_layer="dMaSIF",
        n_layers=1,
        radius=9,
        search=False,
        k=40,
        site=False,
        resolution=.5,
        sup_sampling=100,
        use_mesh=False,
        no_geom=False,
        no_chem=False,
        distance=1.05
        ):
        super(dMaSIF_perc, self).__init__()
        self.dmasif = dMaSIF(
        in_channels=in_channels,
        orientation_units=orientation_units,
        emb_dims=emb_dims,
        post_units=post_units,
        curvature_scales=curvature_scales,
        atom_dims=atom_dims,
        dropout=dropout,
        embedding_layer=embedding_layer,
        n_layers=n_layers,
        radius=radius,
        search=search,
        k=k,
        site=site,
        resolution=resolution,
        sup_sampling=sup_sampling,
        use_mesh=use_mesh,
        no_geom=no_geom,
        no_chem=no_chem,
        distance=distance
        )
        self.perc = PerceiverIO(
            dim = emb_dims,                    # dimension of sequence to be encoded
            queries_dim = 32,            # dimension of decoder queries
            logits_dim = 100,            # dimension of final logits
            depth = 6,                   # depth of net
            num_latents = 32,           # number of latents, or induced set points, or centroids. different papers giving it different names
            latent_dim = 1024,            # latent dimension
            cross_heads = 1,             # number of heads for cross attention. paper said 1
            latent_heads = 8,            # number of heads for latent self attention, 8
            cross_dim_head = 64,         # number of dimensions per cross attention head
            latent_dim_head = 64,        # number of dimensions per latent self attention head
            weight_tie_layers = False    # whether to weight tie layers (optional, as indicated in the diagram)
        )

    def forward(self, molecules):
        dmasif_embedding = self.dmasif(molecules)["molecules"].padded_sorted_embedding1_with_zeromask
        return self.perc(dmasif_embedding).mean(1)


def loss_func(out, distant, y):
    dist_sq = torch.sum(torch.pow(out - distant, 2), 1)
    dist = torch.sqrt(dist_sq + 1e-10)
    mdist = 1 - dist
    dist = torch.clamp(mdist, min=0.0)
    loss = y * dist_sq + (1 - y) * torch.pow(dist, 2)
    loss = torch.sum(loss) / 2.0 / out.size()[0]
    return loss