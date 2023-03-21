from sympy import im
from shape_alignment.molecule import Molecules, batches_to_variable_length, pad_sequence
import torch
from torch import nn
from shape_alignment.dmasif.mol_model import AtomNet_MP
import time
from shape_alignment.dmasif.benchmark_models import DGCNN_seg, PointNet2_seg, dMaSIFConv_seg
from shape_alignment.dmasif.geometry_processing import soft_dimension
from perceiver_pytorch import PerceiverIO


class GeometricusPerc(torch.nn.Module):
    def __init__(self, input_dim, latent_dim, latent_scale, feat_type="geometricus"):
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
        self.atomnet = AtomNet_MP(8)
        self.type = feat_type
        
    def forward(self, molecules: Molecules):
        if self.type == "geometricus":
            sequences = batches_to_variable_length(molecules.surface_geometricus_embeddings, molecules.surface_batches)
        elif self.type == "geometricus+chemical":
            sequences = batches_to_variable_length(molecules.get_geom_and_chem_features(self.atomnet), molecules.surface_batches)
        elif self.type == "geometricus+curvatures":
            sequences = batches_to_variable_length(molecules.get_geom_and_curve_features(), molecules.surface_batches)
        elif self.type == "curvatures":
            sequences = batches_to_variable_length(molecules.estimate_curvatures(curvature_scales=[1., 2., 4., 8., 10.]), molecules.surface_batches)
        elif self.type == "chemical":
            sequences = batches_to_variable_length(molecules.get_chem_features(self.atomnet), molecules.surface_batches)
        else:
            raise(Warning("Feature type not recognised, forward will return None"))
        x = pad_sequence(sequences, batch_first=True)
        x = self.perc(x).swapaxes(1,2)
        return x.mean(1)


class GeometricusPercAtomBased(torch.nn.Module):
    def __init__(self, input_dim, latent_dim, latent_scale, feat_type="geometricus"):
        super(GeometricusPercAtomBased, self).__init__()
        self.perc = PerceiverIO(
                dim = input_dim,                    # dimension of sequence to be encoded
                queries_dim = 34,            # dimension of decoder queries
                logits_dim = 16,            # dimension of final logits
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
        self.atomnet = AtomNet_MP(8)
        self.type = feat_type
        
    def forward(self, molecules: Molecules):
        if self.type == "geometricus":
            sequences = batches_to_variable_length(molecules.surface_geometricus_embeddings, molecules.surface_batches)
        elif self.type == "geometricus+chemical":
            sequences = batches_to_variable_length(molecules.get_geom_and_chem_features(self.atomnet), molecules.surface_batches)
        elif self.type == "geometricus+curvatures":
            sequences = batches_to_variable_length(molecules.get_geom_and_curve_features(), molecules.surface_batches)
        elif self.type == "curvatures":
            sequences = batches_to_variable_length(molecules.estimate_curvatures(curvature_scales=[1., 2., 4., 8., 10.]), molecules.surface_batches)
        elif self.type == "chemical":
            sequences = batches_to_variable_length(molecules.get_chem_features(self.atomnet), molecules.surface_batches)
        else:
            raise(Warning("Feature type not recognised, forward will return None"))
        queries = batches_to_variable_length(
            molecules.get_per_atom_invariants(), molecules.atom_batches
            )
        x = pad_sequence(sequences, batch_first=True)
        x = self.perc(x, queries=pad_sequence(queries, batch_first=True))
        return x.reshape((x.shape[0], -1))


class dMaSIF(nn.Module):
    def __init__(
        self, 
        in_channels=18,
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
        super(dMaSIF, self).__init__()
        # Additional geometric features: mean and Gauss curvatures computed at different scales.
        self.curvature_scales = curvature_scales

        I = in_channels
        O = orientation_units
        E = emb_dims
        H = post_units

        # Computes chemical features
        self.atomnet = AtomNet_MP(atom_dims)
        self.dropout = nn.Dropout(dropout)

        if embedding_layer == "dMaSIF":
            # Post-processing, without batch norm:
            self.orientation_scores = nn.Sequential(
                nn.Linear(I, O),
                nn.LeakyReLU(negative_slope=0.2),
                nn.Linear(O, 1),
            )

            # Segmentation network:
            self.conv = dMaSIFConv_seg(
                in_channels=I,
                out_channels=E,
                n_layers=n_layers,
                radius=radius,
            )

            # Asymmetric embedding
            if search:
                self.orientation_scores2 = nn.Sequential(
                    nn.Linear(I, O),
                    nn.LeakyReLU(negative_slope=0.2),
                    nn.Linear(O, 1),
                )

                self.conv2 = dMaSIFConv_seg(
                    in_channels=I,
                    out_channels=E,
                    n_layers=n_layers,
                    radius=radius,
                )

        elif embedding_layer == "DGCNN":
            self.conv = DGCNN_seg(I + 3, E, n_layers, k)
            if search:
                self.conv2 = DGCNN_seg(I + 3, E, n_layers, k)

        elif embedding_layer == "PointNet++":
            self.conv = PointNet2_seg(I, E, radius, n_layers)
            if search:
                self.conv2 = PointNet2_seg(I, E, radius, n_layers)

        if site:
            # Post-processing, without batch norm:
            self.net_out = nn.Sequential(
                nn.Linear(E, H),
                nn.LeakyReLU(negative_slope=0.2),
                nn.Linear(H, H),
                nn.LeakyReLU(negative_slope=0.2),
                nn.Linear(H, 1),
            )
        
        self.resolution = resolution
        self.sup_sampling = sup_sampling
        self.use_mesh = use_mesh
        self.no_geom = no_geom
        self.no_chem = no_chem
        self.embedding_layer = embedding_layer
        self.distance = distance
        self.site = site
        self.search = search

    def features(self, molecules: Molecules, i=1):
        """Estimates geometric and chemical features from a protein surface or a cloud of atoms."""
        molecules.update_dmasif_features(
            self.atomnet,
            self.curvature_scales
            )
        return molecules.features

    def embed(self, molecules: Molecules):
        """Embeds all points of a protein in a high-dimensional vector space."""

        features = self.dropout(self.features(molecules))

        torch.cuda.synchronize(device=features.device)
        torch.cuda.reset_max_memory_allocated(device=molecules.atom_coordinates.device)
        begin = time.time()

        # Ours:
        if self.embedding_layer == "dMaSIF":
            self.conv.load_mesh(
                molecules.surface_coordinates,
                triangles=None,
                normals=molecules.surface_normals,
                weights=self.orientation_scores(features),
                batch=molecules.surface_batches,
            )
            molecules.embedding1 = self.conv(features)
            if self.search:
                self.conv2.load_mesh(
                    molecules.surface_coordinates,
                    triangles=None,
                    normals=molecules.surface_normals,
                    weights=self.orientation_scores2(features),
                    batch=molecules.surface_batches,
                )
                molecules.embedding2 = self.conv2(features)

        # First baseline:
        elif self.embedding_layer == "DGCNN":
            features = torch.cat([features, molecules.surface_coordinates], dim=-1).contiguous()
            molecules.embedding1 = self.conv(molecules.surface_coordinates, features, molecules.surface_batches)
            if self.search:
                molecules.embedding2 = self.conv2(
                    molecules.surface_coordinates, features, molecules.surface_batches
                )

        # Second baseline
        elif self.embedding_layer == "PointNet++":
            molecules.embedding1 = self.conv(molecules.surface_coordinates, features, molecules.surface_batches)
            if self.search:
                molecules.embedding2 = self.conv2(molecules.surface_coordinates, features, molecules.surface_batches)

        torch.cuda.synchronize(device=features.device)
        end = time.time()
        memory_usage = torch.cuda.max_memory_allocated(device=molecules.atom_coordinates.device)
        conv_time = end - begin

        return conv_time, memory_usage

    def preprocess_surface(self, molecules: Molecules): # TODO: change
        molecules.update_surface_points_and_normals(
            resolution=self.resolution,
            sup_sampling=self.sup_sampling,
            distance=self.distance
        )


    def forward(self, molecules: Molecules):
        # Compute embeddings of the point clouds:
        conv_time, memory_usage = self.embed(molecules)

        # Monitor the approximate rank of our representations:
        R_values = {}
        R_values["input"] = soft_dimension(molecules.features)
        R_values["conv"] = soft_dimension(molecules.embedding1)

        if self.site:
            molecules.interface_pred = self.net_out(molecules.embedding1)

        return {
            "molecules": molecules,
            "R_values": R_values,
            "conv_time": conv_time,
            "memory_usage": memory_usage,
        }


class dMaSIF_perc(torch.nn.Module):
    def __init__(
        self, 
        in_channels=10,
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
            num_latents = 256,           # number of latents, or induced set points, or centroids. different papers giving it different names
            latent_dim = 4,            # latent dimension
            cross_heads = 1,             # number of heads for cross attention. paper said 1
            latent_heads = 8,            # number of heads for latent self attention, 8
            cross_dim_head = 64,         # number of dimensions per cross attention head
            latent_dim_head = 64,        # number of dimensions per latent self attention head
            weight_tie_layers = False    # whether to weight tie layers (optional, as indicated in the diagram)
        )

    def forward(self, molecules):
        dmasif_embedding = self.dmasif(molecules)["molecules"].padded_sorted_embedding1_with_zeromask
        x = self.perc(dmasif_embedding).swapaxes(1,2)
        return x.mean(1)


def loss_func(out, distant, y):
    dist_sq = torch.sum(torch.pow(out - distant, 2), 1)
    dist = torch.sqrt(dist_sq + 1e-10)
    mdist = 1 - dist
    dist = torch.clamp(mdist, min=0.0)
    loss = y * dist_sq + (1 - y) * torch.pow(dist, 2)
    loss = torch.sum(loss) / 2.0 / out.size()[0]
    return loss