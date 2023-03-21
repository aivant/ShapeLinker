

# start with the multi-head attention model using pytorch lightning. 
# This should accept molecule object as batches 
    # Same number of atoms but padded surface

from torch import optim, nn
import torch
import pytorch_lightning as pl
from pytorch3d.loss import chamfer_distance as cmf
from shape_alignment.molecule import Molecules
from pytorch3d import transforms
from torch.nn import MultiheadAttention
import typing as ty
from shape_alignment.dmasif.geometry_processing import ransac_registration, get_registration_result
import numpy as np
from torch_cluster import radius_graph
from torch_scatter import scatter
from e3nn import o3, nn
from e3nn.math import soft_one_hot_linspace, soft_unit_step


class CrossAttention(pl.LightningModule):
    def __init__(self, attention_in_size: int=12, lin_in_size: int=3, lin_out_size: int=3, num_heads: int=4):
        super().__init__()
        self.lin_in = nn.Linear(3, lin_in_size)
        self.attn = nn.MultiheadAttention(attention_in_size, 4)
        self.cross_attn = nn.MultiheadAttention(attention_in_size, 4)
        self.lin_out = nn.Linear(attention_in_size, lin_out_size)
        
    def training_step(self, batch):
        # training_step defines the train loop.
        # it is independent of forward
        x, y, d = batch
        cross_attention = self.forward(x, y)
        loss = contrastive_loss(cross_attention.reshape((x.shape[0], -1)), y.reshape((y.shape[0], -1)), d)
        # Logging to TensorBoard by default
        self.log("train_loss", loss)
        return loss

    def forward(self, x, y):
        xi = self.lin_in(x)
        yi = self.lin_in(y)
        xi = self.attn(xi, xi, xi)[0]
        yi = self.attn(yi, yi, yi)[0]
        cross = self.cross_attn(yi, xi, xi)[0]
        return self.lin_out(cross)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        return optimizer
    
    def validation_step(self, val_batch):
        x, y, d = val_batch
        cross_attention = self.forward(x, y)
        loss = contrastive_loss(cross_attention.reshape((x.shape[0], -1)), y.reshape((y.shape[0], -1)), d)
        self.log('val_loss', loss)


def contrastive_loss(out: torch.Tensor, distant: torch.Tensor, y: torch.Tensor, reduction: str="mean"):
    dist_sq = torch.sum(torch.pow(out - distant, 2), 2)
    dist = torch.sqrt(dist_sq + 1e-10)
    mdist = 1 - dist
    dist = torch.clamp(mdist, min=0.0)
    loss = y * dist_sq + (1 - y) * torch.pow(dist, 2)
    loss = torch.sum(loss) / 2.0 / y.size()[0]
    if reduction == "mean":
        return loss / out.shape[1]
    else:
        return loss


def reflect(U, R, Vt, A):
    SS = torch.zeros((R.shape[0], 3, 3)).cuda()
    for i, r in enumerate(R):
        if torch.linalg.det(r) < 0:
            SS[i] = torch.diag(torch.tensor([1.,1., 1.], device=A.device))
        else:
            SS[i] = torch.diag(torch.tensor([1.,1., 1.], device=A.device))
    
    R =  torch.bmm(U, torch.bmm(torch.transpose(SS, 1, 2), Vt))

    # (Vt.T @ SS) @ U.T
    # print(SS)
    return R


traced_reflect = torch.jit.script(
    reflect
    )


def n_dim_rigid_transform_Kabsch_3D_torch(A: torch.Tensor, B: torch.Tensor)-> tuple[torch.Tensor, torch.Tensor]:
    assert A.shape == B.shape
    assert A.shape[2] == B.shape[2] == 3

    centroid_A = A.mean(1)
    centroid_B = B.mean(1)

    Am = A - centroid_A.reshape((-1, 1, 3))
    Bm = B - centroid_B.reshape((-1, 1, 3))

    H = torch.transpose(Bm, 1, 2) @ Am
    U, S, Vt = torch.linalg.svd(H)
    R = U @ Vt    
    # print(R.shape)
    R = traced_reflect(U, R, Vt, A)

    t = centroid_A.reshape((-1, 1, 3)) - torch.bmm(centroid_B.reshape((-1, 1, 3)), R)
    # t = dim1dot_tscript(centroid_A, centroid_B, R)
    return R, t


def rigid_transform_Kabsch_3D_torch_numpy(coords_1: torch.Tensor, coords_2: torch.Tensor):
    """
    Superpose paired coordinates on each other using Kabsch superposition (SVD)
    Parameters
    ----------
    coords_1
        tensor of coordinate data for the first protein; shape = (n, 3)
    coords_2
        tensor of corresponding coordinate data for the second protein; shape = (n, 3)
    Returns
    -------
    rotation matrix, translation matrix for optimal superposition
    """
    centroid_1, centroid_2 = (
        coords_1.mean(0),
        coords_2.mean(0),
    )
    coords_1_c, coords_2_c = coords_1 - centroid_1, coords_2 - centroid_2
    correlation_matrix = coords_2_c.T @ coords_1_c
    u, s, v = torch.linalg.svd(correlation_matrix)
    reflect = torch.linalg.det(u) * torch.linalg.det(v) < 0
    if reflect:
        s[-1] = -s[-1]
        u[:, -1] = -u[:, -1]
    rotation_matrix = u @ v
    translation_matrix = centroid_1 - (centroid_2 @ rotation_matrix)
    return rotation_matrix, translation_matrix


class PCRBase(torch.nn.Module):
    def __init__(self, input_dim, attention_dim=12, nheads=4) -> None:
        """
        input_dim: dimension of input features per point.
        """
        super(PCRBase, self).__init__()
        self.lin_in = torch.nn.Linear(input_dim, attention_dim).cuda()
        self.attn = MultiheadAttention(attention_dim, nheads, batch_first=True).cuda()
        self.cross_attn = MultiheadAttention(attention_dim, nheads, batch_first=True).cuda()
        self.lin_out = torch.nn.Linear(attention_dim, 3).cuda()

    def forward(self, x_orig, y_orig):
        """
        x_orig: query torch tensor with batches of padded point coordinates as first 3 features and rest of the optional features (batch, points, features)
        y_orig: target torch tensor with batches of padded point coordinates as first 3 features and rest of the optional features (batch, points, features)
        """
        batch_size: int = x_orig.shape[0]

        # key masks to ignore padding
        # mask_x = lengths_to_key_mask(x_orig, x_lengths).cuda() if x_lengths is not None else torch.zeros((x_orig.shape[0], x_orig.shape[1])).type(torch.float32).cuda()
        # mask_y = lengths_to_key_mask(y_orig, y_lengths).cuda() if y_lengths is not None else torch.zeros((y_orig.shape[0], y_orig.shape[1])).type(torch.float32).cuda()

        # centre on zero
        x = x_orig - x_orig.mean(1).reshape((-1, 1, x_orig.shape[-1]))
        y_translate = y_orig.mean(1).reshape((-1, 1, y_orig.shape[-1]))
        y = y_orig - y_translate

        xi, yi = self.lin_in(x), self.lin_in(y) # MLP to resize features
        xi = self.attn(xi, xi, xi)[0] # query self attentin
        yi = self.attn(yi, yi, yi)[0] # target self attention
        cross = self.cross_attn(xi, yi, yi)[0] # Cross attention between query and target
        coords = self.lin_out(cross.reshape(-1, cross.shape[-1])).reshape((batch_size, -1, 3))
        rot, trans = n_dim_rigid_transform_Kabsch_3D_torch(coords + x[:, :, :3], x[:, :, :3]) # add the relative coords and register original coordinates
        return ((x[:,:,:3] @ rot) + trans) + y_translate[:,:,:3] # rotate and translate with Kabsch output, finally center on the original target coordinates
    
    def get_rottrans(self, x_orig, y_orig):
        """
        x_orig: query torch tensor with batches of padded point coordinates as first 3 features and rest of the optional features (batch, points, features)
        y_orig: target torch tensor with batches of padded point coordinates as first 3 features and rest of the optional features (batch, points, features)
        """
        batch_size: int = x_orig.shape[0]

        # key masks to ignore padding
        # mask_x = lengths_to_key_mask(x_orig, x_lengths).cuda() if x_lengths is not None else torch.zeros((x_orig.shape[0], x_orig.shape[1])).type(torch.float32).cuda()
        # mask_y = lengths_to_key_mask(y_orig, y_lengths).cuda() if y_lengths is not None else torch.zeros((y_orig.shape[0], y_orig.shape[1])).type(torch.float32).cuda()

        # centre on zero
        x_centroid = x_orig.mean(1).reshape((-1, 1, x_orig.shape[-1]))
        x = x_orig - x_centroid
        y_translate = y_orig.mean(1).reshape((-1, 1, y_orig.shape[-1]))
        y = y_orig - y_translate

        xi, yi = self.lin_in(x), self.lin_in(y) # MLP to resize features
        xi = self.attn(xi, xi, xi)[0] # query self attentin
        yi = self.attn(yi, yi, yi)[0] # target self attention
        cross = self.cross_attn(xi, yi, yi)[0] # Cross attention between query and target
        coords = self.lin_out(cross.reshape(-1, cross.shape[-1])).reshape((batch_size, -1, 3))
        rot, trans = n_dim_rigid_transform_Kabsch_3D_torch(coords + x[:, :, :3], x[:, :, :3]) # add the relative coords and register original coordinates
        return rot, trans + y_translate[:,:,:3], x_centroid
    
    def get_pseudo_coords(self, x_orig, y_orig):
        batch_size: int = x_orig.shape[0]

        # key masks to ignore padding
        # mask_x = lengths_to_key_mask(x_orig, x_lengths).cuda() if x_lengths is not None else torch.zeros((x_orig.shape[0], x_orig.shape[1])).type(torch.float32).cuda()
        # mask_y = lengths_to_key_mask(y_orig, y_lengths).cuda() if y_lengths is not None else torch.zeros((y_orig.shape[0], y_orig.shape[1])).type(torch.float32).cuda()

        # centre on zero
        x = x_orig - x_orig.mean(1).reshape((-1, 1, x_orig.shape[-1]))
        y_translate = y_orig.mean(1).reshape((-1, 1, y_orig.shape[-1]))
        y = y_orig - y_translate

        xi, yi = self.lin_in(x), self.lin_in(y) # MLP to resize features
        xi = self.attn(xi, xi, xi)[0] # query self attentin
        yi = self.attn(yi, yi, yi)[0] # target self attention
        cross = self.cross_attn(xi, yi, yi)[0] # Cross attention between query and target
        coords = self.lin_out(cross.reshape(-1, cross.shape[-1])).reshape((batch_size, -1, 3))
        return coords + y_translate[:,:,:3]


class EquivariantPCR(torch.nn.Module):
    def __init__(self, input_dim, attn_dim: int=23, radius: int=2, n_head: int=8) -> None:
        self.radius = radius
        self.irreps_sh = o3.Irreps.spherical_harmonics(3)
        self.irreps_input = o3.Irreps([(input_dim - 3, (0, 1))])
        self.irreps_query = o3.Irreps([(attn_dim, (0, 1))])
        self.irreps_key = o3.Irreps("12x0e + 3x1o")
        self.irreps_output = o3.Irreps("14x0e + 6x1o")
        self.h_q = o3.Linear(irreps_input, irreps_query)
        self.tp_k = o3.FullyConnectedTensorProduct(irreps_input, irreps_sh, irreps_key, shared_weights=False)
        self.fc_k = nn.FullyConnectedNet([10, 16, tp_k.weight_numel], act=torch.nn.functional.silu)
        self.tp_v = o3.FullyConnectedTensorProduct(irreps_input, irreps_sh, irreps_output, shared_weights=False)
        self.fc_v = nn.FullyConnectedNet([10, 16, tp_v.weight_numel], act=torch.nn.functional.silu)
        self.dot = o3.FullyConnectedTensorProduct(irreps_query, irreps_key, "0e")

        self.attn = MultiheadAttention(attn_dim, num_heads=n_head, batch_first=True)
        self.cross_attn = MultiheadAttention(attn_dim, num_heads=n_head, batch_first=True)
    
    def equivariant_self_attention(self, x: torch.Tensor):
        pos = x[:, :, :3]
        f = x[:, :, 3:]
        edge_src, edge_dst = radius_graph(pos, self.radius)
        edge_vec = pos[edge_src] - pos[edge_dst]
        edge_length = edge_vec.norm(dim=1)

        edge_length_embedded = soft_one_hot_linspace(
            edge_length,
            start=0.0,
            end=self.radius,
            number=10,
            basis='smooth_finite',
            cutoff=True
        )
        edge_length_embedded = edge_length_embedded.mul(10**0.5)
        edge_weight_cutoff = soft_unit_step(10 * (1 - edge_length / self.radius))

        edge_sh = o3.spherical_harmonics(self.irreps_sh, edge_vec, True, normalization='component')

        q = self.h_q(f)
        k = self.tp_k(f[edge_src], edge_sh, self.fc_k(edge_length_embedded))
        v = self.tp_v(f[edge_src], edge_sh, self.fc_v(edge_length_embedded))

        exp = edge_weight_cutoff[:, None] * self.dot(q[edge_dst], k).exp()
        z = scatter(exp, edge_dst, dim=0, dim_size=len(f))
        z[z == 0] = 1
        alpha = exp / z[edge_dst]

        return scatter(alpha.relu().sqrt() * v, edge_dst, dim=0, dim_size=len(f))
    
    def forward(self, x, y):
        coords_x = x[:, :, :3]
        xi = self.equivariant_self_attention(x)
        yi = self.equivariant_self_attention(y)
        coords_y = y[:, :, :3]
        yii = self.attn(y, y, y)[0]


class PCRBaseSepFeat(torch.nn.Module):
    def __init__(self, input_dim, attention_dim=12, nheads=4) -> None:
        """
        input_dim: dimension of input features per point.
        """
        super(PCRBaseSepFeat, self).__init__()
        self.lin_in = torch.nn.Linear(input_dim - 3, attention_dim).cuda()
        self.lin_in_coords = torch.nn.Linear(3, attention_dim).cuda()
        self.attn = MultiheadAttention(attention_dim, nheads, batch_first=True).cuda()
        self.attn_feat = MultiheadAttention(attention_dim, nheads, batch_first=True).cuda()
        self.cross_attn_coords_feat = MultiheadAttention(attention_dim, nheads, batch_first=True).cuda()
        self.cross_attn = MultiheadAttention(attention_dim, nheads, batch_first=True).cuda()
        self.lin_out = torch.nn.Linear(attention_dim, 3).cuda()

    def forward(self, x_orig, y_orig):
        """
        x_orig: query torch tensor with batches of padded point coordinates as first 3 features and rest of the optional features (batch, points, features)
        y_orig: target torch tensor with batches of padded point coordinates as first 3 features and rest of the optional features (batch, points, features)
        """
        batch_size: int = x_orig.shape[0]
        assert (x_orig.shape[2] > 3) and (y_orig.shape[2] > 3)

        # centre on zero
        x_coords = x_orig[:, :, :3]
        x_feat = x_orig[:, :, 3:]
        y_coords = y_orig[:, :, :3]
        y_feat = y_orig[:, :, 3:]

        x_coords = x_coords - x_coords.mean(1).reshape((-1, 1, x_coords.shape[-1]))
        y_translate = y_coords.mean(1).reshape((-1, 1, y_coords.shape[-1]))
        y_coords = y_coords - y_translate


        xi, yi, xi_feat, yi_feat = self.lin_in_coords(x_coords), self.lin_in_coords(y_coords), self.lin_in(x_feat), self.lin_in(y_feat) # MLP to resize features
        
        xi_feat = self.attn_feat(xi_feat, xi_feat, xi_feat)[0]
        yi_feat = self.attn_feat(yi_feat, yi_feat, yi_feat)[0]
        xi = self.attn(xi, xi, xi)[0] # query self attentin
        yi = self.attn(yi, yi, yi)[0] # target self attention
        cross_feat_x = self.cross_attn_coords_feat(xi, xi_feat, xi_feat)[0]
        cross_feat_y = self.cross_attn_coords_feat(yi, yi_feat, yi_feat)[0]
        cross = self.cross_attn(cross_feat_x, cross_feat_y, cross_feat_y)[0]
        coords = self.lin_out(cross.reshape(-1, cross.shape[-1])).reshape((batch_size, -1, 3))
        rot, trans = n_dim_rigid_transform_Kabsch_3D_torch(coords + x_coords, x_coords) # add the relative coords and register original coordinates
        return ((x_coords @ rot) + trans) + y_translate # rotate and translate with Kabsch output, finally center on the original target coordinates
    
    def get_pseudo_coords(self, x_orig, y_orig):
        batch_size: int = x_orig.shape[0]

        # centre on zero
        x_coords = x_orig[:, :, :3]
        x_feat = x_orig[:, :, 3:]
        y_coords = y_orig[:, :, :3]
        y_feat = y_orig[:, :, 3:]

        x_coords = x_coords - x_coords.mean(1).reshape((-1, 1, x_coords.shape[-1]))
        y_translate = y_coords.mean(1).reshape((-1, 1, y_coords.shape[-1]))
        y_coords = y_coords - y_translate


        xi, yi, xi_feat, yi_feat = self.lin_in_coords(x_coords), self.lin_in_coords(y_coords), self.lin_in(x_feat), self.lin_in(y_feat) # MLP to resize features
        
        xi_feat = self.attn_feat(xi_feat, xi_feat, xi_feat)[0]
        yi_feat = self.attn_feat(yi_feat, yi_feat, yi_feat)[0]
        xi = self.attn(xi, xi, xi)[0] # query self attentin
        yi = self.attn(yi, yi, yi)[0] # target self attention
        cross_feat_x = self.cross_attn_coords_feat(xi, xi_feat, xi_feat)[0]
        cross_feat_y = self.cross_attn_coords_feat(yi, yi_feat, yi_feat)[0]
        cross = self.cross_attn(cross_feat_x, cross_feat_y, cross_feat_y)[0] # Cross attention between query and target
        coords = self.lin_out(cross.reshape(-1, cross.shape[-1])).reshape((batch_size, -1, 3))
        return coords + y_translate


class PCRBaseAtomAnchor(torch.nn.Module):
    def __init__(self, input_dim, attention_dim=12, nheads=4) -> None:
        """
        input_dim: dimension of input features per point.
        """
        super(PCRBaseAtomAnchor, self).__init__()
        self.lin_in = torch.nn.Linear(input_dim, attention_dim).cuda()
        self.lin_in_atom = torch.nn.Linear(3, attention_dim).cuda()

        self.atom_attn = MultiheadAttention(attention_dim, nheads, batch_first=True).cuda()
        self.atom_surface_cross = MultiheadAttention(attention_dim, nheads, batch_first=True).cuda()
        
        self.attn = MultiheadAttention(attention_dim, nheads, batch_first=True).cuda()
        self.cross_attn = MultiheadAttention(attention_dim, nheads, batch_first=True).cuda()
        
        self.lin_out = torch.nn.Linear(attention_dim, 3).cuda()

    def forward(self, x_orig, x_orig_atom, y_orig, y_orig_atom, x_orig_length=None, y_orig_length=None, x_atom_length=None, y_atom_length=None):
        """
        x_orig: query torch tensor with batches of padded point coordinates as first 3 features and rest of the optional features (batch, points, features)
        y_orig: target torch tensor with batches of padded point coordinates as first 3 features and rest of the optional features (batch, points, features)
        """
        batch_size: int = x_orig.shape[0]
        assert x_orig_atom.shape[2] == y_orig_atom.shape[2] == 3

        mask_x = lengths_to_key_mask(x_orig, x_orig_length) if x_orig_length is not None else torch.zeros((x_orig.shape[0], x_orig.shape[1])).type(torch.float32).cuda()
        mask_y = lengths_to_key_mask(y_orig, y_orig_length) if y_orig_length is not None else torch.zeros((y_orig.shape[0], y_orig.shape[1])).type(torch.float32).cuda()
        mask_atom_x = lengths_to_key_mask(x_orig_atom, x_atom_length) if x_atom_length is not None else torch.zeros((x_orig_atom.shape[0], x_orig_atom.shape[1])).type(torch.float32).cuda()
        mask_atom_y = lengths_to_key_mask(y_orig_atom, y_atom_length) if y_atom_length is not None else torch.zeros((y_orig_atom.shape[0], y_orig_atom.shape[1])).type(torch.float32).cuda()

        # centre on zero
        x = x_orig - x_orig.mean(1).reshape((-1, 1, x_orig.shape[-1]))
        x_atom = x_orig_atom - x_orig.mean(1).reshape((-1, 1, x_orig.shape[-1]))
        y_translate = y_orig.mean(1).reshape((-1, 1, y_orig.shape[-1]))
        y = y_orig - y_translate
        y_atom = y_orig_atom - y_translate

        xi, yi, xi_atom, yi_atom = self.lin_in(x), self.lin_in(y), self.lin_in_atom(x_atom), self.lin_in_atom(y_atom) # MLP to resize features

        xi_atom = self.atom_attn(xi_atom, xi_atom, xi_atom, key_padding_mask=mask_atom_x)[0]
        yi_atom = self.atom_attn(yi_atom, yi_atom, yi_atom, key_padding_mask=mask_atom_y)[0]
        xi = self.attn(xi, xi, xi, key_padding_mask=mask_x)[0] # query self attentin
        yi = self.attn(yi, yi, yi, key_padding_mask=mask_y)[0] # target self attention

        atom_cross_xi = self.atom_surface_cross(xi_atom, xi, xi, key_padding_mask=mask_x)[0]
        atom_cross_yi = self.atom_surface_cross(yi_atom, yi, yi, key_padding_mask=mask_y)[0]

        cross = self.cross_attn(atom_cross_xi, atom_cross_yi, atom_cross_yi, key_padding_mask=mask_atom_y)[0] # Cross attention between query and target
        coords = self.lin_out(cross.reshape(-1, cross.shape[-1])).reshape((batch_size, -1, 3))
        rot, trans = n_dim_rigid_transform_Kabsch_3D_torch(coords + x_atom, x_atom) # add the relative coords and register original coordinates
        return ((x[:,:,:3] @ rot) + trans) + y_translate[:,:,:3] # rotate and translate with Kabsch output, finally center on the original target coordinates


class PCRBaseAsym(torch.nn.Module):
    def __init__(self, input_dim, attention_dim=12, nheads=4) -> None:
        """
        input_dim: dimension of input features per point.
        """
        super(PCRBaseAsym, self).__init__()
        self.lin_in = torch.nn.Linear(input_dim, attention_dim).cuda()
        self.lin_in_2 = torch.nn.Linear(input_dim, attention_dim).cuda()
        self.attn = MultiheadAttention(attention_dim, nheads, batch_first=True).cuda()
        self.attn_2 = MultiheadAttention(attention_dim, nheads, batch_first=True).cuda()
        self.cross_attn = MultiheadAttention(attention_dim, nheads, batch_first=True).cuda()
        self.lin_out = torch.nn.Linear(attention_dim, 3).cuda()

    def forward(self, x_orig, y_orig):
        """
        x_orig: query torch tensor with batches of padded point coordinates as first 3 features and rest of the optional features (batch, points, features)
        y_orig: target torch tensor with batches of padded point coordinates as first 3 features and rest of the optional features (batch, points, features)
        """
        batch_size: int = x_orig.shape[0]

        # centre on zero
        x = x_orig - x_orig.mean(1).reshape((-1, 1, 3))
        y_translate = y_orig.mean(1).reshape((-1, 1, 3))
        y = y_orig - y_translate

        xi, yi = self.lin_in(x), self.lin_in_2(y) # MLP to resize features
        xi = self.attn(xi, xi, xi)[0] # query self attentin
        yi = self.attn_2(yi, yi, yi)[0] # target self attention
        cross = self.cross_attn(xi, yi, yi)[0] # Cross attention between query and target
        coords = self.lin_out(cross.reshape(-1, cross.shape[-1])).reshape((batch_size, -1, 3))
        rot, trans = n_dim_rigid_transform_Kabsch_3D_torch(coords + x[:, :, :3], x[:, :, :3]) # add the relative coords and register original coordinates
        return ((x[:,:,:3] @ rot) + trans) + y_translate # rotate and translate with Kabsch output, finally center on the original target coordinates


class PCR(pl.LightningModule):
    def __init__(
        self, input_dim: int, lr: float = 1e-3, coarse_attention_dim: int =12, coarse_nheads: int=4, 
        fine_attention_dim: int=12, fine_nheads: int=4, wd=0., validation_data=None) -> None:
        super().__init__()
        self.learning_rate = lr
        self.weight_decay: float = wd
        self.coarse = PCRBase(input_dim, attention_dim=coarse_attention_dim, nheads=coarse_nheads)
        self.fine = PCRBase(3, attention_dim=fine_attention_dim, nheads=fine_nheads)
        self.validation_ransac_distance = torch.Tensor([0.]).cuda()[0]
        if validation_data is not None:
            ransac_scores = []
            for x,y in validation_data:
                for i in range(x.shape[0]):
                    current_x = x[i, :, :3].reshape((-1, 3)).cpu().detach().numpy()
                    current_y = y[i, :, :3].reshape((-1, 3)).cpu().detach().numpy()
                    ransac_result = ransac_registration(current_x, current_y, voxel_size=.65, return_score=False)
                    q, t = get_registration_result(current_x, current_y, ransac_result.transformation)
                    q, t = np.asarray(q.points, dtype="float32"), np.asarray(t.points, dtype="float32")
                    ransac_scores.append(cmf(torch.tensor(q.reshape((1, -1, 3))), torch.tensor(t.reshape((1, -1, 3))))[0].item())
            self.validation_ransac_distance: torch.Tensor = torch.tensor(ransac_scores).cuda().mean()
        self.save_hyperparameters()
        
    def forward(self, x_orig, y_orig, x_len, y_len):
        coords = self.coarse(x_orig, y_orig, x_len, y_len)
        return self.fine(coords, y_orig[:, :, :3])

    def training_step(self, batch, batch_idx):
        x, y, x_len, y_len = batch#[batch_idx]
        pred_xy = self(x, y, x_len, y_len)
        pred_yx = self(y, x, y_len, x_len)
        x_coords = x[:, :, :3]
        y_coords = y[:, :, :3]
        loss = (cmf(pred_xy, y_coords)[0] + cmf(pred_yx, x_coords)[0]) / 2
        self.log("train_chamfer_distance", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y, x_len, y_len = batch#[batch_idx]
        pred_xy = self(x, y, x_len, y_len)
        y_coords = y[:, :, :3]
        x_coords = x[:, :, :3]
        loss = cmf(pred_xy, y_coords)[0]
        init_loss = cmf(y_coords - y_coords.mean(1).reshape((-1, 1, 3)), x_coords - x_coords.mean(1).reshape((-1, 1, 3)))[0]

        # Calling self.log will surface up scalars for you in TensorBoard
        if self.validation_ransac_distance != 0:
            self.log("val_improvements_over_ransac", self.validation_ransac_distance/loss, prog_bar=True)
        self.log("val_chamfer_distance", loss, prog_bar=True)
        self.log("val_improvements_over_random", init_loss/loss, prog_bar=True)

    def test_step(self, batch, batch_idx):
        x, y = batch#[batch_idx]
        pred_xy = self(x, y)
        y_coords = y[:, :, :3]
        x_coords = x[:, :, :3]
        loss = cmf(pred_xy, y_coords)[0]
        init_loss = cmf(y_coords - y_coords.mean(1).reshape((-1, 1, 3)), x_coords - x_coords.mean(1).reshape((-1, 1, 3)))[0]

        self.log("test_chamfer_distance", loss, prog_bar=True)
        self.log("test_improvements_over_random", 1/(loss/init_loss), prog_bar=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        return optimizer


class PCRSepFeat(pl.LightningModule):
    def __init__(
        self, input_dim: int, lr: float = 1e-3, coarse_attention_dim: int =12, coarse_nheads: int=4, wd=0., validation_data=None) -> None:
        super().__init__()
        self.learning_rate = lr
        self.weight_decay: float = wd
        self.coarse = PCRBaseSepFeat(input_dim, attention_dim=coarse_attention_dim, nheads=coarse_nheads)
        self.validation_ransac_distance = torch.Tensor([0.]).cuda()[0]
        if validation_data is not None:
            ransac_scores = []
            for x,y in validation_data:
                for i in range(x.shape[0]):
                    current_x = x[i, :, :3].reshape((-1, 3)).cpu().detach().numpy()
                    current_y = y[i, :, :3].reshape((-1, 3)).cpu().detach().numpy()
                    ransac_result = ransac_registration(current_x, current_y, voxel_size=.65, return_score=False)
                    q, t = get_registration_result(current_x, current_y, ransac_result.transformation)
                    q, t = np.asarray(q.points, dtype="float32"), np.asarray(t.points, dtype="float32")
                    ransac_scores.append(cmf(torch.tensor(q.reshape((1, -1, 3))), torch.tensor(t.reshape((1, -1, 3))))[0].item())
            self.validation_ransac_distance: torch.Tensor = torch.tensor(ransac_scores).cuda().mean()
        self.save_hyperparameters()
        
    def forward(self, x_orig, y_orig):
        coords = self.coarse(x_orig, y_orig)
        return coords #self.fine(coords, y_orig[:, :, :3])

    def training_step(self, batch, batch_idx):
        x, y = batch#[batch_idx]
        pred_xy = self(x, y)
        pred_yx = self(y, x)
        x_coords = x[:, :, :3]
        y_coords = y[:, :, :3]
        loss = (cmf(pred_xy, y_coords)[0] + cmf(pred_yx, x_coords)[0]) / 2
        self.log("train_chamfer_distance", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch#[batch_idx]
        pred_xy = self(x, y)
        y_coords = y[:, :, :3]
        x_coords = x[:, :, :3]
        loss = cmf(pred_xy, y_coords)[0]
        init_loss = cmf(y_coords - y_coords.mean(1).reshape((-1, 1, 3)), x_coords - x_coords.mean(1).reshape((-1, 1, 3)))[0]

        # Calling self.log will surface up scalars for you in TensorBoard
        if self.validation_ransac_distance != 0:
            self.log("val_improvements_over_ransac", self.validation_ransac_distance/loss, prog_bar=True)
        self.log("val_chamfer_distance", loss, prog_bar=True)
        self.log("val_improvements_over_random", init_loss/loss, prog_bar=True)

    def test_step(self, batch, batch_idx):
        x, y = batch#[batch_idx]
        pred_xy = self(x, y)
        y_coords = y[:, :, :3]
        x_coords = x[:, :, :3]
        loss = cmf(pred_xy, y_coords)[0]
        init_loss = cmf(y_coords - y_coords.mean(1).reshape((-1, 1, 3)), x_coords - x_coords.mean(1).reshape((-1, 1, 3)))[0]

        self.log("test_chamfer_distance", loss, prog_bar=True)
        self.log("test_improvements_over_random", 1/(loss/init_loss), prog_bar=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        return optimizer


class PCRSingleAtom(pl.LightningModule):
    def __init__(
        self, input_dim: int, lr: float = 1e-3, coarse_attention_dim: int =12, coarse_nheads: int=4, wd=0., validation_data=None) -> None:
        super().__init__()
        self.learning_rate = lr
        self.weight_decay: float = wd
        self.coarse = PCRBaseAtomAnchor(input_dim, attention_dim=coarse_attention_dim, nheads=coarse_nheads)
        self.validation_ransac_distance = torch.Tensor([0.]).cuda()[0]
        if validation_data is not None:
            ransac_scores = []
            for x,y,xl,yl in validation_data:
                for i in range(x.shape[0]):
                    current_x = x[i, :int(xl[i]), :3].reshape((-1, 3)).cpu().detach().numpy()
                    current_y = y[i, :int(yl[i]), :3].reshape((-1, 3)).cpu().detach().numpy()
                    ransac_result = ransac_registration(current_x, current_y, voxel_size=.65, return_score=False)
                    q, t = get_registration_result(current_x, current_y, ransac_result.transformation)
                    q, t = np.asarray(q.points, dtype="float32"), np.asarray(t.points, dtype="float32")
                    ransac_scores.append(cmf(torch.tensor(q.reshape((1, -1, 3))), torch.tensor(t.reshape((1, -1, 3))))[0].item())
            self.validation_ransac_distance: torch.Tensor = torch.tensor(ransac_scores).cuda().mean()
        self.save_hyperparameters()
        
    def forward(self, x_orig, x_orig_atom, y_orig, y_orig_atom):
        coords = self.coarse(x_orig, x_orig_atom, y_orig, y_orig_atom)
        return coords #self.fine(coords, y_orig[:, :, :3])

    def training_step(self, batch, batch_idx):
        x, x_a, y, y_a = batch#[batch_idx]
        pred_xy = self(x, x_a, y, y_a)
        pred_yx = self(y, y_a, x, x_a)
        x_coords = x[:, :, :3]
        y_coords = y[:, :, :3]
        loss = (cmf(pred_xy, y_coords)[0] + cmf(pred_yx, x_coords)[0]) / 2
        self.log("train_chamfer_distance", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, x_a, y, y_a = batch
        pred_xy = self(x, x_a, y, y_a)
        y_coords = y[:, :, :3]
        x_coords = x[:, :, :3]
        loss = cmf(pred_xy, y_coords)[0]
        init_loss = cmf(y_coords - y_coords.mean(1).reshape((-1, 1, 3)), x_coords - x_coords.mean(1).reshape((-1, 1, 3)))[0]

        # Calling self.log will surface up scalars for you in TensorBoard
        if self.validation_ransac_distance != 0:
            self.log("val_improvements_over_ransac", self.validation_ransac_distance/loss, prog_bar=True)
        self.log("val_chamfer_distance", loss, prog_bar=True)
        self.log("val_improvements_over_random", init_loss/loss, prog_bar=True)

    def test_step(self, batch, batch_idx):
        x, x_a, y, y_a = batch
        pred_xy = self(x, x_a, y, y_a)
        y_coords = y[:, :, :3]
        x_coords = x[:, :, :3]
        loss = cmf(pred_xy, y_coords)[0]
        init_loss = cmf(y_coords - y_coords.mean(1).reshape((-1, 1, 3)), x_coords - x_coords.mean(1).reshape((-1, 1, 3)))[0]

        # Calling self.log will surface up scalars for you in TensorBoard
        if self.validation_ransac_distance != 0:
            self.log("val_improvements_over_ransac", self.validation_ransac_distance/loss, prog_bar=True)
        self.log("val_chamfer_distance", loss, prog_bar=True)
        self.log("val_improvements_over_random", init_loss/loss, prog_bar=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        return optimizer


class PCRSingle(pl.LightningModule):
    def __init__(
        self, input_dim: int, lr: float = 1e-3, coarse_attention_dim: int =12, coarse_nheads: int=4, wd=0., validation_data=None) -> None:
        super().__init__()
        self.learning_rate = lr
        self.weight_decay: float = wd
        self.coarse = PCRBase(input_dim, attention_dim=coarse_attention_dim, nheads=coarse_nheads)
        self.validation_ransac_distance = torch.Tensor([0.]).cuda()[0]
        if validation_data is not None:
            ransac_scores = []
            for x,y in validation_data:
                for i in range(x.shape[0]):
                    current_x = x[i, :, :3].reshape((-1, 3)).cpu().detach().numpy()
                    current_y = y[i, :, :3].reshape((-1, 3)).cpu().detach().numpy()
                    ransac_result = ransac_registration(current_x, current_y, voxel_size=.65, return_score=False)
                    q, t = get_registration_result(current_x, current_y, ransac_result.transformation)
                    q, t = np.asarray(q.points, dtype="float32"), np.asarray(t.points, dtype="float32")
                    ransac_scores.append(cmf(torch.tensor(q.reshape((1, -1, 3))), torch.tensor(t.reshape((1, -1, 3))))[0].item())
            self.validation_ransac_distance: torch.Tensor = torch.tensor(ransac_scores).cuda().mean()
        self.save_hyperparameters()
        
    def forward(self, x_orig, y_orig):
        coords = self.coarse(x_orig, y_orig)
        return coords #self.fine(coords, y_orig[:, :, :3])

    def training_step(self, batch, batch_idx):
        x, y = batch#[batch_idx]
        pred_xy = self(x, y)
        pred_yx = self(y, x)
        x_coords = x[:, :, :3]
        y_coords = y[:, :, :3]
        loss = (cmf(pred_xy, y_coords)[0] + cmf(pred_yx, x_coords)[0]) / 2
        self.log("train_chamfer_distance", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch#[batch_idx]
        pred_xy = self(x, y)
        y_coords = y[:, :, :3]
        x_coords = x[:, :, :3]
        loss = cmf(pred_xy, y_coords)[0]
        init_loss = cmf(y_coords - y_coords.mean(1).reshape((-1, 1, 3)), x_coords - x_coords.mean(1).reshape((-1, 1, 3)))[0]

        # Calling self.log will surface up scalars for you in TensorBoard
        if self.validation_ransac_distance != 0:
            self.log("val_improvements_over_ransac", self.validation_ransac_distance/loss, prog_bar=True)
        self.log("val_chamfer_distance", loss, prog_bar=True)
        self.log("val_improvements_over_random", init_loss/loss, prog_bar=True)

    def test_step(self, batch, batch_idx):
        x, y= batch#[batch_idx]
        pred_xy = self(x, y)
        y_coords = y[:, :, :3]
        x_coords = x[:, :, :3]
        loss = cmf(pred_xy, y_coords)[0]
        init_loss = cmf(y_coords - y_coords.mean(1).reshape((-1, 1, 3)), x_coords - x_coords.mean(1).reshape((-1, 1, 3)))[0]

        self.log("test_chamfer_distance", loss, prog_bar=True)
        self.log("test_improvements_over_random", 1/(loss/init_loss), prog_bar=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        return optimizer


class PCRBaseMasked(torch.nn.Module):
    def __init__(self, input_dim, attention_dim=12, nheads=4) -> None:
        """
        input_dim: dimension of input features per point.
        """
        super(PCRBaseMasked, self).__init__()
        self.lin_in = torch.nn.Linear(input_dim, attention_dim).cuda()
        self.attn = MultiheadAttention(attention_dim, nheads, batch_first=True).cuda()
        self.cross_attn = MultiheadAttention(attention_dim, nheads, batch_first=True).cuda()
        self.lin_out = torch.nn.Linear(attention_dim, 3).cuda()

    def forward(self, x_orig, y_orig, y_lengths=None):
        """
        x_orig: query torch tensor with batches of padded point coordinates as first 3 features and rest of the optional features (batch, points, features)
        y_orig: target torch tensor with batches of padded point coordinates as first 3 features and rest of the optional features (batch, points, features)
        """
        batch_size: int = x_orig.shape[0]

        # key masks to ignore padding

        # centre on zero
        x = x_orig - x_orig.mean(1).reshape((-1, 1, x_orig.shape[-1]))
        y_translate = y_orig.mean(1).reshape((-1, 1, y_orig.shape[-1]))
        y = y_orig - y_translate

        xi, yi = self.lin_in(x), self.lin_in(y) # MLP to resize features
        xi = self.attn(xi, xi, xi)[0] # query self attentin
        if y_lengths is not None:
            mask_y = lengths_to_key_mask(y_orig, y_lengths).cuda()
            yi = self.attn(yi, yi, yi, key_padding_mask=mask_y)[0] # target self attention
            cross = self.cross_attn(xi, yi, yi, key_padding_mask=mask_y)[0] # Cross attention between query and target
        else:
            yi = self.attn(yi, yi, yi)[0] # target self attention
            cross = self.cross_attn(xi, yi, yi)[0] # Cross attention between query and target
        coords = self.lin_out(cross.reshape(-1, cross.shape[-1])).reshape((batch_size, -1, 3))
        rot, trans = n_dim_rigid_transform_Kabsch_3D_torch(coords + x[:, :, :3], x[:, :, :3]) # add the relative coords and register original coordinates
        return ((x[:,:,:3] @ rot) + trans) + y_translate[:,:,:3] # rotate and translate with Kabsch output, finally center on the original target coordinates

    def get_rottrans(self, x_orig, y_orig, y_lengths=None):
        """
        x_orig: query torch tensor with batches of padded point coordinates as first 3 features and rest of the optional features (batch, points, features)
        y_orig: target torch tensor with batches of padded point coordinates as first 3 features and rest of the optional features (batch, points, features)
        """
        batch_size: int = x_orig.shape[0]

        # key masks to ignore padding

        # centre on zero
        x_centroid = x_orig.mean(1).reshape((-1, 1, x_orig.shape[-1]))
        x = x_orig - x_centroid
        # x = x_orig - x_orig.mean(1).reshape((-1, 1, x_orig.shape[-1]))
        y_translate = y_orig.mean(1).reshape((-1, 1, y_orig.shape[-1]))
        y = y_orig - y_translate

        xi, yi = self.lin_in(x), self.lin_in(y) # MLP to resize features
        xi = self.attn(xi, xi, xi)[0] # query self attentin
        if y_lengths is not None:
            mask_y = lengths_to_key_mask(y_orig, y_lengths).cuda()
            yi = self.attn(yi, yi, yi, key_padding_mask=mask_y)[0] # target self attention
            cross = self.cross_attn(xi, yi, yi, key_padding_mask=mask_y)[0] # Cross attention between query and target
        else:
            yi = self.attn(yi, yi, yi)[0] # target self attention
            cross = self.cross_attn(xi, yi, yi)[0] # Cross attention between query and target
        coords = self.lin_out(cross.reshape(-1, cross.shape[-1])).reshape((batch_size, -1, 3))
        rot, trans = n_dim_rigid_transform_Kabsch_3D_torch(coords + x[:, :, :3], x[:, :, :3]) # add the relative coords and register original coordinates
        return rot, trans, y_translate[:,:,:3], x_centroid
        # return ((x[:,:,:3] @ rot) + trans) + y_translate[:,:,:3] # rotate and translate with Kabsch output, finally center on the original target coordinates
    
    def get_pseudo_coords(self, x_orig, y_orig, y_lengths=None):
        batch_size: int = x_orig.shape[0]

        # key masks to ignore padding
        # mask_x = lengths_to_key_mask(x_orig, x_lengths).cuda() if x_lengths is not None else torch.zeros((x_orig.shape[0], x_orig.shape[1])).type(torch.float32).cuda()
        # mask_y = lengths_to_key_mask(y_orig, y_lengths).cuda() if y_lengths is not None else torch.zeros((y_orig.shape[0], y_orig.shape[1])).type(torch.float32).cuda()

        # centre on zero
        x = x_orig - x_orig.mean(1).reshape((-1, 1, x_orig.shape[-1]))
        y_translate = y_orig.mean(1).reshape((-1, 1, y_orig.shape[-1]))
        y = y_orig - y_translate

        xi, yi = self.lin_in(x), self.lin_in(y) # MLP to resize features
        xi = self.attn(xi, xi, xi)[0] # query self attentin
        if y_lengths is not None:
            mask_y = lengths_to_key_mask(y_orig, y_lengths).cuda()
            yi = self.attn(yi, yi, yi, key_padding_mask=mask_y)[0] # target self attention
            cross = self.cross_attn(xi, yi, yi, key_padding_mask=mask_y)[0] # Cross attention between query and target
        else:
            yi = self.attn(yi, yi, yi)[0] # target self attention
            cross = self.cross_attn(xi, yi, yi)[0] # Cross attention between query and target
        coords = self.lin_out(cross.reshape(-1, cross.shape[-1])).reshape((batch_size, -1, 3))
        return coords + y_translate[:,:,:3]


class PCRSingleMasked(pl.LightningModule):
    def __init__(
        self, input_dim: int, lr: float = 1e-3, coarse_attention_dim: int =12, coarse_nheads: int=4, wd=0., validation_data=None) -> None:
        super().__init__()
        self.learning_rate = lr
        self.weight_decay: float = wd
        self.coarse = PCRBaseMasked(input_dim, attention_dim=coarse_attention_dim, nheads=coarse_nheads)
        self.validation_ransac_distance = torch.Tensor([0.]).cuda()[0]
        if validation_data is not None:
            ransac_scores = []
            for x,y in validation_data:
                for i in range(x.shape[0]):
                    current_x = x[i, :, :3].reshape((-1, 3)).cpu().detach().numpy()
                    current_y = y[i, :, :3].reshape((-1, 3)).cpu().detach().numpy()
                    ransac_result = ransac_registration(current_x, current_y, voxel_size=.65, return_score=False)
                    q, t = get_registration_result(current_x, current_y, ransac_result.transformation)
                    q, t = np.asarray(q.points, dtype="float32"), np.asarray(t.points, dtype="float32")
                    ransac_scores.append(cmf(torch.tensor(q.reshape((1, -1, 3))), torch.tensor(t.reshape((1, -1, 3))))[0].item())
            self.validation_ransac_distance: torch.Tensor = torch.tensor(ransac_scores).cuda().mean()
        self.save_hyperparameters()
        
    def forward(self, x_orig, y_orig):
        coords = self.coarse(x_orig, y_orig)
        return coords #self.fine(coords, y_orig[:, :, :3])

    def training_step(self, batch, batch_idx):
        loss = None
        pred_xy = None
        pred_yx = None
        x, y = batch#[batch_idx]
        for i in range(1):
            if pred_xy is None:
                pred_xy = self(x, y)
            else:
                pred_xy = self(pred_xy, y)
            if pred_yx is None:
                pred_yx = self(y, x)
            else:
                pred_yx = self(pred_yx, x)
            x_coords = x[:, :, :3]
            y_coords = y[:, :, :3]
        loss = ((cmf(pred_xy, y_coords)[0] + cmf(pred_yx, x_coords)[0]) / 2) #/ (x_coords.shape[0] + y_coords.shape[0])
            # if loss is None:
            #     loss = (l*(i+1))
            # else:
            #     loss += (l*(i+1))
        self.log("train_chamfer_distance", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        pred_xy = None
        x, y = batch#[batch_idx]
        for i in range(1):
            if pred_xy is None:
                pred_xy = self(x, y)
            else:
                pred_xy = self(pred_xy, y)
        x_coords = x[:, :, :3]
        y_coords = y[:, :, :3]
        loss = cmf(pred_xy, y_coords)[0]
        init_loss = cmf(x_coords, y_coords)[0]

        # Calling self.log will surface up scalars for you in TensorBoard
        if self.validation_ransac_distance != 0:
            self.log("val_improvements_over_ransac", self.validation_ransac_distance/loss, prog_bar=True)
        self.log("val_chamfer_distance", loss, prog_bar=True)
        self.log("val_improvements_over_random", init_loss/loss, prog_bar=True)

    def test_step(self, batch, batch_idx):
        x, y= batch#[batch_idx]
        pred_xy = self(x, y)
        y_coords = y[:, :, :3]
        x_coords = x[:, :, :3]
        loss = cmf(pred_xy, y_coords)[0]
        init_loss = cmf(y_coords - y_coords.mean(1).reshape((-1, 1, 3)), x_coords - x_coords.mean(1).reshape((-1, 1, 3)))[0]

        self.log("test_chamfer_distance", loss, prog_bar=True)
        self.log("test_improvements_over_random", 1/(loss/init_loss), prog_bar=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        return optimizer


def create_attention_mask(padded_1, padded_2, lengths_1, lengths_2):
    assert padded_1.shape[0] == padded_2.shape[0]
    assert (torch.max(lengths_2) <= padded_2.shape[1]) and (torch.max(lengths_1) <= padded_1.shape[1])
    all_masks = torch.zeros((padded_1.shape[0], padded_1.shape[1], padded_2.shape[1]))
    for i in range(padded_1.shape[0]):
        all_masks[i, int(-lengths_1[i]):, int(-lengths_2[i])] = 1
    return all_masks.cuda()

def lengths_to_key_mask(padded, lengths):
    mask = torch.zeros((padded.shape[0], padded.shape[1]))
    for i in range(padded.shape[0]):
        mask[i, int(lengths[i]):] = 1
    return mask


class DataLoader:
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

