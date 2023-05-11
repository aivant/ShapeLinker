

# start with the multi-head attention model using pytorch lightning.
# This should accept molecule object as batches
    # Same number of atoms but padded surface

import torch
import pytorch_lightning as pl
from pytorch3d.loss import chamfer_distance as cmf
from torch.nn import MultiheadAttention
from structural.dmasif_pcg.geometry_processing import ransac_registration, get_registration_result
import numpy as np


def reflect(U, R, Vt, A):
    SS = torch.zeros((R.shape[0], 3, 3)).cuda()
    for i, r in enumerate(R):
        if torch.linalg.det(r) < 0:
            SS[i] = torch.diag(torch.tensor([1.,1., 1.], device=A.device))
        else:
            SS[i] = torch.diag(torch.tensor([1.,1., 1.], device=A.device))

    R =  torch.bmm(U, torch.bmm(torch.transpose(SS, 1, 2), Vt))

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

    def get_pseudo_coords(self, x_orig, y_orig, y_lengths=None):
        batch_size: int = x_orig.shape[0]

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
        return coords

    def training_step(self, batch, batch_idx):
        loss = None
        pred_xy = None
        pred_yx = None
        x, y = batch
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
        loss = ((cmf(pred_xy, y_coords)[0] + cmf(pred_yx, x_coords)[0]) / 2)
        self.log("train_chamfer_distance", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        pred_xy = None
        x, y = batch
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
        x, y= batch
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

