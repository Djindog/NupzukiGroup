import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import cupy as cp
from cuml.cluster import DBSCAN as cuDBSCAN
from torch_cluster import radius_graph, radius as tc_radius

import spconv.pytorch as spconv
from spconv.pytorch.utils import PointToVoxel

import warnings


def _gpu_clustering(coords, radius):
    """Fallback: connected components via label propagation on a GPU radius graph."""
    n = coords.shape[0]
    device = coords.device
    edge_index = radius_graph(coords, r=radius, max_num_neighbors=128)
    src, dst = edge_index[0], edge_index[1]
    
    labels = torch.arange(n, dtype=torch.long, device=device)
    for _ in range(100):
        prev = labels.clone()
        labels.scatter_reduce_(0, src, labels[dst], reduce='amin', include_self=True)
        labels.scatter_reduce_(0, dst, labels[src], reduce='amin', include_self=True)
        if (labels == prev).all():
            break
    return labels


# ── Sparse U-Net building blocks ───────────────────────────────────────────────

def _make_subm_block(in_ch, out_ch, key):
    return spconv.SparseSequential(
        spconv.SubMConv3d(in_ch, out_ch, 3, padding=1, bias=False, indice_key=key),
        nn.BatchNorm1d(out_ch, eps=1e-3, momentum=0.01),
        nn.ReLU(inplace=True),
        spconv.SubMConv3d(out_ch, out_ch, 3, padding=1, bias=False, indice_key=key),
        nn.BatchNorm1d(out_ch, eps=1e-3, momentum=0.01),
        nn.ReLU(inplace=True),
    )


def _make_down(in_ch, out_ch, key):
    return spconv.SparseSequential(
        spconv.SparseConv3d(in_ch, out_ch, 3, stride=2, padding=1, bias=False, indice_key=key),
        nn.BatchNorm1d(out_ch, eps=1e-3, momentum=0.01),
        nn.ReLU(inplace=True),
    )


def _make_dec_block(in_ch, out_ch, key):
    return spconv.SparseSequential(
        spconv.SubMConv3d(in_ch, out_ch, 3, padding=1, bias=False, indice_key=key),
        nn.BatchNorm1d(out_ch, eps=1e-3, momentum=0.01),
        nn.ReLU(inplace=True),
    )


def _make_head(in_ch, hidden_ch, out_ch):
    return nn.Sequential(
        nn.Linear(in_ch, hidden_ch),
        nn.ReLU(inplace=True),
        nn.Linear(hidden_ch, out_ch),
    )


# ── Main model ─────────────────────────────────────────────────────────────────

class SoftGroupModel(nn.Module):
    SPATIAL_SHAPE = [220, 220, 220]
    VOXEL_SIZE = 0.01
    COORD_RANGE = [-1.1, -1.1, -1.1, 1.1, 1.1, 1.1]

    def __init__(self, in_channels=9, m=64, use_xyz_offset=False):
        super().__init__()
        self.use_xyz_offset = use_xyz_offset
        self._voxelizer = None
        self._voxelizer_device = None

        # Encoder
        self.input_conv = spconv.SparseSequential(
            spconv.SubMConv3d(in_channels, m, 3, padding=1, bias=False,
                              indice_key='subm_input'),
            nn.BatchNorm1d(m, eps=1e-3, momentum=0.01),
            nn.ReLU(inplace=True),
        )
        self.enc0  = _make_subm_block(m,    m,    'subm0')
        self.down1 = _make_down(m,    m*2,  'down1')
        self.enc1  = _make_subm_block(m*2,  m*2,  'subm1')
        self.down2 = _make_down(m*2,  m*4,  'down2')
        self.enc2  = _make_subm_block(m*4,  m*4,  'subm2')
        self.down3 = _make_down(m*4,  m*8,  'down3')
        self.enc3  = _make_subm_block(m*8,  m*8,  'subm3')

        # Decoder
        self.up3   = spconv.SparseInverseConv3d(m*8, m*4, 3, indice_key='down3', bias=False)
        self.bn_up3 = nn.BatchNorm1d(m*4, eps=1e-3, momentum=0.01)
        self.dec3  = _make_dec_block(m*8, m*4, 'subm_dec3')

        self.up2   = spconv.SparseInverseConv3d(m*4, m*2, 3, indice_key='down2', bias=False)
        self.bn_up2 = nn.BatchNorm1d(m*2, eps=1e-3, momentum=0.01)
        self.dec2  = _make_dec_block(m*4, m*2, 'subm_dec2')

        self.up1   = spconv.SparseInverseConv3d(m*2, m,   3, indice_key='down1', bias=False)
        self.bn_up1 = nn.BatchNorm1d(m, eps=1e-3, momentum=0.01)
        self.dec1  = _make_dec_block(m*2, m, 'subm_dec1')

        # Per-point prediction heads
        self.semantic_head = _make_head(m, m, 2)
        self.offset_head   = _make_head(m + 3 if use_xyz_offset else m, m, 3)

        # Per-proposal heads
        self.cls_head   = _make_head(m, m, 2)
        self.seg_head   = _make_head(m, m, 1)
        self.score_head = _make_head(m, m, 1)

        # Bias seg_head to pass all points at init; head gradually learns to filter
        # self.seg_head[-1].bias.data.fill_(2.0)

    # ── Voxelization ────────────────────────────────────────────────────────

    def _get_voxelizer(self, device):
        if self._voxelizer is None or self._voxelizer_device != device:
            self._voxelizer = PointToVoxel(
                vsize_xyz=[self.VOXEL_SIZE] * 3,
                coors_range_xyz=self.COORD_RANGE,
                num_point_features=9,
                max_num_voxels=500_000,
                max_num_points_per_voxel=5,
                device=device,
            )
            self._voxelizer_device = device
        return self._voxelizer

    def _voxelize(self, feat_N9):
        device = feat_N9.device
        vg = self._get_voxelizer(device)
        voxels, indices, num_per_vox, pc_voxel_id = vg.generate_voxel_with_id(feat_N9)
        voxel_feats = voxels.sum(1) / num_per_vox.float().unsqueeze(1).clamp(min=1)
        batch_col = torch.zeros(voxel_feats.shape[0], 1, dtype=torch.int32, device=device)
        indices_4d = torch.cat([batch_col, indices], dim=1)
        sp = spconv.SparseConvTensor(
            features=voxel_feats,
            indices=indices_4d,
            spatial_shape=self.SPATIAL_SHAPE,
            batch_size=1,
        )
        return sp, pc_voxel_id

    # ── Backbone ────────────────────────────────────────────────────────────

    def _backbone(self, sp):
        x = self.input_conv(sp)
        e0 = self.enc0(x)

        x = self.down1(e0)
        e1 = self.enc1(x)

        x = self.down2(e1)
        e2 = self.enc2(x)

        x = self.down3(e2)
        e3 = self.enc3(x)

        # Decoder
        d = self.up3(e3)
        d = d.replace_feature(F.relu(self.bn_up3(d.features), inplace=True))
        d = d.replace_feature(torch.cat([d.features, e2.features], dim=1))
        d = self.dec3(d)

        d = self.up2(d)
        d = d.replace_feature(F.relu(self.bn_up2(d.features), inplace=True))
        d = d.replace_feature(torch.cat([d.features, e1.features], dim=1))
        d = self.dec2(d)

        d = self.up1(d)
        d = d.replace_feature(F.relu(self.bn_up1(d.features), inplace=True))
        d = d.replace_feature(torch.cat([d.features, e0.features], dim=1))
        d = self.dec1(d)

        return d

    # ── Devoxelization ──────────────────────────────────────────────────────

    def _devoxelize(self, vox_feats, pc_voxel_id, N):
        device = vox_feats.device
        point_feats = torch.zeros(N, vox_feats.shape[1], device=device)
        valid = pc_voxel_id >= 0
        point_feats[valid] = vox_feats[pc_voxel_id[valid].long()]
        return point_feats

    # ── Forward (backbone + per-point heads) ────────────────────────────────

    def forward(self, feat_N9):
        N = feat_N9.shape[0]
        sp, pc_voxel_id = self._voxelize(feat_N9)
        backbone_out = self._backbone(sp)
        point_feats = self._devoxelize(backbone_out.features, pc_voxel_id, N)
        sem_logits  = self.semantic_head(point_feats)
        if self.use_xyz_offset:
            offset_pred = self.offset_head(torch.cat([point_feats, feat_N9[:, :3]], dim=1))
        else:
            offset_pred = self.offset_head(point_feats)
        return {
            'point_feats': point_feats,
            'sem_logits':  sem_logits,
            'offset_pred': offset_pred,
        }

    # ── Grouping + proposal scoring ─────────────────────────────────────────

    def _cluster(self, xyz, sem_logits, offset_pred,
                 fg_threshold=0.2, radius=0.02, min_cluster_size=400,
                 use_offset=True, max_cluster_pts=15_000):
        sem_prob = torch.softmax(sem_logits.detach(), dim=1)
        fg_mask  = sem_prob[:, 1] >= fg_threshold
        if not fg_mask.any():
            return []
        fg_idx  = torch.where(fg_mask)[0]
        offset  = offset_pred[fg_mask].detach() if use_offset else torch.zeros_like(xyz[fg_mask])
        shifted = xyz[fg_mask] + offset
        n_fg    = len(fg_idx)

        # Subsample for DBSCAN; O(N²) → O(max_cluster_pts²)
        do_expand = n_fg > max_cluster_pts
        if do_expand:
            sub         = torch.randperm(n_fg, device=xyz.device)[:max_cluster_pts]
            shifted_sub = shifted[sub]
            sub_ratio   = max_cluster_pts / n_fg
        else:
            shifted_sub = shifted
            sub_ratio   = 1.0

        db         = cuDBSCAN(eps=radius, min_samples=6, output_type='cupy', verbose=False)
        labels_sub = torch.as_tensor(db.fit_predict(shifted_sub.detach().contiguous()),
                                     device=shifted_sub.device)

        # Pre-filter threshold: scale min_cluster_size by sampling ratio so we
        # don't call tc_radius on clusters that will be too small after expansion.
        pre_min = max(6, int(min_cluster_size * sub_ratio))

        proposals = []
        for lbl in labels_sub.unique():
            if lbl.item() < 0:
                continue
            members_sub = torch.where(labels_sub == lbl)[0]
            if len(members_sub) < pre_min:
                continue

            if do_expand:
                # Re-expand: for every fg point, check proximity to individual
                # cluster members (not a centroid sphere) — respects actual shape.
                # tc_radius uses spatial hashing: O(N_fg·logK) vs O(N_fg·K) for cdist.
                cluster_pts = shifted_sub[members_sub]          # (K, 3)
                row, _      = tc_radius(cluster_pts, shifted,
                                        r=radius, max_num_neighbors=1)
                if len(row) == 0:
                    continue
                expanded = fg_idx[row.unique()]
            else:
                expanded = fg_idx[members_sub]

            if len(expanded) >= min_cluster_size:
                proposals.append(expanded)

        return proposals

    # ── Full inference ───────────────────────────────────────────────────────

    def predict(self, feat_N9, _outputs=None):
        N = feat_N9.shape[0]
        device = feat_N9.device
        outputs = _outputs if _outputs is not None else self.forward(feat_N9)
        point_feats = outputs['point_feats']
        sem_logits  = outputs['sem_logits']
        offset_pred = outputs['offset_pred']
        xyz = feat_N9[:, :3]

        use_offset = getattr(self, 'use_offset_clustering', True)
        proposals = self._cluster(xyz, sem_logits, offset_pred, fg_threshold=0.2, use_offset=use_offset)
        result = torch.zeros(N, dtype=torch.long, device=device)
        if not proposals:
            return result

        scored = []
        for P_i in proposals:
            pool = point_feats[P_i].mean(0)
            cls_l  = self.cls_head(pool)
            score  = self.score_head(pool)
            conf   = torch.softmax(cls_l, dim=0)[1] * torch.sigmoid(score.squeeze())
            scored.append((conf.item(), P_i))

        scored.sort(key=lambda x: x[0], reverse=True)

        inst_id = 1
        for conf, P_i in scored:
            if inst_id > 5:
                break
            mask_logit = self.seg_head(point_feats[P_i])   # [k, 1]
            keep = mask_logit.squeeze(1) > 0               # filtered by seg_head
            refined = P_i[keep]
            if len(refined) == 0:
                continue
            result[refined] = inst_id
            inst_id += 1
        return result


# ── Evaluator-compatible interface ─────────────────────────────────────────────

def initialize_model(ckpt_path: str, device: torch.device, **kwargs) -> nn.Module:
    checkpoint = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    if isinstance(checkpoint, dict):
        state = (checkpoint.get('model_state_dict')
                 or checkpoint.get('state_dict')
                 or checkpoint)
        use_xyz = checkpoint.get('use_xyz_offset', False)
    else:
        state = checkpoint
        use_xyz = False
    state = {k.replace('module.', '', 1): v for k, v in state.items()}
    model = SoftGroupModel(use_xyz_offset=use_xyz)
    missing, unexpected = model.load_state_dict(state, strict=False)
    # Only warn if something other than the expected offset_head mismatch is missing
    unexpected_missing = [k for k in missing if 'offset_head' not in k]
    if unexpected_missing:
        print(f'[initialize_model] WARNING: unexpected missing keys: {unexpected_missing}')
    if missing:
        print(f'[initialize_model] offset_head randomly initialised '
              f'(use_xyz_offset={use_xyz}, {len(missing)} keys)')
    model.to(device)
    model.eval()
    return model


def run_inference(model: nn.Module, features: torch.Tensor, **kwargs) -> torch.Tensor:
    """
    features: [B, 9, N]
    returns:  [B, N] long tensor of instance labels (0=bg, 1-5=instances)
    """
    B = features.shape[0]
    results = []
    for b in range(B):
        feat_N9 = features[b].T.contiguous()
        pred = model.predict(feat_N9)
        results.append(pred)
    return torch.stack(results, dim=0)
