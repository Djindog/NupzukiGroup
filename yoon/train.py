#!/usr/bin/env python3
import os, sys
os.environ.setdefault('PYTHONUNBUFFERED', '1')
sys.stdout.reconfigure(line_buffering=True)
"""
SoftGroup training script for Nubzuki instance segmentation.

Usage (from CS479-Seg root):
    conda run -n 3d-seg python yoon/train.py \
        --data-dir yoon/generated_data \
        --ckpt-dir yoon/checkpoints \
        --epochs 30 --lr 1e-3

    # Resume from checkpoint
    conda run -n 3d-seg python yoon/train.py \
        --data-dir yoon/generated_data \
        --ckpt-dir yoon/checkpoints \
        --epochs 30 --resume yoon/checkpoints/best_model.pth
"""

import argparse
import datetime
import json
import os
import sys
import time
from collections import defaultdict

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from dataset import InstancePointCloudDataset
from model import SoftGroupModel


# ── Loss helpers ───────────────────────────────────────────────────────────────

def compute_gt_offsets(xyz, instance_labels):
    """Vectorised: per-point offset toward instance centre. Returns (offset_gt, fg_mask)."""
    offset_gt = torch.zeros_like(xyz)
    fg_mask = instance_labels > 0
    if not fg_mask.any():
        return offset_gt, fg_mask
    fg_xyz = xyz[fg_mask]
    fg_ids = instance_labels[fg_mask]
    _, inv  = torch.unique(fg_ids, return_inverse=True)
    K       = int(inv.max().item()) + 1
    csum    = torch.zeros(K, 3, device=xyz.device)
    count   = torch.zeros(K,    device=xyz.device)
    csum.scatter_add_(0, inv.unsqueeze(1).expand(-1, 3), fg_xyz)
    count.scatter_add_(0, inv, torch.ones(len(fg_xyz), device=xyz.device))
    centers = csum / count.unsqueeze(1).clamp(min=1)
    offset_gt[fg_mask] = centers[inv] - fg_xyz
    return offset_gt, fg_mask


def compute_offset_losses(xyz, instance_labels, offset_pred):
    """Per-instance normalised smooth-L1 + directional cosine loss."""
    device = xyz.device
    offset_gt, fg_mask = compute_gt_offsets(xyz, instance_labels)
    if not fg_mask.any():
        zero = torch.tensor(0.0, device=device)
        return zero, zero

    fg_pred = offset_pred[fg_mask]
    fg_gt   = offset_gt[fg_mask]
    fg_ids  = instance_labels[fg_mask]
    _, inv  = torch.unique(fg_ids, return_inverse=True)
    K       = int(inv.max().item()) + 1
    inst_cnt = torch.zeros(K, device=device).scatter_add_(
        0, inv, torch.ones(len(fg_ids), device=device))

    # Per-instance normalised smooth-L1
    pt_l1   = F.smooth_l1_loss(fg_pred, fg_gt, reduction='none').mean(1)
    inst_l1 = torch.zeros(K, device=device).scatter_add_(0, inv, pt_l1)
    L_off   = (inst_l1 / inst_cnt.clamp(min=1)).mean()

    # Directional loss: cosine angle to GT offset direction.
    # Masked at near-centroid points where GT magnitude ≈ 0 (cosine undefined).
    # L_cons (shifted-point variance) was tried but doesn't prevent bridging: the
    # failure mode is wrong mean direction, not high spread — L_dir targets that directly.
    gt_mag   = fg_gt.norm(dim=1)
    dir_mask = gt_mag > 1e-3
    if dir_mask.any():
        L_dir = (1.0 - F.cosine_similarity(fg_pred[dir_mask], fg_gt[dir_mask], dim=1)).mean()
    else:
        L_dir = torch.tensor(0.0, device=device)

    return L_off, L_dir


def compute_proposal_losses(model, point_feats, xyz, sem_logits, offset_pred,
                             instance_labels, max_proposals=20, backward_every=4, epoch=0):
    """
    Compute proposal-head losses, calling backward() every `backward_every` proposals.
    Pool features, IoU matrix, and centroid distances are pre-computed in batch.
    `point_feats` must already be detached from the backbone graph before calling.
    """
    device   = point_feats.device
    feat_dim = point_feats.shape[1]

    with torch.no_grad():
        proposals = model._cluster(xyz, sem_logits, offset_pred,
                                   fg_threshold=0.2, use_offset=(epoch >= 3))
        if len(proposals) > max_proposals:
            perm = torch.randperm(len(proposals))[:max_proposals]
            proposals = [proposals[i] for i in perm.tolist()]

    if not proposals:
        return 0.0

    P = len(proposals)

    with torch.no_grad():
        gt_id_list = instance_labels.unique()
        gt_id_list = gt_id_list[gt_id_list > 0].tolist()
        G = len(gt_id_list)

        # Batch pool: one scatter_add for all proposals
        all_pts = torch.cat(proposals)
        assigns = torch.cat([torch.full((len(p),), j, dtype=torch.long, device=device)
                              for j, p in enumerate(proposals)])
        pool_sum = torch.zeros(P, feat_dim, device=device)
        pool_sum.scatter_add_(0, assigns.unsqueeze(1).expand(-1, feat_dim),
                               point_feats[all_pts])
        prop_sizes = torch.tensor([len(p) for p in proposals],
                                   dtype=torch.float, device=device)
        pool_feats = pool_sum / prop_sizes.unsqueeze(1).clamp(min=1)   # (P, C)

        # Batch IoU: label-count scatter
        all_gtlbl = instance_labels[all_pts]
        iou_inter = torch.zeros(P, G, device=device)
        for g, gt_id in enumerate(gt_id_list):
            iou_inter[:, g].scatter_add_(0, assigns, (all_gtlbl == gt_id).float())
        gt_sizes = torch.tensor([(instance_labels == gid).sum().item()
                                  for gid in gt_id_list],
                                 dtype=torch.float, device=device)
        union    = prop_sizes.unsqueeze(1) + gt_sizes.unsqueeze(0) - iou_inter
        iou_mat  = iou_inter / union.clamp(min=1)
        best_iou_vals, best_gt_cols = iou_mat.max(dim=1)

        # Batch centroid distances: one cdist call for all proposals
        centroids = torch.stack([xyz[p].mean(0) for p in proposals])
        r_expands = torch.stack([
            torch.norm(xyz[p] - centroids[j], dim=1).max() * 1.5
            for j, p in enumerate(proposals)
        ])
        all_dists = torch.cdist(xyz, centroids)   # (N, P)

    total_loss = 0.0
    batch_loss = None

    for idx in range(P):
        best_iou   = best_iou_vals[idx].item()
        best_gt_id = gt_id_list[best_gt_cols[idx].item()] if G > 0 else -1

        cls_logit  = model.cls_head(pool_feats[idx])
        score_pred = model.score_head(pool_feats[idx])

        cls_target = torch.tensor([1 if best_iou > 0.5 else 0], device=device)
        prop_loss  = F.cross_entropy(cls_logit.unsqueeze(0), cls_target)

        if best_iou > 0.5:
            with torch.no_grad():
                ext_idx    = torch.where(all_dists[:, idx] <= r_expands[idx])[0]
                gt_seg     = (instance_labels[ext_idx] == best_gt_id).float().unsqueeze(1)
                gt_iou_t   = torch.tensor([[best_iou]], dtype=torch.float, device=device)
                n_fg       = gt_seg.sum().clamp(min=1)
                n_bg       = (1 - gt_seg).sum().clamp(min=1)
                pos_weight = torch.tensor([n_bg / n_fg], device=device)

            seg_logit = model.seg_head(point_feats[ext_idx])
            prop_loss = prop_loss + F.binary_cross_entropy_with_logits(
                seg_logit, gt_seg, pos_weight=pos_weight)
            prop_loss = prop_loss + F.mse_loss(
                torch.sigmoid(score_pred.unsqueeze(0)), gt_iou_t)

        batch_loss = prop_loss if batch_loss is None else batch_loss + prop_loss

        is_last     = (idx == P - 1)
        is_boundary = ((idx + 1) % backward_every == 0)
        if is_boundary or is_last:
            batch_loss.backward()
            total_loss += batch_loss.item()
            batch_loss  = None

    return total_loss / P


# ── Log / curve helpers ────────────────────────────────────────────────────────

def _append_log(log_path, entry):
    with open(log_path, 'a') as f:
        f.write(json.dumps(entry) + '\n')


def _save_curve(log_path, curve_path):
    entries = []
    try:
        with open(log_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    entries.append(json.loads(line))
    except FileNotFoundError:
        return
    if not entries:
        return

    epochs     = [e['epoch']      for e in entries]
    tr_loss    = [e['train_loss'] for e in entries]
    total_hrs  = [e['total_s'] / 3600.0 for e in entries]

    eval_entries = [e for e in entries if e.get('is_eval')]
    eval_epochs  = [e['epoch']      for e in eval_entries]
    f1_25        = [e['val_f1_25'] for e in eval_entries]
    f1_50        = [e['val_f1_50'] for e in eval_entries]
    off_err      = [e['offset_err'] for e in eval_entries]

    # Collect per-prefix F1@25 series (only present when using mix data)
    PREFIX_STYLE = {'yoon': ('#2ca02c', 'o'), 'cjm': ('#1f77b4', 's'), 'seyoon': ('#ff7f0e', '^')}
    pfx_series = {}
    for pfx in PREFIX_STYLE:
        key = f'f1_25_{pfx}'
        vals = [(e['epoch'], e[key]) for e in eval_entries if key in e]
        if vals:
            pfx_series[pfx] = vals

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle('Training progress', fontsize=13)

    axes[0, 0].plot(epochs, tr_loss, 'b-o', ms=3, lw=1.2)
    axes[0, 0].set_title('Train loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].grid(True, alpha=0.4)

    axes[0, 1].plot(eval_epochs, f1_25, 'k-o', ms=5, lw=2, label='overall@25')
    axes[0, 1].plot(eval_epochs, f1_50, 'k--o', ms=5, lw=1.5, label='overall@50')
    for pfx, vals in pfx_series.items():
        col, mk = PREFIX_STYLE[pfx]
        ep_v, f_v = zip(*vals)
        axes[0, 1].plot(ep_v, f_v, linestyle='--', marker=mk, ms=4, lw=1.2,
                        color=col, alpha=0.85, label=f'{pfx}@25')
    axes[0, 1].set_title('Validation F1')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylim(0, 1)
    axes[0, 1].legend(fontsize=8)
    axes[0, 1].grid(True, alpha=0.4)

    axes[1, 0].plot(eval_epochs, off_err, 'm-o', ms=5)
    axes[1, 0].set_title('Offset error (relative)')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].grid(True, alpha=0.4)

    axes[1, 1].plot(epochs, total_hrs, 'k-', lw=1.5)
    axes[1, 1].set_title('Cumulative training time')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Hours')
    axes[1, 1].grid(True, alpha=0.4)

    plt.tight_layout()
    plt.savefig(curve_path, dpi=120, bbox_inches='tight')
    plt.close()


# ── Training / validation loops ────────────────────────────────────────────────

def train_one_epoch(model, loader, optimizer, device, epoch):
    model.train()
    total_loss = 0.0

    for batch in tqdm(loader, desc=f'Train E{epoch}', leave=False):
        features        = batch['features'].to(device)
        instance_labels = batch['instance_labels'][0].to(device)

        feat_N9 = features[0].T.contiguous()
        xyz     = feat_N9[:, :3]

        outputs     = model.forward(feat_N9)
        point_feats = outputs['point_feats']
        sem_logits  = outputs['sem_logits']
        offset_pred = outputs['offset_pred']

        # Semantic loss with class weighting; clamped to avoid extreme imbalance
        sem_gt    = (instance_labels > 0).long()
        fg_count  = (sem_gt == 1).sum().float().clamp(min=1)
        bg_count  = (sem_gt == 0).sum().float().clamp(min=1)
        fg_weight = min((bg_count / fg_count).item(), 20.0)
        sem_weight = torch.tensor([1.0, fg_weight], device=device)
        L_sem = F.cross_entropy(sem_logits, sem_gt, weight=sem_weight)

        L_off, L_dir = compute_offset_losses(xyz, instance_labels, offset_pred)
        backbone_loss = L_sem + L_off + 0.5 * L_dir

        optimizer.zero_grad()
        backbone_loss.backward()

        prop_loss_val = 0.0
        if epoch >= 3:
            prop_loss_val = compute_proposal_losses(
                model, point_feats.detach(), xyz,
                sem_logits.detach(), offset_pred.detach(), instance_labels,
                epoch=epoch,
            )

        # Clip after both backbone and proposal backward passes
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
        optimizer.step()
        total_loss += backbone_loss.item() + prop_loss_val

    return total_loss / max(len(loader), 1)


def _iou_masks(m1, m2):
    inter = np.logical_and(m1, m2).sum()
    union = np.logical_or(m1, m2).sum()
    return inter / union if union > 0 else 0.0


def _scene_prefix(scene_path):
    """Extract generator prefix (yoon/cjm/seyoon) from file name, or 'other'."""
    name = os.path.basename(scene_path)
    for p in ('yoon_', 'cjm_', 'seyoon_'):
        if name.startswith(p):
            return p[:-1]
    return 'other'


def validate(model, loader, device):
    model.eval()
    total_tp25, total_fp25, total_fn25 = 0, 0, 0
    total_tp50, total_fp50, total_fn50 = 0, 0, 0
    total_off_err = 0.0
    total_fg_pts  = 0

    pfx_tp25 = defaultdict(int);  pfx_fp25 = defaultdict(int);  pfx_fn25 = defaultdict(int)
    pfx_tp50 = defaultdict(int);  pfx_fp50 = defaultdict(int);  pfx_fn50 = defaultdict(int)

    with torch.inference_mode():
        for batch in tqdm(loader, desc='Val', leave=False):
            features            = batch['features'].to(device)
            instance_labels_dev = batch['instance_labels'][0].to(device)
            instance_labels     = instance_labels_dev.cpu().numpy().astype(np.int64)
            prefix              = _scene_prefix(batch['scene_path'][0])

            feat_N9 = features[0].T.contiguous()
            xyz     = feat_N9[:, :3]

            # Single forward pass shared by offset-error diagnostic and predict()
            outputs     = model.forward(feat_N9)
            offset_pred = outputs['offset_pred']

            off_gt, fg_mask = compute_gt_offsets(xyz, instance_labels_dev)
            if fg_mask.any():
                n_fg     = int(fg_mask.sum().item())
                pred_err = (offset_pred[fg_mask] - off_gt[fg_mask]).norm(dim=1)
                gt_mag   = off_gt[fg_mask].norm(dim=1).clamp(min=1e-6)
                off_err  = (pred_err / gt_mag).mean().item()
                total_off_err += off_err * n_fg
                total_fg_pts  += n_fg

            pred = model.predict(feat_N9, _outputs=outputs).cpu().numpy().astype(np.int64)

            pred_ids   = [int(x) for x in np.unique(pred)            if x > 0]
            gt_ids     = [int(x) for x in np.unique(instance_labels) if x > 0]
            pred_masks = [(pred == i)            for i in pred_ids]
            gt_masks   = [(instance_labels == i) for i in gt_ids]

            K, M = len(pred_masks), len(gt_masks)
            if K == 0 or M == 0:
                total_fn25 += M;  total_fp25 += K
                total_fn50 += M;  total_fp50 += K
                pfx_fn25[prefix] += M;  pfx_fp25[prefix] += K
                pfx_fn50[prefix] += M;  pfx_fp50[prefix] += K
                continue

            iou_mat = np.zeros((K, M), dtype=np.float32)
            for i, pm in enumerate(pred_masks):
                for j, gm in enumerate(gt_masks):
                    iou_mat[i, j] = _iou_masks(pm, gm)

            row_ind, col_ind = linear_sum_assignment(1.0 - iou_mat)
            matched = iou_mat[row_ind, col_ind]

            tp25 = int(np.sum(matched >= 0.25))
            total_tp25 += tp25;  total_fp25 += K - tp25;  total_fn25 += M - tp25
            pfx_tp25[prefix] += tp25;  pfx_fp25[prefix] += K - tp25;  pfx_fn25[prefix] += M - tp25

            tp50 = int(np.sum(matched >= 0.50))
            total_tp50 += tp50;  total_fp50 += K - tp50;  total_fn50 += M - tp50
            pfx_tp50[prefix] += tp50;  pfx_fp50[prefix] += K - tp50;  pfx_fn50[prefix] += M - tp50

    def _f1(tp, fp, fn):
        d = 2 * tp + fp + fn
        return (2 * tp / d) if d > 0 else 0.0

    prefix_f1s = {
        pfx: (_f1(pfx_tp25[pfx], pfx_fp25[pfx], pfx_fn25[pfx]),
              _f1(pfx_tp50[pfx], pfx_fp50[pfx], pfx_fn50[pfx]))
        for pfx in sorted(set(pfx_tp25) | set(pfx_fn25))
    }

    mean_off_err = total_off_err / max(total_fg_pts, 1)
    return (_f1(total_tp25, total_fp25, total_fn25),
            _f1(total_tp50, total_fp50, total_fn50),
            mean_off_err,
            prefix_f1s)


# ── Entry point ────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--data-dir',       default='yoon/generated_data')
    p.add_argument('--ckpt-dir',       default='yoon/checkpoints')
    p.add_argument('--epochs',         type=int, default=30)
    p.add_argument('--lr',             type=float, default=1e-3)
    p.add_argument('--resume',         default=None, help='checkpoint path to resume from')
    p.add_argument('--use-xyz-offset', action='store_true',
                   help='Concatenate absolute xyz to offset head input (smooth offsets within voxels)')
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.ckpt_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    log_path   = os.path.join(args.ckpt_dir, 'training_log.jsonl')
    curve_path = os.path.join(args.ckpt_dir, 'training_curve.png')

    train_dir = os.path.join(args.data_dir, 'train')
    val_dir   = os.path.join(args.data_dir, 'val')
    train_ds  = InstancePointCloudDataset(train_dir, split='all')
    val_ds    = InstancePointCloudDataset(val_dir,   split='all')
    print(f'Train: {len(train_ds)} scenes  Val: {len(val_ds)} scenes')

    train_loader = DataLoader(train_ds, batch_size=1, shuffle=True,
                              num_workers=2, pin_memory=(device.type == 'cuda'),
                              persistent_workers=True)
    val_loader   = DataLoader(val_ds, batch_size=1, shuffle=False,
                              num_workers=0, pin_memory=False)

    # --use-xyz-offset from CLI takes precedence; otherwise inherit from checkpoint
    use_xyz = args.use_xyz_offset
    if args.resume and not use_xyz:
        _peek = torch.load(args.resume, map_location='cpu', weights_only=False)
        if isinstance(_peek, dict):
            use_xyz = _peek.get('use_xyz_offset', False)

    model = SoftGroupModel(use_xyz_offset=use_xyz).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f'Parameters: {total_params:,}  use_xyz_offset={use_xyz}')
    assert total_params < 50_000_000, f'Too many params: {total_params:,}'

    optimizer = Adam(model.parameters(), lr=args.lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

    start_epoch           = 0
    best_f1               = 0.0
    last_val              = (0.0, 0.0, 0.0)   # (f1@25, f1@50, offset_err)
    last_prefix_f1s       = {}
    total_training_time_s = 0.0               # cumulative across all sessions
    session_start         = time.perf_counter()

    if args.resume:
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        missing, unexpected = model.load_state_dict(ckpt['model_state_dict'], strict=False)
        if missing:
            print(f'  offset_head re-initialised ({len(missing)} keys) — '
                  f'xyz_offset switched: {not use_xyz} → {use_xyz}')
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        if 'scheduler_state_dict' in ckpt:
            scheduler.load_state_dict(ckpt['scheduler_state_dict'])
        start_epoch           = ckpt.get('epoch', 0) + 1
        best_f1               = ckpt.get('val_f1_25', 0.0)
        last_val              = ckpt.get('last_val', (0.0, 0.0, 0.0))
        total_training_time_s = ckpt.get('total_training_time_s', 0.0)
        print(f'Resumed from epoch {start_epoch - 1}  '
              f'best_f1@25={best_f1:.4f}  '
              f'prior training={total_training_time_s/3600:.2f}h')

    for epoch in range(start_epoch, args.epochs):
        model.use_offset_clustering = (epoch >= 3)
        train_loss = train_one_epoch(model, train_loader, optimizer, device, epoch)

        # Validate every 2 epochs and always on the final epoch
        is_eval = (epoch % 2 == 0) or (epoch == args.epochs - 1)
        if is_eval:
            val_f1_25, val_f1_50, mean_off_err, last_prefix_f1s = validate(model, val_loader, device)
            last_val = (val_f1_25, val_f1_50, mean_off_err)
        else:
            val_f1_25, val_f1_50, mean_off_err = last_val

        scheduler.step()

        # ── Timing ──────────────────────────────────────────────────────────
        session_s    = time.perf_counter() - session_start
        cumulative_s = total_training_time_s + session_s

        # ── Console ─────────────────────────────────────────────────────────
        cached_tag = '         ' if is_eval else ' (cached)'
        print(f'Epoch {epoch:3d} | loss={train_loss:.4f} | '
              f'f1@25={val_f1_25:.4f} | f1@50={val_f1_50:.4f} | '
              f'off_err={mean_off_err:.4f}{cached_tag} | '
              f'session={session_s/60:.1f}m total={cumulative_s/3600:.2f}h')
        if is_eval and len(last_prefix_f1s) > 1:
            parts = [f'{p}={v[0]:.3f}/{v[1]:.3f}'
                     for p, v in sorted(last_prefix_f1s.items()) if p != 'other']
            print(f'           per-prefix f1@25/50: {" | ".join(parts)}')

        # ── Checkpoint ──────────────────────────────────────────────────────
        is_best = is_eval and (val_f1_25 > best_f1)
        if is_best:
            best_f1 = val_f1_25

        ckpt_data = {
            'epoch':                 epoch,
            'model_state_dict':      model.state_dict(),
            'optimizer_state_dict':  optimizer.state_dict(),
            'scheduler_state_dict':  scheduler.state_dict(),
            'val_f1_25':             val_f1_25,
            'val_f1_50':             val_f1_50,
            'last_val':              last_val,
            'total_training_time_s': cumulative_s,
            'use_xyz_offset':        use_xyz,
        }
        torch.save(ckpt_data, os.path.join(args.ckpt_dir, 'last_model.pth'))
        if is_best:
            torch.save(ckpt_data, os.path.join(args.ckpt_dir, 'best_model.pth'))
            print(f'  → New best! val_f1@25={best_f1:.4f} at epoch {epoch}')

        # ── Log (every epoch) + curve ────────────────────────────────────────
        log_entry = {
            'epoch':      epoch,
            'train_loss': round(train_loss, 6),
            'val_f1_25':  round(val_f1_25,  4),
            'val_f1_50':  round(val_f1_50,  4),
            'offset_err': round(mean_off_err, 4),
            'lr':         round(scheduler.get_last_lr()[0], 8),
            'is_eval':    is_eval,
            'best':       is_best,
            'session_s':  round(session_s, 1),
            'total_s':    round(cumulative_s, 1),
            'timestamp':  datetime.datetime.now().isoformat(timespec='seconds'),
        }
        if is_eval:
            for pfx, (f25, f50) in last_prefix_f1s.items():
                log_entry[f'f1_25_{pfx}'] = round(f25, 4)
                log_entry[f'f1_50_{pfx}'] = round(f50, 4)
        _append_log(log_path, log_entry)
        _save_curve(log_path, curve_path)


if __name__ == '__main__':
    main()
