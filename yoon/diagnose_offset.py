#!/usr/bin/env python3
"""
Diagnose offset head predictions for test_case_004.
Produces 4-panel plots (top + front views) showing:
  - GT instance labels on original xyz
  - GT offset vectors (where points SHOULD shift)
  - Predicted offset vectors (where they actually shift)
  - Shifted xyz colored by GT label (what DBSCAN sees)
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from model import initialize_model

# ── Config ──────────────────────────────────────────────────────────────────
CKPT   = os.path.join(os.path.dirname(__file__), 'checkpoints', 'best_model.pth')
DATA   = os.path.join(os.path.dirname(__file__), '..', 'assets', 'test_case_004.npy')
OUT    = os.path.join(os.path.dirname(__file__), 'diagnosis', 'offset_debug')
N_ARROW = 400   # quiver subsampling per instance
SEED   = 0
# ────────────────────────────────────────────────────────────────────────────

os.makedirs(OUT, exist_ok=True)

# ── Load & normalise (mirrors dataset.py) ───────────────────────────────────
raw = np.load(DATA, allow_pickle=True).item()
xyz_raw = np.asarray(raw['xyz'],            dtype=np.float32)
rgb     = np.asarray(raw['rgb'],            dtype=np.float32)
normal  = np.asarray(raw['normal'],         dtype=np.float32)
gt_inst = np.asarray(raw['instance_labels'],dtype=np.int64)

if rgb.max() > 1.0:
    rgb = rgb / 255.0

centroid = xyz_raw.mean(0)
xyz = xyz_raw - centroid
radius = np.sqrt((xyz**2).sum(1)).max()
if radius > 1e-8:
    xyz = xyz / radius

nn = np.linalg.norm(normal, axis=1, keepdims=True)
normal = np.divide(normal, nn, out=normal, where=nn > 0)

feat = np.concatenate([xyz, rgb, normal], axis=1).T   # (9, N)
feat_t = torch.tensor(feat, dtype=torch.float32).unsqueeze(0).cuda()  # (1,9,N)

# ── Run model ───────────────────────────────────────────────────────────────
model = initialize_model(CKPT, device=torch.device('cuda'))
model.eval()
with torch.no_grad():
    feat_N9 = feat_t[0].T.contiguous()   # (N,9)
    out = model.forward(feat_N9)
    sem_logits  = out['sem_logits'].cpu().numpy()   # (N,2)
    offset_pred = out['offset_pred'].cpu().numpy()  # (N,3)

sem_prob = torch.softmax(torch.tensor(sem_logits), dim=1).numpy()
fg_mask  = sem_prob[:, 1] >= 0.2
fg_idx   = np.where(fg_mask)[0]

# GT offsets (center - point for each fg point)
gt_inst_t = torch.tensor(gt_inst)
xyz_t     = torch.tensor(xyz)
offset_gt = np.zeros_like(xyz)
for iid in np.unique(gt_inst):
    if iid <= 0:
        continue
    m    = gt_inst == iid
    ctr  = xyz[m].mean(0)
    offset_gt[m] = ctr - xyz[m]

shifted_pred = xyz + offset_pred   # all points; meaningful only for fg

# ── Colour helpers ───────────────────────────────────────────────────────────
PALETTE = {
    0: (0.75, 0.75, 0.75),
    1: (0.90, 0.20, 0.20),
    2: (0.20, 0.70, 0.20),
    3: (0.20, 0.40, 0.90),
    4: (0.90, 0.60, 0.10),
    5: (0.70, 0.10, 0.80),
}

def inst_colors(labels, alpha_bg=0.05):
    cols = np.zeros((len(labels), 4), dtype=np.float32)
    for iid, c in PALETTE.items():
        m = labels == iid
        cols[m, :3] = c
        cols[m,  3] = alpha_bg if iid == 0 else 1.0
    return cols

def make_legend(ax, ids):
    handles = []
    for iid in sorted(ids):
        if iid <= 0:
            continue
        c = PALETTE.get(iid, (0.5, 0.5, 0.5))
        handles.append(mpatches.Patch(color=c, label=f'inst {iid}'))
    ax.legend(handles=handles, fontsize=7, loc='upper right')


# ── Plot factory ─────────────────────────────────────────────────────────────
def scatter2d(ax, pts, colors, title, s=1.0):
    ax.scatter(pts[:, 0], pts[:, 1], c=colors, s=s, linewidths=0)
    ax.set_title(title, fontsize=9)
    ax.set_aspect('equal')
    ax.set_xticks([]); ax.set_yticks([])

def quiver2d(ax, pts_from, pts_to, labels, title, max_per_inst=N_ARROW):
    """Arrow from original to shifted position, coloured by GT label."""
    rng = np.random.default_rng(SEED)
    for iid, c in PALETTE.items():
        if iid == 0:
            continue
        m = np.where(labels == iid)[0]
        if len(m) == 0:
            continue
        sub = rng.choice(m, min(max_per_inst, len(m)), replace=False)
        dx  = pts_to[sub, 0] - pts_from[sub, 0]
        dy  = pts_to[sub, 1] - pts_from[sub, 1]
        ax.quiver(pts_from[sub, 0], pts_from[sub, 1], dx, dy,
                  color=c, angles='xy', scale_units='xy', scale=1,
                  width=0.003, headwidth=4, headlength=4, alpha=0.6)
    ax.set_title(title, fontsize=9)
    ax.set_aspect('equal')
    ax.set_xticks([]); ax.set_yticks([])


# ── Per-view figures ─────────────────────────────────────────────────────────
VIEWS = {
    'top':   (0, 1),   # X-Y
    'front': (0, 2),   # X-Z
    'side':  (1, 2),   # Y-Z
}

for vname, (ax_h, ax_v) in VIEWS.items():
    fig, axes = plt.subplots(2, 4, figsize=(22, 11))
    fig.suptitle(f'test_case_004 – offset debug ({vname} view)', fontsize=12)

    # --- ROW 0: GT analysis ---

    # (0,0) Raw RGB
    scatter2d(axes[0, 0], xyz[:, [ax_h, ax_v]], rgb, 'Input RGB')

    # (0,1) GT instance labels
    scatter2d(axes[0, 1], xyz[:, [ax_h, ax_v]], inst_colors(gt_inst), 'GT instances')
    make_legend(axes[0, 1], np.unique(gt_inst))

    # (0,2) GT offset arrows (what the model SHOULD learn)
    axes[0, 2].scatter(xyz[:, ax_h], xyz[:, ax_v],
                       c=inst_colors(gt_inst), s=0.5, linewidths=0)
    quiver2d(axes[0, 2],
             xyz[:, [ax_h, ax_v]],
             (xyz + offset_gt)[:, [ax_h, ax_v]],
             gt_inst, 'GT offset vectors')

    # (0,3) Shifted GT (xyz + offset_gt) – should form tight blobs
    shifted_gt = xyz + offset_gt
    scatter2d(axes[0, 3], shifted_gt[:, [ax_h, ax_v]],
              inst_colors(gt_inst, alpha_bg=0.05),
              'Shifted by GT offset (fg only)')
    make_legend(axes[0, 3], np.unique(gt_inst))

    # --- ROW 1: Predicted analysis ---

    # (1,0) Foreground probability
    fg_prob_vals = sem_prob[:, 1]
    sc = axes[1, 0].scatter(xyz[:, ax_h], xyz[:, ax_v],
                            c=fg_prob_vals, cmap='hot', s=0.8, vmin=0, vmax=1)
    plt.colorbar(sc, ax=axes[1, 0], fraction=0.046, pad=0.04)
    axes[1, 0].set_title('FG probability', fontsize=9)
    axes[1, 0].set_aspect('equal')
    axes[1, 0].set_xticks([]); axes[1, 0].set_yticks([])

    # (1,1) Predicted offset arrows (what the model actually does)
    axes[1, 1].scatter(xyz[:, ax_h], xyz[:, ax_v],
                       c=inst_colors(gt_inst), s=0.5, linewidths=0)
    quiver2d(axes[1, 1],
             xyz[:, [ax_h, ax_v]],
             shifted_pred[:, [ax_h, ax_v]],
             gt_inst, 'Pred offset vectors (coloured by GT label)')

    # (1,2) Shifted by pred offset, coloured by GT label (what DBSCAN sees)
    scatter2d(axes[1, 2], shifted_pred[fg_mask][:, [ax_h, ax_v]],
              inst_colors(gt_inst[fg_mask]),
              'Shifted by PRED offset\n(fg pts, GT colour)')
    make_legend(axes[1, 2], np.unique(gt_inst[fg_mask]))

    # (1,3) Offset error magnitude heatmap (fg only)
    err = np.linalg.norm(offset_pred - offset_gt, axis=1)
    fg_err = err[fg_mask]
    sc2 = axes[1, 3].scatter(xyz[fg_mask, ax_h], xyz[fg_mask, ax_v],
                              c=fg_err, cmap='RdYlGn_r', s=1.0,
                              vmin=0, vmax=np.percentile(fg_err, 95))
    plt.colorbar(sc2, ax=axes[1, 3], fraction=0.046, pad=0.04)
    axes[1, 3].set_title(f'Offset error (fg pts)\nmedian={np.median(fg_err):.3f}', fontsize=9)
    axes[1, 3].set_aspect('equal')
    axes[1, 3].set_xticks([]); axes[1, 3].set_yticks([])

    plt.tight_layout()
    out_path = os.path.join(OUT, f'offset_debug_{vname}.png')
    plt.savefig(out_path, dpi=130, bbox_inches='tight')
    plt.close()
    print(f'Saved: {out_path}')

# ── Per-instance stats ───────────────────────────────────────────────────────
print('\n── Per-instance offset stats ──')
for iid in sorted(np.unique(gt_inst)):
    if iid <= 0:
        continue
    m      = gt_inst == iid
    ctr_gt = xyz[m].mean(0)
    pred_c = (xyz[m] + offset_pred[m]).mean(0)   # predicted centroid
    err_m  = np.linalg.norm(offset_pred[m] - offset_gt[m], axis=1)
    print(f'  inst {iid}: n_pts={m.sum():5d}  '
          f'GT_ctr=({ctr_gt[0]:+.3f},{ctr_gt[1]:+.3f},{ctr_gt[2]:+.3f})  '
          f'pred_ctr=({pred_c[0]:+.3f},{pred_c[1]:+.3f},{pred_c[2]:+.3f})  '
          f'err median={np.median(err_m):.3f} mean={err_m.mean():.3f}')

# Also show centroid separation
print('\n── Predicted centroid distances (should be > DBSCAN eps=0.02) ──')
insts = [i for i in sorted(np.unique(gt_inst)) if i > 0]
pred_ctrs = {}
for iid in insts:
    m = gt_inst == iid
    pred_ctrs[iid] = (xyz[m] + offset_pred[m]).mean(0)
for i in range(len(insts)):
    for j in range(i+1, len(insts)):
        a, b = insts[i], insts[j]
        d = np.linalg.norm(pred_ctrs[a] - pred_ctrs[b])
        print(f'  inst {a} ↔ inst {b}: pred_centroid_dist = {d:.4f}')
