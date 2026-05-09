#!/usr/bin/env python3
"""Evaluate semantic head fg_mask precision on the 5 official test cases."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import torch
import torch.nn.functional as F

from model import initialize_model

CKPT = os.path.join(os.path.dirname(__file__), 'checkpoints', 'best_model.pth')
TEST_DIR = os.path.join(os.path.dirname(__file__), '..', 'assets')
FG_THRESHOLDS = [0.2, 0.3, 0.5]


def load_scene(path):
    data = np.load(path, allow_pickle=True).item()
    xyz    = np.asarray(data['xyz'],    dtype=np.float32)
    rgb    = np.asarray(data['rgb'],    dtype=np.float32)
    normal = np.asarray(data['normal'], dtype=np.float32)
    labels = np.asarray(data['instance_labels'], dtype=np.int64)

    if rgb.max() > 1.0:
        rgb = rgb / 255.0

    centroid = xyz.mean(0)
    xyz -= centroid
    radius = np.sqrt((xyz ** 2).sum(1)).max()
    if radius > 1e-8:
        xyz /= radius

    nn = np.linalg.norm(normal, axis=1, keepdims=True)
    normal = np.divide(normal, nn, out=normal, where=nn != 0)

    feat = np.concatenate([xyz, rgb, normal], axis=1)  # (N, 9)
    return feat, labels


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    if not os.path.exists(CKPT):
        print(f'Checkpoint not found: {CKPT}')
        sys.exit(1)

    model = initialize_model(CKPT, device)

    test_files = sorted([
        os.path.join(TEST_DIR, f)
        for f in os.listdir(TEST_DIR)
        if f.startswith('test_case_') and f.endswith('.npy')
    ])
    print(f'Found {len(test_files)} test cases\n')

    totals = {thr: {'tp': 0, 'fp': 0, 'fn': 0} for thr in FG_THRESHOLDS}

    with torch.inference_mode():
        for path in test_files:
            name = os.path.basename(path)
            feat, labels = load_scene(path)
            feat_t = torch.tensor(feat, dtype=torch.float32, device=device)

            outputs = model.forward(feat_t)
            sem_prob = F.softmax(outputs['sem_logits'], dim=1)  # (N, 2)

            gt_fg = labels > 0
            n_gt  = gt_fg.sum()
            print(f'{name}  |  GT fg pts: {n_gt}  /  total: {len(labels)}')

            for thr in FG_THRESHOLDS:
                fg_mask = (sem_prob[:, 1] >= thr).cpu().numpy()
                tp = int((fg_mask &  gt_fg).sum())
                fp = int((fg_mask & ~gt_fg).sum())
                fn = int((~fg_mask & gt_fg).sum())
                prec   = tp / (tp + fp) if (tp + fp) > 0 else float('nan')
                recall = tp / (tp + fn) if (tp + fn) > 0 else float('nan')
                totals[thr]['tp'] += tp
                totals[thr]['fp'] += fp
                totals[thr]['fn'] += fn
                print(f'  thr={thr:.1f}  tp={tp:6d}  fp={fp:6d}  fn={fn:6d}'
                      f'  prec={prec:.4f}  recall={recall:.4f}')
            print()

    print('=== Overall ===')
    for thr in FG_THRESHOLDS:
        tp = totals[thr]['tp']
        fp = totals[thr]['fp']
        fn = totals[thr]['fn']
        prec   = tp / (tp + fp) if (tp + fp) > 0 else float('nan')
        recall = tp / (tp + fn) if (tp + fn) > 0 else float('nan')
        f1     = 2*tp / (2*tp + fp + fn) if (2*tp + fp + fn) > 0 else float('nan')
        print(f'thr={thr:.1f}  prec={prec:.4f}  recall={recall:.4f}  F1={f1:.4f}')


if __name__ == '__main__':
    main()
