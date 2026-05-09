#!/usr/bin/env python3
"""
generate_mix.py — orchestrates yoon / cjm / seyoon generators and merges
their outputs into yoon/generated_data_mix/{train,val,test}/.

Target breakdown (multiscan has ~174 train / 42 val / 41 test scenes):
  train : 174 × N_TRAIN_COPIES × 3 generators  ≈ 1044 files (N=2)
  val   : 42  × 1 copy × yoon only              ≈  42  files
  test  : 41  × 1 copy × yoon only              ≈  41  files

The val/test splits come from yoon only so they remain clean and
deterministic (CJM + Seyoon training diversity is train-side only).

Usage:
    conda run -n 3d-seg python yoon/generate_mix.py
    conda run -n 3d-seg python yoon/generate_mix.py --skip-cjm --skip-seyoon
    conda run -n 3d-seg python yoon/generate_mix.py --n-copies 3  # more data
"""
import argparse
import importlib
import os
import random
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).parent.resolve()
ROOT       = SCRIPT_DIR.parent
GLB        = str(ROOT / 'assets' / 'sample.glb')
MULTISCAN  = str(ROOT / 'assets' / 'multiscan')
MIX_OUT    = SCRIPT_DIR / 'generated_data_mix'
SEED       = 42


def _ensure(d: Path):
    d.mkdir(parents=True, exist_ok=True)


# ── YOON generator (clean CLI — run as subprocess) ────────────────────────────

def run_yoon(n_train_copies: int):
    print(f'[yoon] {len(sorted((Path(MULTISCAN)/"train").glob("*.pth")))} train scenes '
          f'× {n_train_copies} copies → train', flush=True)
    tmp = SCRIPT_DIR / '_tmp_yoon'
    subprocess.run(
        [sys.executable, str(SCRIPT_DIR / 'generation.py'),
         '--multiscan-dir', MULTISCAN,
         '--glb-path', GLB,
         '--output-dir', str(tmp),
         '--n-train', str(n_train_copies),
         '--n-val',   '1',
         '--n-test',  '1',
         '--seed',    str(SEED)],
        check=True,
    )
    # Move outputs with 'yoon_' prefix to avoid name collisions across generators
    for split in ('train', 'val', 'test'):
        src = tmp / split
        if not src.exists():
            continue
        dst = MIX_OUT / split
        _ensure(dst)
        moved = 0
        for f in sorted(src.glob('*.npy')):
            target = dst / f'yoon_{f.name}'
            if not target.exists():
                shutil.move(str(f), str(target))
                moved += 1
        print(f'  [yoon][{split}] moved {moved} files', flush=True)
    shutil.rmtree(tmp, ignore_errors=True)


# ── CJM generator (import directly, custom train loop) ────────────────────────

def run_cjm(n_train_copies: int):
    sys.path.insert(0, str(SCRIPT_DIR))
    cjm = importlib.import_module('generation_cjm')

    np.random.seed(SEED + 100)
    random.seed(SEED + 100)

    print('[cjm] Loading raw mesh ...', flush=True)
    obj_raw = cjm.load_nubjuki_model_raw(GLB)

    ok = skipped = 0
    for split, n_copies, ms_split in [('train', n_train_copies, 'train'), ('val', 1, 'val')]:
        ms_dir  = Path(MULTISCAN) / ms_split
        out_dir = MIX_OUT / split
        _ensure(out_dir)
        pth_files = sorted(ms_dir.glob('*.pth'))
        print(f'[cjm][{split}] {len(pth_files)} scenes × {n_copies} copies', flush=True)
        for pth in pth_files:
            for copy_idx in range(n_copies):
                use_aug  = (copy_idx > 0)
                hard_neg = np.random.rand() < cjm.HARD_NEGATIVE_RATE
                out_path = out_dir / f'cjm_{pth.stem}_aug{copy_idx:02d}.npy'
                if out_path.exists():
                    skipped += 1
                    continue
                print(f'  [cjm][{split}] {pth.name} copy={copy_idx} aug={use_aug} hard={hard_neg}', flush=True)
                success = False
                while not success:
                    success = cjm.synthesize_scene(
                        str(pth), obj_raw, str(out_path),
                        split=split, use_scene_aug=use_aug, hard_negative=hard_neg,
                    )
                ok += 1
    print(f'[cjm] done — generated={ok} skipped={skipped}', flush=True)


# ── Seyoon generator (import directly, custom train loop) ─────────────────────

def run_seyoon(n_train_copies: int):
    sys.path.insert(0, str(SCRIPT_DIR))
    seyoon = importlib.import_module('generation_seyoon')

    np.random.seed(SEED + 200)
    random.seed(SEED + 200)

    print('[seyoon] Loading mesh ...', flush=True)
    mesh = seyoon.load_nubjuki_mesh(GLB)

    # Cycle through keep_ratios so each copy uses a different density
    keep_ratios = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5]

    ok = skipped = 0
    for split, n_copies, ms_split in [('train', n_train_copies, 'train'), ('val', 1, 'val')]:
        ms_dir  = Path(MULTISCAN) / ms_split
        out_dir = MIX_OUT / split
        _ensure(out_dir)
        pth_files = sorted(ms_dir.glob('*.pth'))
        print(f'[seyoon][{split}] {len(pth_files)} scenes × {n_copies} copies', flush=True)
        for pth in pth_files:
            for copy_idx in range(n_copies):
                kr        = keep_ratios[copy_idx % len(keep_ratios)]
                scene_aug = (copy_idx > 0)
                out_path  = out_dir / f'seyoon_{pth.stem}_c{copy_idx:02d}_kr{int(kr * 100):03d}.npy'
                if out_path.exists():
                    skipped += 1
                    continue
                print(f'  [seyoon][{split}] {pth.name} copy={copy_idx} kr={kr} aug={scene_aug}', flush=True)
                seyoon.synthesize_scene(str(pth), mesh, str(out_path),
                                        keep_ratio=kr, scene_aug=scene_aug)
                ok += 1
    print(f'[seyoon] done — generated={ok} skipped={skipped}', flush=True)


# ── Summary ───────────────────────────────────────────────────────────────────

def print_summary():
    print('\n=== generated_data_mix summary ===')
    total = 0
    for split in ('train', 'val', 'test'):
        d = MIX_OUT / split
        n = len(list(d.glob('*.npy'))) if d.exists() else 0
        total += n
        print(f'  {split:5}: {n:4d} files')
        if n > 0 and d.exists():
            for prefix in ('yoon_', 'cjm_', 'seyoon_'):
                k = len([f for f in d.glob(f'{prefix}*.npy')])
                if k:
                    print(f'           {prefix[:-1]:7}: {k}')
    print(f'  total  : {total}')
    print()
    print('Train command:')
    print('  conda run -n 3d-seg python yoon/train.py \\')
    print(f'    --data-dir yoon/generated_data_mix/train \\')
    print('    --ckpt-dir yoon/checkpoints --epochs 30 --lr 1e-3')
    print()
    print('Or fine-tune from existing checkpoint:')
    print('  conda run -n 3d-seg python yoon/train.py \\')
    print(f'    --data-dir yoon/generated_data_mix/train \\')
    print('    --ckpt-dir yoon/checkpoints --epochs 15 --lr 1e-4 \\')
    print('    --resume yoon/checkpoints/best_model.pth')


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description='Mixed dataset generator')
    p.add_argument('--skip-yoon',   action='store_true', help='Skip yoon generator')
    p.add_argument('--skip-cjm',    action='store_true', help='Skip CJM generator')
    p.add_argument('--skip-seyoon', action='store_true', help='Skip Seyoon generator')
    p.add_argument('--n-copies', type=int, default=2,
                   help='Copies per multiscan-train scene per generator (default: 2 → ~1044 train)')
    args = p.parse_args()

    _ensure(MIX_OUT)

    if not args.skip_yoon:
        print('=== YOON generator ===', flush=True)
        run_yoon(args.n_copies)
        print()

    if not args.skip_cjm:
        print('=== CJM generator ===', flush=True)
        run_cjm(args.n_copies)
        print()

    if not args.skip_seyoon:
        print('=== SEYOON generator ===', flush=True)
        run_seyoon(args.n_copies)
        print()

    print_summary()


if __name__ == '__main__':
    main()
