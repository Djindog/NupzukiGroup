#!/usr/bin/env python3
"""
test_all.py — evaluate a checkpoint on all four test datasets and print a table.

Datasets tested:
  official  : assets/          (course-provided test_case_*.npy)
  yoon      : yoon/generated_data/test/
  cjm       : yoon/generated_data_cjm/test/
  seyoon    : yoon/generated_data_seyoon/test/

Usage (from CS479-Seg root):
    conda run -n 3d-seg python yoon/test_all.py
    conda run -n 3d-seg python yoon/test_all.py --ckpt-path yoon/checkpoints/best_model.pth
"""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent.resolve()
ROOT       = SCRIPT_DIR.parent

DATASETS = [
    ('official', "/home/ubuntu/CS479-Seg/assets"),
    ('yoon',     str(SCRIPT_DIR / 'generated_data_yoon' / 'test')),
    ('cjm',      str(SCRIPT_DIR / 'generated_data_cjm' / 'test')),
    ('seyoon',   str(SCRIPT_DIR / 'generated_data_seyoon' / 'test')),
]


def run_eval(test_data_dir: str, ckpt_path: str, output_dir: str):
    """
    Run evaluate.py for one dataset.
    Returns (f1_25, f1_50, n_scenes) on success, or (None, None, error_msg) on failure.
    """
    proc = subprocess.run(
        [sys.executable, str(ROOT / 'evaluate.py'),
         '--test-data-dir', test_data_dir,
         '--ckpt-path',     ckpt_path,
         '--output-dir',    output_dir],
        capture_output=True, text=True,
    )

    if proc.returncode != 0:
        # Surface the first meaningful error line
        err_lines = [l for l in proc.stderr.splitlines() if l.strip()]
        return None, None, (err_lines[-1] if err_lines else 'unknown error')

    # Primary source: metrics.json written by evaluate.py
    metrics_path = os.path.join(output_dir, 'metrics.json')
    if os.path.exists(metrics_path):
        with open(metrics_path) as f:
            m = json.load(f)
        return m['instance_f1_25'], m['instance_f1_50'], m['num_scenes']

    # Fallback: parse the stdout line "Instance F1 -> @25: X, @50: Y"
    for line in proc.stdout.splitlines():
        if 'Instance F1' in line:
            try:
                parts = line.split()
                f25 = float(parts[parts.index('@25:') + 1].rstrip(','))
                f50 = float(parts[parts.index('@50:') + 1])
                return f25, f50, -1
            except (ValueError, IndexError):
                pass

    return None, None, 'could not parse output'


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt-path', default='yoon/checkpoints/best_model.pth',
                   help='Checkpoint path (relative to CS479-Seg root or absolute)')
    args = p.parse_args()

    ckpt_path = args.ckpt_path
    if not os.path.isabs(ckpt_path):
        ckpt_path = str(ROOT / ckpt_path)
    if not os.path.exists(ckpt_path):
        print(f'ERROR: checkpoint not found: {ckpt_path}')
        sys.exit(1)

    print(f'Checkpoint : {ckpt_path}')
    print()

    rows = []
    for name, data_dir in DATASETS:
        n_files = len(list(Path(data_dir).glob('*.npy'))) if Path(data_dir).is_dir() else 0

        if n_files == 0:
            reason = 'dir not found' if not Path(data_dir).is_dir() else 'no .npy files'
            print(f'  [{name:8s}]  SKIP — {reason}: {data_dir}')
            rows.append((name, None, None, None))
            continue

        out_dir = str(SCRIPT_DIR / 'eval_out' / f'test_all_{name}')
        print(f'  [{name:8s}]  {n_files} scenes  →  evaluating ...', flush=True)

        f25, f50, n_or_err = run_eval(data_dir, ckpt_path, out_dir)

        if f25 is None:
            print(f'  [{name:8s}]  ERROR: {n_or_err}')
            rows.append((name, None, None, None))
        else:
            n = n_or_err
            print(f'  [{name:8s}]  F1@25={f25:.4f}  F1@50={f50:.4f}  ({n} scenes)')
            rows.append((name, f25, f50, n))

    # ── Summary table ─────────────────────────────────────────────────────────
    print()
    w = 52
    print('─' * w)
    print(f'  {"Dataset":10s}  {"F1@25":>8s}  {"F1@50":>8s}  {"Scenes":>7s}')
    print('─' * w)
    for name, f25, f50, n in rows:
        if f25 is None:
            print(f'  {name:10s}  {"—":>8s}  {"—":>8s}  {"—":>7s}')
        else:
            print(f'  {name:10s}  {f25:8.4f}  {f50:8.4f}  {n:>7d}')
    print('─' * w)


if __name__ == '__main__':
    main()
