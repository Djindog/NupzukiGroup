#!/usr/bin/env python3
"""
Synthetic Nubzuki segmentation dataset generator.

Derived from the README's test-data generation procedure:
  - For each base MultiScan scene (background, all points = label 0):
      • insert 1-5 Nubzuki instances
      • each object: random scale (0.025-0.2 × scene diagonal),
        per-axis anisotropic scaling (0.5–1.5), full random rotation (±180° / axis)
        and placement on an upward-facing surface
      • per-object independent colour jitter
  - Background xyz AND rgb are kept exactly as-is (confirmed by test-case analysis).

Output format (matches test .npy files):
    xyz            float32  (N, 3)
    rgb            uint8    (N, 3)   0–255
    normal         float32  (N, 3)
    instance_labels int32   (N,)    0 = background, 1…K = Nubzuki instances

Usage:
    conda run -n seg python generation.py [options]
"""

import argparse
import glob
import os
from pathlib import Path

import numpy as np
import torch
import trimesh
from tqdm import tqdm

# ── Nubzuki mesh constants (native .glb coordinate space) ─────────────────────
_NUBZUKI_BOUNDS_MIN = np.array([-0.46418998, -0.49921232, -0.40371862], dtype=np.float32)
_NUBZUKI_BOUNDS_MAX = np.array([ 0.4839057,   0.49988693,  0.40684772], dtype=np.float32)
_NUBZUKI_CENTER     = (_NUBZUKI_BOUNDS_MIN + _NUBZUKI_BOUNDS_MAX) * 0.5
_NUBZUKI_DIAG       = float(np.linalg.norm(_NUBZUKI_BOUNDS_MAX - _NUBZUKI_BOUNDS_MIN))



# ── Mesh loading & point sampling ─────────────────────────────────────────────

def load_nubzuki_mesh(glb_path: str) -> trimesh.Trimesh:
    scene = trimesh.load(glb_path)
    return list(scene.geometry.values())[0]


def _bake_vertex_colors(mesh: trimesh.Trimesh) -> np.ndarray:
    """Return (V, 3) uint8 vertex colours, baked from UV texture."""
    return mesh.visual.to_color().vertex_colors[:, :3]   # drop alpha


def sample_nubzuki_points(mesh: trimesh.Trimesh, n_pts: int,
                           vc: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Sample n_pts surface points.

    Returns
    -------
    xyz    (n_pts, 3) float32 – native mesh coordinates
    rgb    (n_pts, 3) float32 – interpolated vertex colours [0,255]
    normal (n_pts, 3) float32 – face normals (unit length)
    """
    pts, face_idx = trimesh.sample.sample_surface(mesh, count=n_pts)

    normals = mesh.face_normals[face_idx].astype(np.float32)

    # Barycentric colour interpolation
    face_verts = mesh.faces[face_idx]                          # (n,3)
    tris       = mesh.vertices[face_verts]                     # (n,3,3)
    bary       = trimesh.triangles.points_to_barycentric(tris, pts)  # (n,3)
    bary       = np.clip(bary, 0.0, 1.0)
    bary      /= bary.sum(axis=1, keepdims=True) + 1e-9

    face_colors = vc[face_verts].astype(np.float32)            # (n,3,3)
    rgb = (bary[:, :, None] * face_colors).sum(axis=1)         # (n,3)
    rgb = np.clip(rgb, 0.0, 255.0).astype(np.float32)

    return pts.astype(np.float32), rgb, normals


# ── MultiScan scene loading ────────────────────────────────────────────────────

def load_multiscan_scene(pth_path: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load one MultiScan .pth file.

    Returns xyz (N,3) float32, rgb (N,3) float32 [0-255], normal (N,3) float32.
    All multiscan points are treated as background (label 0).
    """
    data   = torch.load(pth_path, map_location="cpu", weights_only=False)
    xyz    = np.asarray(data["xyz"],    dtype=np.float32)
    rgb    = np.asarray(data["rgb"],    dtype=np.float32)   # already [0,255]
    normal = np.asarray(data["normal"], dtype=np.float32)

    # Normalise normals (multiscan sometimes has zero-norm entries)
    nlen = np.linalg.norm(normal, axis=1, keepdims=True)
    normal = np.divide(normal, nlen, out=np.zeros_like(normal), where=nlen > 1e-8)

    return xyz, rgb, normal


# ── Surface placement helpers ──────────────────────────────────────────────────

def _build_height_grid(xyz: np.ndarray, normal: np.ndarray,
                        res: float = 0.15, up_thresh: float = 0.7):
    """
    Build a 2-D grid (xi, yi) → max_z of upward-facing surface points.

    Returns (grid, x_min, y_min, res) or None if not enough surface points.
    """
    up_mask = normal[:, 2] > up_thresh
    if up_mask.sum() < 50:
        return None

    sp = xyz[up_mask]
    x_min, y_min = float(sp[:, 0].min()), float(sp[:, 1].min())
    x_max, y_max = float(sp[:, 0].max()), float(sp[:, 1].max())
    nx = max(2, int((x_max - x_min) / res) + 2)
    ny = max(2, int((y_max - y_min) / res) + 2)

    grid = np.full((nx, ny), -1e9, dtype=np.float32)
    ix = np.clip(((sp[:, 0] - x_min) / res).astype(np.int32), 0, nx - 1)
    iy = np.clip(((sp[:, 1] - y_min) / res).astype(np.int32), 0, ny - 1)
    np.maximum.at(grid, (ix, iy), sp[:, 2])

    return grid, x_min, y_min, res


def _find_placement(rng, xyz: np.ndarray, normal: np.ndarray,
                    obj_half_xy: float) -> tuple[float, float, float]:
    """
    Return (cx, cy, surface_z) for placing an object on a surface.
    Falls back to scene floor if no grid entry found.
    """
    result = _build_height_grid(xyz, normal)

    x_lo, x_hi = float(xyz[:, 0].min()), float(xyz[:, 0].max())
    y_lo, y_hi = float(xyz[:, 1].min()), float(xyz[:, 1].max())
    z_floor    = float(np.percentile(xyz[:, 2], 5))

    if result is None:
        cx = rng.uniform(x_lo + obj_half_xy, max(x_lo + obj_half_xy + 0.01, x_hi - obj_half_xy))
        cy = rng.uniform(y_lo + obj_half_xy, max(y_lo + obj_half_xy + 0.01, y_hi - obj_half_xy))
        return cx, cy, z_floor

    grid, x_min, y_min, res = result

    valid_cells = np.argwhere(grid > -1e8)
    if len(valid_cells) == 0:
        cx = rng.uniform(x_lo + obj_half_xy, max(x_lo + obj_half_xy + 0.01, x_hi - obj_half_xy))
        cy = rng.uniform(y_lo + obj_half_xy, max(y_lo + obj_half_xy + 0.01, y_hi - obj_half_xy))
        return cx, cy, z_floor

    pick = rng.integers(0, len(valid_cells))
    gi, gj = valid_cells[pick]

    cx = float(x_min + (gi + rng.uniform(0.1, 0.9)) * res)
    cy = float(y_min + (gj + rng.uniform(0.1, 0.9)) * res)
    cx = float(np.clip(cx, x_lo + obj_half_xy, x_hi - obj_half_xy))
    cy = float(np.clip(cy, y_lo + obj_half_xy, y_hi - obj_half_xy))
    sz = float(grid[gi, gj])

    return cx, cy, sz


# ── Geometric transforms ───────────────────────────────────────────────────────

def _rotation_matrix(rx_deg: float, ry_deg: float, rz_deg: float) -> np.ndarray:
    """ZYX Euler rotation matrix (float64)."""
    rx = np.radians(rx_deg)
    ry = np.radians(ry_deg)
    rz = np.radians(rz_deg)
    cx, sx = np.cos(rx), np.sin(rx)
    cy, sy = np.cos(ry), np.sin(ry)
    cz, sz = np.cos(rz), np.sin(rz)
    Rx = np.array([[1,  0,   0 ], [0,  cx, -sx], [0, sx, cx]], dtype=np.float64)
    Ry = np.array([[cy, 0,  sy ], [0,  1,   0 ], [-sy, 0, cy]], dtype=np.float64)
    Rz = np.array([[cz, -sz, 0], [sz,  cz,  0 ], [0,  0,  1]], dtype=np.float64)
    return Rz @ Ry @ Rx


def _apply_scale_to_normals(normals: np.ndarray, scale_xyz: np.ndarray) -> np.ndarray:
    """Transform normals under anisotropic scaling (inverse-transpose rule), then renorm."""
    n = normals / (scale_xyz + 1e-9)          # inverse-transpose of diag scale
    nlen = np.linalg.norm(n, axis=1, keepdims=True)
    return n / (nlen + 1e-9)


# ── Colour jitter ──────────────────────────────────────────────────────────────

def _color_jitter(rgb: np.ndarray, rng) -> np.ndarray:
    """
    HSV-space colour augmentation.  Applied to one object at a time.

    1. Hue    : uniformly sampled from [0, 1) — covers all perceived colours equally.
    2. Saturation: additive shift in [−0.2, +0.2], clamped to [0, 1].
    3. Value  : additive shift in [−0.2, +0.2], clamped to [0, 1].

    Every point in the instance gets the same target hue (per-instance uniform jitter),
    preserving per-point S/V texture detail while fully randomising perceived colour.
    """
    from matplotlib.colors import rgb_to_hsv, hsv_to_rgb

    rgb_norm = rgb.astype(np.float32) / 255.0          # (N, 3) in [0, 1]
    hsv      = rgb_to_hsv(rgb_norm)                    # (N, 3)  H∈[0,1], S∈[0,1], V∈[0,1]

    hsv[:, 0] = rng.uniform(0.0, 1.0)                  # replace all H with one sampled value
    hsv[:, 1] = np.clip(hsv[:, 1] + rng.uniform(-0.4, 0.4), 0.0, 1.0)
    hsv[:, 2] = np.clip(hsv[:, 2] + rng.uniform(-0.4, 0.4), 0.0, 1.0)

    return np.clip(hsv_to_rgb(hsv) * 255.0, 0.0, 255.0).astype(np.float32)


# ── Single Nubzuki insertion ───────────────────────────────────────────────────

def insert_nubzuki(rng, mesh: trimesh.Trimesh, vc: np.ndarray,
                   scene_xyz: np.ndarray, scene_normal: np.ndarray,
                   scene_diagonal: float, bg_n_pts: int,
                   scale_range: tuple = (0.015, 0.2),
                   aniso_range: tuple  = (0.5,   1.5)
                   ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Place one Nubzuki in the scene.

    Returns obj_xyz (M,3), obj_rgb (M,3) float32 [0-255], obj_normal (M,3).
    """
    # 1. Scale ratio → base scale factor relative to native mesh size
    scale_ratio = float(rng.uniform(*scale_range))
    base_scale  = (scale_ratio * scene_diagonal) / _NUBZUKI_DIAG

    # 2. Number of points – uniform sample in [1500, 40000]
    n_pts = int(rng.integers(1500, 40001))

    # 3. Sample mesh surface points
    obj_xyz, obj_rgb, obj_normal = sample_nubzuki_points(mesh, n_pts, vc)

    # 4. Centre the object at origin (native coords)
    obj_xyz -= _NUBZUKI_CENTER

    # 5. Per-axis anisotropic scale (applied BEFORE rotation, matches README)
    aniso       = rng.uniform(aniso_range[0], aniso_range[1], size=3).astype(np.float32)
    full_scale  = (base_scale * aniso).astype(np.float32)
    obj_xyz    *= full_scale
    obj_normal  = _apply_scale_to_normals(obj_normal, full_scale)

    # 6. Random rotation ±180° per axis
    rx = float(rng.uniform(-180, 180))
    ry = float(rng.uniform(-180, 180))
    rz = float(rng.uniform(-180, 180))
    R  = _rotation_matrix(rx, ry, rz).astype(np.float32)

    obj_xyz    = (R @ obj_xyz.T).T
    obj_normal = (R @ obj_normal.T).T

    # 7. Placement: surface (85%) or floating in mid-air (15%)
    if rng.uniform() < 0.15:
        # Floating: centre the object at a random position inside the scene volume.
        # z is sampled above the floor with enough clearance for the object's half-height.
        x_lo, x_hi = float(scene_xyz[:, 0].min()), float(scene_xyz[:, 0].max())
        y_lo, y_hi = float(scene_xyz[:, 1].min()), float(scene_xyz[:, 1].max())
        z_floor    = float(np.percentile(scene_xyz[:, 2], 5))
        z_ceil     = float(np.percentile(scene_xyz[:, 2], 95))
        obj_half_z = float((obj_xyz[:, 2].max() - obj_xyz[:, 2].min()) * 0.5)
        z_lo = z_floor + obj_half_z + 0.10
        z_hi = max(z_lo + 0.01, z_ceil - obj_half_z)
        cx  = float(rng.uniform(x_lo, x_hi))
        cy  = float(rng.uniform(y_lo, y_hi))
        cz  = float(rng.uniform(z_lo, z_hi))
        obj_xyz[:, 0] += cx
        obj_xyz[:, 1] += cy
        obj_xyz[:, 2] += cz
    else:
        # Surface placement: bottom of object → local surface height
        obj_bottom_z = float(obj_xyz[:, 2].min())
        obj_half_xy  = float(max(
            (obj_xyz[:, 0].max() - obj_xyz[:, 0].min()),
            (obj_xyz[:, 1].max() - obj_xyz[:, 1].min())
        ) * 0.5)
        cx, cy, surface_z = _find_placement(rng, scene_xyz, scene_normal, obj_half_xy)
        obj_xyz[:, 0] += cx
        obj_xyz[:, 1] += cy
        obj_xyz[:, 2] += (surface_z - obj_bottom_z)

    # 8. Per-object colour jitter
    obj_rgb = _color_jitter(obj_rgb, rng)

    return obj_xyz, obj_rgb, obj_normal


# ── Full scene generation ──────────────────────────────────────────────────────

def generate_scene(rng,
                   scene_xyz: np.ndarray,
                   scene_rgb: np.ndarray,
                   scene_normal: np.ndarray,
                   mesh: trimesh.Trimesh,
                   vc: np.ndarray,
                   min_objects: int = 1,
                   max_objects: int = 5
                   ) -> dict:
    """
    Insert 1–5 Nubzukis into a background scene.

    Background xyz & rgb are kept unmodified (confirmed by test-case analysis).

    Returns a dict with keys: xyz, rgb (uint8), normal, instance_labels (int32).
    """
    n_objects      = int(rng.integers(min_objects, max_objects + 1))
    scene_diagonal = float(np.linalg.norm(scene_xyz.max(0) - scene_xyz.min(0)))

    all_xyz    = [scene_xyz]
    all_rgb    = [scene_rgb]
    all_normal = [scene_normal]
    all_labels = [np.zeros(len(scene_xyz), dtype=np.int32)]

    for inst_id in range(1, n_objects + 1):
        # Stack previously placed objects so new ones can be placed on top
        current_xyz    = np.concatenate(all_xyz,    axis=0)
        current_normal = np.concatenate(all_normal, axis=0)

        obj_xyz, obj_rgb, obj_normal = insert_nubzuki(
            rng, mesh, vc, current_xyz, current_normal, scene_diagonal, len(scene_xyz)
        )

        all_xyz.append(obj_xyz)
        all_rgb.append(obj_rgb)
        all_normal.append(obj_normal)
        all_labels.append(np.full(len(obj_xyz), inst_id, dtype=np.int32))

    xyz    = np.concatenate(all_xyz,    axis=0).astype(np.float32)
    rgb    = np.clip(np.concatenate(all_rgb,    axis=0), 0, 255).astype(np.uint8)
    normal = np.concatenate(all_normal, axis=0).astype(np.float32)
    labels = np.concatenate(all_labels, axis=0).astype(np.int32)

    return {"xyz": xyz, "rgb": rgb, "normal": normal, "instance_labels": labels}


# ── Entry point ────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Generate Nubzuki segmentation training data")
    p.add_argument("--multiscan-dir", default=
        "../assets/multiscan",
        help="MultiScan directory containing train/val/test subdirs of .pth files")
    p.add_argument("--glb-path", default=
        "../assets/sample.glb",
        help="Path to Nubzuki .glb mesh")
    p.add_argument("--output-dir", default="generated_data",
        help="Root output directory (train/, val/, test/ created inside)")
    p.add_argument("--n-train", type=int, default=6,
        help="Generated samples per MultiScan train scene")
    p.add_argument("--n-val",   type=int, default=2,
        help="Generated samples per MultiScan val scene")
    p.add_argument("--n-test",  type=int, default=2,
        help="Generated samples per MultiScan test scene")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    rng  = np.random.default_rng(args.seed)

    # Resolve paths relative to this script's directory
    script_dir   = Path(__file__).parent.resolve()
    multiscan_dir = Path(args.multiscan_dir) if Path(args.multiscan_dir).is_absolute() \
                    else script_dir / args.multiscan_dir
    glb_path     = Path(args.glb_path) if Path(args.glb_path).is_absolute() \
                    else script_dir / args.glb_path
    output_dir   = Path(args.output_dir) if Path(args.output_dir).is_absolute() \
                    else script_dir / args.output_dir

    print(f"Loading Nubzuki mesh: {glb_path}")
    mesh = load_nubzuki_mesh(str(glb_path))
    vc   = _bake_vertex_colors(mesh)
    print(f"  vertices={len(mesh.vertices)}, faces={len(mesh.faces)}, "
          f"vc_mean={vc.mean(0).round(1)}")

    split_aug = {"train": args.n_train, "val": args.n_val, "test": args.n_test}

    for split, n_aug in split_aug.items():
        in_dir  = multiscan_dir / split
        out_dir = output_dir / split
        out_dir.mkdir(parents=True, exist_ok=True)

        pth_files = sorted(glob.glob(str(in_dir / "*.pth")))
        if not pth_files:
            print(f"[{split}] No .pth files found in {in_dir}, skipping.")
            continue

        total = len(pth_files) * n_aug
        print(f"\n[{split}] {len(pth_files)} scenes × {n_aug} aug = {total} samples → {out_dir}")

        count = 0
        for pth_file in tqdm(pth_files, desc=split):
            scene_name = Path(pth_file).stem
            scene_xyz, scene_rgb, scene_normal = load_multiscan_scene(pth_file)

            diag = float(np.linalg.norm(scene_xyz.max(0) - scene_xyz.min(0)))

            for aug_i in range(n_aug):
                sample = generate_scene(
                    rng, scene_xyz, scene_rgb, scene_normal, mesh, vc
                )

                out_name = f"{scene_name}_aug{aug_i:02d}.npy"
                np.save(str(out_dir / out_name), sample)
                count += 1

        print(f"  Saved {count} files to {out_dir}")

    print("\nDone.")


if __name__ == "__main__":
    main()
