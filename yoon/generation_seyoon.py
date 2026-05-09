"""generate_data6.py — generate_data4 + generate_data5 통합 generator.

generate_data4 의 train (random keep_ratio 0.95-1.0 + scene aug) 와
generate_data5 의 val/test (fixed keep_ratio cycle) 를 하나의 generator 에서 처리.

Split 별 정책 (SPLIT_CONFIG 로 명시):
  - train : 12 copies × multiscan/train, keep_ratios [1.0, 0.9, 0.8, 0.7, 0.6, 0.5] × 2 cycle
            scene aug 적용 — 학습 데이터에 fg/bg 비율 다양 분포 + 풍부한 augmentation
  - val   : 6 copies × multiscan/val, keep_ratios [1.0, 0.9, 0.8, 0.7, 0.6, 0.5]
            scene aug 없음 — 학습 monitor 시 결정적 평가
  - test  : 1 copy × multiscan/test, keep_ratio = 1.0 만
            hidden test hold-out (deterministic, 분포 중성)

출력 파일명:
  - 모든 split 공통: scene_XXXXX_YY_c{NN}_kr{NNN}.npy
    (NN = copy_idx, NNN = keep_ratio × 100, copy_idx 포함으로 중복 keep_ratio 충돌 방지)

데이터 크기 (예상):
  - train: 174 × 12 = 2,088 files (~27 GB)
  - val  : 42  × 6  =   252 files (~3 GB)
  - test : 41  × 1  =    41 files (~0.5 GB)
  - 총 ~30 GB. 디스크 여유 확인 필수.

Resume safe: 이미 존재하는 파일은 skip.
"""

import os
import random
import time
import multiprocessing as mp

import numpy as np
import torch
import trimesh
from scipy.spatial.transform import Rotation as R
from matplotlib.colors import rgb_to_hsv, hsv_to_rgb


# ─────────────────────────────────────────────
#  Mesh 로딩 / sampling (generate_data4 와 동일)
# ─────────────────────────────────────────────

def load_nubjuki_mesh(glb_path):
    mesh = trimesh.load(glb_path, force='mesh')
    if isinstance(mesh.visual, trimesh.visual.TextureVisuals):
        mesh.visual = mesh.visual.to_color()
    return mesh


def sample_nubjuki(mesh, num_points):
    points, face_indices = trimesh.sample.sample_surface(mesh, num_points)
    normals = mesh.face_normals[face_indices]
    colors = mesh.visual.face_colors[face_indices][:, :3].astype(np.uint8)
    return np.asarray(points, dtype=np.float32), colors, np.asarray(normals, dtype=np.float32)


# ─────────────────────────────────────────────
#  Object-level / Scene-level augmentations (generate_data4 와 동일)
# ─────────────────────────────────────────────

def apply_augmentations(points, colors, normals):
    scale_factors = np.random.uniform(0.5, 1.5, size=3)
    points = points * scale_factors
    normals = normals / (scale_factors + 1e-8)
    normals /= (np.linalg.norm(normals, axis=1, keepdims=True) + 1e-8)

    rot = R.from_euler('xyz', np.random.uniform(-180, 180, size=3), degrees=True)
    points = rot.apply(points)
    normals = rot.apply(normals)

    colors_f = colors.astype(np.float64) / 255.0
    hsv_colors = rgb_to_hsv(colors_f.reshape(1, -1, 3))
    hsv_colors[0, :, 0] = np.random.uniform(0.0, 1.0)
    hsv_colors[0, :, 1] = np.random.uniform(0.7, 1.0)
    new_rgb = hsv_to_rgb(hsv_colors).reshape(-1, 3)
    colors = (new_rgb * 255).astype(np.uint8)
    return points, colors, normals


def augment_object_extra(points, colors, normals):
    points = points + np.random.normal(0, 0.002, size=points.shape)
    colors_f = colors.astype(np.float64) / 255.0
    hsv = rgb_to_hsv(colors_f.reshape(1, -1, 3))
    hsv[0, :, 2] = np.clip(hsv[0, :, 2] + np.random.uniform(-0.15, 0.15), 0.0, 1.0)
    colors = (hsv_to_rgb(hsv).reshape(-1, 3) * 255).astype(np.uint8)
    return points, colors, normals


def scene_aug_jitter_and_brightness(xyz, rgb, normal):
    """generate_data4 의 augment_scene_pointcloud 에서 keep_ratio 부분 분리한 버전.

    keep_ratio 는 synthesize_scene 안에서 별도로 처리 (config 에 따라 random 또는 fixed).
    여기서는 점 jitter + 밝기 shift 만.
    """
    xyz = xyz + np.random.normal(0, 0.005, size=xyz.shape).astype(np.float32)
    brightness_shift = np.random.randint(-20, 21)
    rgb = np.clip(rgb.astype(np.int32) + brightness_shift, 0, 255).astype(np.uint8)
    return xyz, rgb, normal


def get_voxel_indices(points, voxel_size):
    return set(tuple(idx) for idx in np.floor(points / voxel_size).astype(int))


# ─────────────────────────────────────────────
#  Scene 합성 — keep_ratio + scene_aug 통합
# ─────────────────────────────────────────────

NUBJUKI_NPOINTS_RANGE = (2000, 27000)


def synthesize_scene(scene_pth_path, mesh, output_path,
                     keep_ratio,                  # 1.0 이면 drop 없음
                     scene_aug,                   # True 면 jitter + brightness shift
                     voxel_size=0.1,
                     npoints_range=NUBJUKI_NPOINTS_RANGE):
    """배경 로드 → keep_ratio 적용 → scene_aug 적용 → 1-5 마리 nubjuki 배치 → 정규화 + 저장.

    keep_ratio < 1.0 이면 background 점들 일부 random drop.
    scene_aug=True 면 점 jitter (σ=0.005) + brightness shift (±20) 적용.
    """
    scene_data = torch.load(scene_pth_path, weights_only=False)
    bg_xyz    = scene_data['xyz'].numpy() if torch.is_tensor(scene_data['xyz']) else scene_data['xyz']
    bg_rgb    = scene_data['rgb'].numpy() if torch.is_tensor(scene_data['rgb']) else scene_data['rgb']
    bg_normal = scene_data.get('normal', np.zeros_like(bg_xyz))
    if torch.is_tensor(bg_normal):
        bg_normal = bg_normal.numpy()
    if bg_rgb.max() <= 1.1:
        bg_rgb = (bg_rgb * 255).astype(np.uint8)

    # ── 1. keep_ratio 적용 (background 점만 drop) ──
    if keep_ratio < 1.0:
        n_full = len(bg_xyz)
        n_keep = int(n_full * keep_ratio)
        keep_idx = np.random.choice(n_full, n_keep, replace=False)
        bg_xyz = bg_xyz[keep_idx]
        bg_rgb = bg_rgb[keep_idx]
        bg_normal = bg_normal[keep_idx]

    # ── 2. scene aug (jitter + brightness, optional) ──
    if scene_aug:
        bg_xyz, bg_rgb, bg_normal = scene_aug_jitter_and_brightness(bg_xyz, bg_rgb, bg_normal)

    bg_labels = np.zeros(len(bg_xyz), dtype=np.int32)

    # ── 3. spawn 영역 + nubjuki 배치 (generate_data4 와 동일) ──
    bg_voxel_set = get_voxel_indices(bg_xyz, voxel_size)
    bg_min = np.percentile(bg_xyz, 2, axis=0)
    bg_max = np.percentile(bg_xyz, 98, axis=0)
    diag   = np.linalg.norm(bg_max - bg_min)
    margin = (bg_max - bg_min) * 0.10
    spawn_min = bg_min + margin
    spawn_max = bg_max - margin

    num_objects = random.randint(1, 5)
    all_xyz, all_rgb, all_normal, all_labels = [bg_xyz], [bg_rgb], [bg_normal], [bg_labels]

    for i in range(1, num_objects + 1):
        success = False
        for _ in range(100):
            n_pts = int(np.random.randint(npoints_range[0], npoints_range[1] + 1))
            obj_pts_raw, obj_cls_raw, obj_nrm_raw = sample_nubjuki(mesh, n_pts)
            curr_pts, curr_cls, curr_nrm = apply_augmentations(obj_pts_raw, obj_cls_raw, obj_nrm_raw)
            curr_pts, curr_cls, curr_nrm = augment_object_extra(curr_pts, curr_cls, curr_nrm)

            scale_ratio = np.random.uniform(0.025, 0.2)
            curr_pts *= (diag * scale_ratio /
                         (np.linalg.norm(curr_pts.max(0) - curr_pts.min(0)) + 1e-6))
            rand_pos = np.random.uniform(spawn_min, spawn_max)
            temp_pts = curr_pts - curr_pts.min(0) + rand_pos

            obj_voxels = get_voxel_indices(temp_pts, voxel_size)
            if len(obj_voxels) == 0:
                continue
            intersection_voxels = obj_voxels & bg_voxel_set
            overlap_ratio = len(intersection_voxels) / len(obj_voxels)
            if overlap_ratio <= 0.20:
                all_xyz.append(temp_pts.astype(np.float32))
                all_rgb.append(curr_cls.astype(np.uint8))
                all_normal.append(curr_nrm.astype(np.float32))
                all_labels.append(np.full(len(temp_pts), i, dtype=np.int32))
                bg_voxel_set.update(obj_voxels)
                success = True
                break
        if not success:
            print(f"  Warning: Object {i} placement failed after 100 attempts.", flush=True)

    # ── 4. 후처리: concat + RGB normalize + unit-ball + normal unit-vector ──
    xyz    = np.concatenate(all_xyz,    axis=0).astype(np.float32)
    rgb    = np.concatenate(all_rgb,    axis=0).astype(np.float32)
    normal = np.concatenate(all_normal, axis=0).astype(np.float32)
    labels = np.concatenate(all_labels, axis=0).astype(np.int32)

    if rgb.max() > 1.0:
        rgb = rgb / 255.0

    centroid = xyz.mean(axis=0)
    xyz = xyz - centroid
    radius = float(np.max(np.sqrt((xyz ** 2).sum(axis=1))))
    if radius > 1e-8:
        xyz = xyz / radius
    n_norm = np.linalg.norm(normal, axis=1, keepdims=True)
    normal = np.divide(normal, n_norm, out=normal, where=n_norm != 0)

    np.save(output_path, {
        'xyz': xyz.astype(np.float32),
        'rgb': rgb.astype(np.float32),
        'normal': normal.astype(np.float32),
        'instance_labels': labels.astype(np.int32),
    })


# ─────────────────────────────────────────────
#  Split 별 정책
# ─────────────────────────────────────────────
#
# 'keep_ratios' 의 의미:
#   - None         : 매 copy 마다 random keep_ratio in [0.95, 1.0] (generate_data4 식)
#   - [list]       : list cycle (generate_data5 식). copies 는 list 길이로 결정.
#
# 'scene_aug':
#   - True  : 첫 copy 제외 모든 copy 에 scene aug (jitter + brightness) 적용
#   - False : aug 없음 (generate_data5 식, 결정적)

SPLIT_CONFIG = {
    # 'train': {
    #     # 6 keep_ratio × 2 cycles = 12 copies (학습 데이터에 fg/bg 다양화)
    #     'keep_ratios': [1.0, 0.9, 0.8, 0.7, 0.6, 0.5,
    #                     1.0, 0.9, 0.8, 0.7, 0.6, 0.5],
    #     'scene_aug': True,             # copy_idx > 0 에 jitter + brightness shift
    # },
    # 'val': {
    #     # 6 keep_ratio × 1 cycle = 6 copies (결정적 monitor)
    #     'keep_ratios': [1.0, 0.9, 0.8, 0.7, 0.6, 0.5],
    #     'scene_aug': False,
    # },
    'test': {
        # 1 copy, keep_ratio = 1.0 만 (hold-out)
        'keep_ratios': [1.0],
        'scene_aug': False,
    },
}


# ─────────────────────────────────────────────
#  Worker 헬퍼
# ─────────────────────────────────────────────

_MESH = None


def _init_worker(glb_path, seed_base):
    global _MESH
    _MESH = load_nubjuki_mesh(glb_path)
    pid = os.getpid()
    np.random.seed((seed_base + pid) % (2 ** 31 - 1))
    random.seed((seed_base + pid) % (2 ** 31 - 1))


def _gen_one(task):
    src_path, out_path, keep_ratio, scene_aug = task
    try:
        synthesize_scene(src_path, _MESH, out_path,
                         keep_ratio=keep_ratio, scene_aug=scene_aug, voxel_size=0.1)
        return ('ok', out_path)
    except Exception as e:
        return ('err', f'{out_path}: {e}')


# ─────────────────────────────────────────────
#  Task 수집
# ─────────────────────────────────────────────

def collect_tasks(multiscan_root, output_root):
    """SPLIT_CONFIG 따라 task list 만들기. 모든 split 통일된 처리.

    - copy 수 = len(keep_ratios)
    - keep_ratio[copy_idx] 적용
    - 파일명: {base}_c{copy_idx:02d}_kr{kr*100:03d}.npy   (copy_idx 포함으로 중복 ratio 충돌 방지)
    - scene_aug: split 의 scene_aug=True 면 copy_idx > 0 에만 jitter+brightness 적용
                 (copy_idx == 0 은 baseline 보존)
    """
    tasks = []
    summary = {}
    for split, cfg in SPLIT_CONFIG.items():
        ms_dir = os.path.join(multiscan_root, split)
        out_dir = os.path.join(output_root, split)
        os.makedirs(out_dir, exist_ok=True)
        if not os.path.exists(ms_dir):
            print(f'[skip] multiscan/{split} not found', flush=True)
            continue
        pth_files = sorted(f for f in os.listdir(ms_dir) if f.endswith('.pth'))

        keep_ratios = cfg['keep_ratios']
        scene_aug = cfg['scene_aug']
        n_copies = len(keep_ratios)

        added = skipped = 0
        for fn in pth_files:
            base_name = fn[:-4]
            src_path = os.path.join(ms_dir, fn)
            for copy_idx, kr in enumerate(keep_ratios):
                out_name = f'{base_name}_c{copy_idx:02d}_kr{int(kr*100):03d}.npy'
                out_path = os.path.join(out_dir, out_name)
                if os.path.exists(out_path):
                    skipped += 1
                    continue
                # scene_aug: copy_idx == 0 (baseline) 만 제외, 나머지 적용
                aug_this = scene_aug and copy_idx > 0
                tasks.append((src_path, out_path, float(kr), aug_this))
                added += 1

        summary[split] = (len(pth_files), n_copies, skipped, added)
    return tasks, summary


# ─────────────────────────────────────────────
#  메인
# ─────────────────────────────────────────────

if __name__ == '__main__':
    glb_path = '/home/ubuntu/CS479-Seg/assets/sample.glb'
    multiscan_root = '/home/ubuntu/CS479-Seg/assets/multiscan'
    output_root = '/home/ubuntu/CS479-Seg/yoon/generated_data_seyoon'
    SEED_BASE = 20260504
    NUM_WORKERS = 6

    np.random.seed(SEED_BASE); random.seed(SEED_BASE)
    os.makedirs(output_root, exist_ok=True)

    tasks, summary = collect_tasks(multiscan_root, output_root)
    print(f'=== generate_data6 (4+5 통합) ===', flush=True)
    print(f'output_root: {output_root}', flush=True)
    for split, (n_src, n_copies, skipped, added) in summary.items():
        cfg = SPLIT_CONFIG[split]
        kr_set = sorted(set(cfg['keep_ratios']), reverse=True)
        n_per_kr = len(cfg['keep_ratios']) // len(kr_set) if kr_set else 0
        print(f'  [{split:5}] sources={n_src:3} × copies={n_copies:2}  '
              f'keep_ratios={kr_set} ×{n_per_kr}  scene_aug={cfg["scene_aug"]}  '
              f'existing={skipped}  to_generate={added}', flush=True)
    print(f'  total to generate: {len(tasks)}', flush=True)
    if not tasks:
        print('nothing to do'); exit(0)

    t0 = time.time()
    with mp.Pool(NUM_WORKERS, initializer=_init_worker, initargs=(glb_path, SEED_BASE)) as pool:
        n_ok = n_err = 0
        for i, (status, msg) in enumerate(pool.imap_unordered(_gen_one, tasks), start=1):
            if status == 'ok': n_ok += 1
            else: n_err += 1; print(f'  [err] {msg}', flush=True)
            if i % 100 == 0:
                rate = i / (time.time() - t0)
                eta = (len(tasks) - i) / max(1e-6, rate) / 60
                print(f'  progress {i}/{len(tasks)}  rate={rate:.1f} files/s  eta={eta:.1f}min', flush=True)
    dt = (time.time() - t0) / 60
    print(f'\n=== done in {dt:.1f}min — ok={n_ok} err={n_err} ===', flush=True)
