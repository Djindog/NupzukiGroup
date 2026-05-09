"""
generate_data_final.py — Ultimate Reverse-Engineered Masterpiece (Zero-Gap)

[최종 업데이트 통합 내역]
1. 평면 클리핑 절대 방어벽: 얇은 벽(가벽)을 뚫고 2cm 이상 나가는 점이 있으면 배치 즉시 폐기.
2. 0.04m 정밀 복셀 해시: 씬이 커져도 얇은 구조물 충돌을 완벽히 감지하도록 복셀 크기 고정.
3. 스케일/밀도 동적 재탐색: 500번의 루프마다 넙죽이의 크기와 점 개수를 완전히 새로 뽑아 배치 성공률 극대화.
4. 건물 이탈 방지: 법선(Normal) 벡터 방향을 방 한가운데로 강제 교정 & BBox 이탈 차단.
5. 무결성 보장 & 이어하기: 넙죽이가 0개면 무한 재시도하며, 기존에 생성된 npy 파일은 자동 스킵(Resume).
"""

import os
import random

import numpy as np
import torch
import trimesh
from matplotlib.colors import hsv_to_rgb, rgb_to_hsv
from scipy.spatial.transform import Rotation as R


# ─────────────────────────────────────────────
#  상수 — README 파이프라인 규칙 및 하이퍼파라미터
# ─────────────────────────────────────────────

# 객체 개수 (README 규칙)
NUM_OBJECTS_MIN = 1
NUM_OBJECTS_MAX = 5

# 스케일 비율 (Scene Diagonal의 0.025 ~ 0.2)
SCALE_RATIO_MIN = 0.025
SCALE_RATIO_MAX = 0.200

# 자연스러운 점 개수 하한/상한선
NUM_POINTS_MIN = 500
NUM_POINTS_MAX = 50_000 

# [Train 전용] 전체 씬 포인트 상한선 (train.py 환경과 동기화, OOM 방어)
MAX_SAFE_POINTS = 150_000  

# 기타 확률
HARD_NEGATIVE_RATE = 0.10
STACKING_RATE = 0.20
MESH_INIT_SAMPLE = 80_000
SEED = 42


# ─────────────────────────────────────────────
#  GLB 로드 및 유틸리티
# ─────────────────────────────────────────────

def load_nubjuki_model_raw(glb_path: str):
    mesh = trimesh.load(glb_path, force='mesh')
    if isinstance(mesh.visual, trimesh.visual.TextureVisuals):
        mesh.visual = mesh.visual.to_color()

    n = min(MESH_INIT_SAMPLE, max(10_000, len(mesh.faces) * 5))
    points, face_idx = trimesh.sample.sample_surface(mesh, n)
    normals = mesh.face_normals[face_idx]
    colors  = mesh.visual.face_colors[face_idx][:, :3].astype(np.uint8)
    return points, colors, normals


def _resample(points, colors, normals, n: int):
    N = len(points)
    idx = np.random.choice(N, n, replace=(n > N))
    return points[idx], colors[idx], normals[idx]


# ─────────────────────────────────────────────
#  Augmentations (README 규칙 엄수)
# ─────────────────────────────────────────────

def apply_augmentations(points, colors, normals, hard_negative=False, bg_color_mean=None):
    # Anisotropic scaling: (0.5, 1.5)
    sf = np.random.uniform(0.5, 1.5, size=3)
    points  = points * sf
    normals = normals / (sf + 1e-8)
    normals /= (np.linalg.norm(normals, axis=1, keepdims=True) + 1e-8)

    # Affine transform: (-180, 180)
    rot     = R.from_euler('xyz', np.random.uniform(-180, 180, size=3), degrees=True)
    points  = rot.apply(points)
    normals = rot.apply(normals)

    # Color map jittering
    hsv = rgb_to_hsv((colors.astype(np.float64) / 255.0).reshape(1, -1, 3))
    if hard_negative and bg_color_mean is not None:
        bg_hsv = rgb_to_hsv(np.array([[bg_color_mean]]))
        bg_hue = bg_hsv[0, 0, 0]
        hsv[0, :, 0] = np.clip(bg_hue + np.random.uniform(-0.1, 0.1), 0.0, 1.0)
        hsv[0, :, 1] = np.random.uniform(0.5, 0.8)
    else:
        hsv[0, :, 0] = np.random.uniform(0.0, 1.0)
        hsv[0, :, 1] = np.random.uniform(0.7, 1.0)
    colors = (hsv_to_rgb(hsv).reshape(-1, 3) * 255).astype(np.uint8)

    return points, colors, normals


def augment_object_extra(points, colors, normals):
    points = points + np.random.normal(0, 0.002, size=points.shape)
    hsv = rgb_to_hsv((colors.astype(np.float64) / 255.0).reshape(1, -1, 3))
    hsv[0, :, 2] = np.clip(hsv[0, :, 2] + np.random.uniform(-0.15, 0.15), 0.0, 1.0)
    colors = (hsv_to_rgb(hsv).reshape(-1, 3) * 255).astype(np.uint8)
    return points, colors, normals


def augment_scene_pointcloud(xyz, rgb, normal):
    xyz = xyz + np.random.normal(0, 0.005, size=xyz.shape).astype(np.float32)
    n = len(xyz)
    keep = np.random.choice(n, int(n * np.random.uniform(0.95, 1.0)), replace=False)
    xyz, rgb, normal = xyz[keep], rgb[keep], normal[keep]
    shift = np.random.randint(-20, 21)
    rgb = np.clip(rgb.astype(np.int32) + shift, 0, 255).astype(np.uint8)
    return xyz, rgb, normal


def _voxel_set(points, voxel_size):
    return set(map(tuple, np.floor(points / voxel_size).astype(int)))


# ─────────────────────────────────────────────
#  Scene 합성 메인 로직
# ─────────────────────────────────────────────

def synthesize_scene(scene_pth_path: str,
                     obj_raw: tuple,
                     output_path: str,
                     split: str = 'train',
                     use_scene_aug: bool = False,
                     hard_negative: bool = False) -> bool:
    """
    합성 성공 시 True 반환. (넙죽이가 0개 안착했으면 False 반환하여 씬 전체 무한 재시도)
    """
    scene_data = torch.load(scene_pth_path, weights_only=False)

    def _to_np(x):
        return x.numpy() if torch.is_tensor(x) else np.asarray(x)

    bg_xyz    = _to_np(scene_data['xyz']).astype(np.float32)
    bg_rgb    = _to_np(scene_data['rgb'])
    bg_normal = _to_np(scene_data.get('normal', np.zeros_like(bg_xyz))).astype(np.float32)

    if bg_rgb.max() <= 1.1:
        bg_rgb = (bg_rgb * 255).astype(np.uint8)
    else:
        bg_rgb = bg_rgb.astype(np.uint8)

    if use_scene_aug and split == 'train':
        bg_xyz, bg_rgb, bg_normal = augment_scene_pointcloud(bg_xyz, bg_rgb, bg_normal)

    bg_labels = np.zeros(len(bg_xyz), dtype=np.int32)
    bg_color_mean = bg_rgb.mean(0).astype(np.float64) / 255.0

    bg_min_p   = np.percentile(bg_xyz, 2,  axis=0)
    bg_max_p   = np.percentile(bg_xyz, 98, axis=0)
    scene_diag = float(np.linalg.norm(bg_max_p - bg_min_p))
    room_center = bg_xyz.mean(axis=0)  # 건물 이탈 방지용 방 중심점

    # 💡 [핵심 픽스 1] 얇은 벽 충돌을 완벽하게 잡아내기 위해 4cm 정밀 고정 복셀 사용
    voxel_size = 0.04
    bg_voxel   = _voxel_set(bg_xyz, voxel_size)

    num_objects = random.randint(NUM_OBJECTS_MIN, NUM_OBJECTS_MAX)
    all_xyz, all_rgb, all_normal, all_labels = [bg_xyz], [bg_rgb], [bg_normal], [bg_labels]

    placed_objects = []
    obj_pts_raw, obj_cls_raw, obj_nrm_raw = obj_raw

    # 벽면(Normal Z ≈ 0) 스폰 후보 필터링
    wall_mask = np.abs(bg_normal[:, 2]) < 0.3
    valid_spawn_indices = np.where(wall_mask)[0]
    if len(valid_spawn_indices) == 0:  
        valid_spawn_indices = np.arange(len(bg_xyz))

    overlap_thr = 0.40 if hard_negative else 0.20
    bg_n = len(bg_xyz)

    for inst_id in range(1, num_objects + 1):
        do_stack = (inst_id > 1
                    and len(placed_objects) > 0
                    and np.random.rand() < STACKING_RATE)

        placed = False
        # 스폰 시도 500회 루프
        for _ in range(500):
            # 💡 [핵심 픽스 2] 매 루프마다 스케일과 밀도(점 개수)를 무작위로 아예 새롭게 갱신
            target_ratio = np.random.uniform(SCALE_RATIO_MIN, SCALE_RATIO_MAX)
            target_abs_size = scene_diag * target_ratio

            K = np.exp(np.random.uniform(np.log(7), np.log(177)))
            material_variance = np.random.uniform(0.8, 1.2)
            num_points = int(K * bg_n * (target_ratio**2) * material_variance)
            num_points = max(NUM_POINTS_MIN, min(NUM_POINTS_MAX, num_points))

            # 리샘플링
            pts, cls, nrm = _resample(obj_pts_raw, obj_cls_raw, obj_nrm_raw, num_points)

            # Base Scale을 먼저 맞춤
            obj_diag = np.linalg.norm(pts.max(0) - pts.min(0)) + 1e-6
            pts *= (target_abs_size / obj_diag)

            # 스케일이 맞춰진 상태에서 비등방성 증강(Anisotropic) 적용
            pts, cls, nrm = apply_augmentations(pts, cls, nrm, hard_negative, bg_color_mean)
            pts, cls, nrm = augment_object_extra(pts, cls, nrm)

            if do_stack:
                target = random.choice(placed_objects)
                target_top_z, target_xy, target_extent = target
                obj_size = pts.max(0) - pts.min(0)
                offset_xy = np.random.uniform(-target_extent*0.3, target_extent*0.3, size=2)
                
                pos = np.array([
                    target_xy[0] + offset_xy[0],
                    target_xy[1] + offset_xy[1],
                    target_top_z - (obj_size[2] * 0.01), 
                ])
                local_overlap_thr = 0.60 
            else:
                rand_idx = random.choice(valid_spawn_indices)
                surface_pt = bg_xyz[rand_idx]
                surface_nrm = bg_normal[rand_idx]
                surface_nrm = surface_nrm / (np.linalg.norm(surface_nrm) + 1e-8)

                # 방향 교정: 법선 벡터가 항상 방 안쪽을 향하도록 (Void Escaping 방지)
                if np.dot(surface_nrm, room_center - surface_pt) < 0:
                    surface_nrm = -surface_nrm

                obj_center = (pts.max(0) + pts.min(0)) / 2.0
                
                # 내적을 이용해 벽면과 넙죽이 뒷면까지의 실제 두께 거리 계산
                projections = np.dot(pts - obj_center, surface_nrm)
                actual_distance_to_back = np.abs(np.min(projections))
                
                push_dist = actual_distance_to_back + np.random.uniform(0.05, 0.15)
                desired_center = surface_pt + (surface_nrm * push_dist)
                pos = desired_center - (obj_center - pts.min(0))
                
                local_overlap_thr = overlap_thr

            pts_placed = pts - pts.min(0) + pos

            # 씬의 전체 Bounding Box 범위를 벗어난 허공 스폰 원천 차단
            if (pts_placed.min(0) < bg_xyz.min(0)).any() or (pts_placed.max(0) > bg_xyz.max(0)).any():
                continue

            # 💡 [핵심 픽스 3] 평면 클리핑 (Plane Clipping) - 얇은 벽 파묻힘 완전 차단!
            # 벽 방향으로 밀어낸 넙죽이 점들 중 단 1개라도 벽(surface_pt) 반대편으로 2cm 초과 관통했다면 실패
            if not do_stack:
                penetration_depths = np.dot(pts_placed - surface_pt, surface_nrm)
                if (penetration_depths < -0.02).any():
                    continue

            # 복셀 충돌(Overlap) 검사
            obj_vox = _voxel_set(pts_placed, voxel_size)
            if not obj_vox:
                continue

            overlap = len(obj_vox & bg_voxel) / len(obj_vox)
            if overlap <= local_overlap_thr:
                all_xyz.append(pts_placed.astype(np.float32))
                all_rgb.append(cls.astype(np.uint8))
                all_normal.append(nrm.astype(np.float32))
                all_labels.append(np.full(len(pts_placed), inst_id, dtype=np.int32))
                bg_voxel.update(obj_vox)
                
                top_z = pts_placed[:, 2].max()
                center_xy = pts_placed[:, :2].mean(0)
                max_ext = float(np.linalg.norm(pts_placed.max(0) - pts_placed.min(0)))
                placed_objects.append((top_z, center_xy, max_ext))
                placed = True
                break

        if not placed:
            tag = " [STACK]" if do_stack else ""
            print(f"  Warning: Object {inst_id}{tag} placement failed.")

    # 넙죽이가 0개라면 저장 취소 후 메인 루프에서 재시도 유도
    if len(placed_objects) == 0:
        return False

    # ── 합성 후 최종 다운샘플 및 VRAM 안전장치 ──────────────────────
    final_xyz    = np.concatenate(all_xyz,    axis=0)
    final_rgb    = np.concatenate(all_rgb,    axis=0)
    final_normal = np.concatenate(all_normal, axis=0)
    final_labels = np.concatenate(all_labels, axis=0)

    total_points = len(final_xyz)
    n_keep = total_points

    # Train셋 비율 증강(노이즈) 및 VRAM 폭발 방지 캡 적용
    if split == 'train':
        keep_ratio = np.random.uniform(0.4, 0.9)
        n_keep = int(total_points * keep_ratio)
        if n_keep > MAX_SAFE_POINTS:
            n_keep = MAX_SAFE_POINTS

    # val, test 셋은 아무 제약 없이 다운샘플링 0%, 원본 비율 100% (평가 환경과 완벽 동기화)
    if total_points > n_keep:
        keep_all = np.random.choice(total_points, n_keep, replace=False)
        final_xyz    = final_xyz[keep_all]
        final_rgb    = final_rgb[keep_all]
        final_normal = final_normal[keep_all]
        final_labels = final_labels[keep_all]

    # ── 저장 ──────────────────────────────────────────
    final = {
        'xyz':             final_xyz,
        'rgb':             final_rgb,
        'normal':          final_normal,
        'instance_labels': final_labels,
    }
    np.save(output_path, final)

    total = len(final['xyz'])
    fg    = (final['instance_labels'] > 0).sum()
    tag = " [HARD]" if hard_negative else ""
    
    # 실패한 목표 개수 오류 픽스: 실제 안착한 개수(len(placed_objects)) 출력
    print(f"  → total={total:,}, fg={fg:,} ({fg/total*100:.1f}%), "
          f"objects={len(placed_objects)}{tag} [split={split}]")
          
    return True


# ─────────────────────────────────────────────
#  메인 실행부
# ─────────────────────────────────────────────

def collect_scene_files(base_dir: str, splits=('train', 'val')) -> list:
    all_files = []
    for split in splits:
        split_path = os.path.join(base_dir, split)
        if not os.path.exists(split_path):
            print(f"  경로 없음: {split_path}")
            continue
        files = sorted(f for f in os.listdir(split_path) if f.endswith('.pth'))
        all_files.extend([(split, f) for f in files])
    return all_files


if __name__ == "__main__":
    np.random.seed(SEED)
    random.seed(SEED)

    glb_model_path  = '/home/ubuntu/CS479-Seg/assets/sample.glb'
    base_dir        = '/home/ubuntu/CS479-Seg/assets/multiscan'
    output_base_dir = '/home/ubuntu/CS479-Seg/yoon/generated_data_cjm'

    TRAIN_COPIES = 10 

    print("GLB 초기 대량 샘플링 중...")
    obj_raw_data = load_nubjuki_model_raw(glb_model_path)
    print(f"  초기 샘플: {len(obj_raw_data[0]):,}pts\n")

    # train_files = collect_scene_files(base_dir, splits=('train',))
    # print(f"[train] MultiScan train: {len(train_files)} scenes × {TRAIN_COPIES} copies")

    # out_train = os.path.join(output_base_dir, 'train')
    # os.makedirs(out_train, exist_ok=True)

    # for src_split, file_name in train_files:
    #     src_path  = os.path.join(base_dir, src_split, file_name)
    #     base_name = file_name.replace('.pth', '')

    #     for copy_idx in range(TRAIN_COPIES):
    #         use_scene_aug = (copy_idx > 0)
    #         hard_negative = (np.random.rand() < HARD_NEGATIVE_RATE)

    #         out_name = f"{base_name}_aug{copy_idx:02d}.npy"
    #         out_path = os.path.join(out_train, out_name)

    #         # 💡 [핵심 픽스 4] 이미 완성된 npy 파일이 있으면 스킵 (Resume 이어하기 기능)
    #         if os.path.exists(out_path):
    #             print(f"[train] {out_name} already exists. Skipping...")
    #             continue

    #         print(f"[train] {out_name} (scene_aug={use_scene_aug}, hard={hard_negative}) ...")
            
    #         # 💡 [핵심 픽스 5] 넙죽이가 1개라도 무사히 스폰될 때까지 씬 자체를 무한 재시도!
    #         success = False
    #         while not success:
    #             success = synthesize_scene(
    #                 scene_pth_path=src_path,
    #                 obj_raw=obj_raw_data,
    #                 output_path=out_path,
    #                 split='train',
    #                 use_scene_aug=use_scene_aug,
    #                 hard_negative=hard_negative,
    #             )

    # n_train = len(train_files) * TRAIN_COPIES
    # print(f"[train] 완료 — {len(train_files)} × {TRAIN_COPIES} = {n_train} files\n")

    # val_files = collect_scene_files(base_dir, splits=('val',))
    # out_val   = os.path.join(output_base_dir, 'val')
    # os.makedirs(out_val, exist_ok=True)

    # print(f"[val] MultiScan val: {len(val_files)} scenes × 1 copy (증강 없음)")
    # for src_split, file_name in val_files:
    #     src_path = os.path.join(base_dir, src_split, file_name)
    #     out_path = os.path.join(out_val, file_name.replace('.pth', '.npy'))

    #     if os.path.exists(out_path):
    #         print(f"[val] {file_name} already exists. Skipping...")
    #         continue

    #     print(f"[val] {file_name} ...")
    #     success = False
    #     while not success:
    #         success = synthesize_scene(
    #             scene_pth_path=src_path,
    #             obj_raw=obj_raw_data,
    #             output_path=out_path,
    #             split='val',
    #             use_scene_aug=False,
    #             hard_negative=False,
    #         )

    # print(f"[val] 완료 — {len(val_files)} files\n")

    test_files = collect_scene_files(base_dir, splits=('test',))
    if test_files:
        out_test = os.path.join(output_base_dir, 'test')
        os.makedirs(out_test, exist_ok=True)

        print(f"[test] MultiScan test: {len(test_files)} scenes × 1 copy (증강 없음)")
        for src_split, file_name in test_files:
            src_path = os.path.join(base_dir, src_split, file_name)
            out_path = os.path.join(out_test, file_name.replace('.pth', '.npy'))

            if os.path.exists(out_path):
                print(f"[test] {file_name} already exists. Skipping...")
                continue

            print(f"[test] {file_name} ...")
            success = False
            while not success:
                success = synthesize_scene(
                    scene_pth_path=src_path,
                    obj_raw=obj_raw_data,
                    output_path=out_path,
                    split='test',
                    use_scene_aug=False,
                    hard_negative=False,
                )

        print(f"[test] 완료 — {len(test_files)} files\n")

    print("=" * 60)
    print("전체 데이터 생성 완료")
    print("=" * 60)