import argparse
import os

import numpy as np
import matplotlib
from matplotlib.patches import Rectangle

matplotlib.use("Agg", force=True)


def _load_npy_dict(file_path: str):
    loaded = np.load(file_path, allow_pickle=True)
    if isinstance(loaded, np.ndarray) and loaded.shape == ():
        return loaded.item()
    if isinstance(loaded, np.lib.npyio.NpzFile):
        return {k: loaded[k] for k in loaded.files}
    raise ValueError(f"Unsupported npy/npz structure: {file_path}")


def _color_for_id(inst_id: int, salt: int = 7919):
    rng = np.random.default_rng(int(inst_id) * int(salt))
    return rng.uniform(0.1, 1.0, size=3).astype(np.float32)


def _validate_bbox_quantiles(q_low: float, q_high: float):
    ql = float(q_low)
    qh = float(q_high)
    if not (0.0 <= ql < qh <= 1.0):
        raise ValueError(f"Invalid bbox quantiles: low={ql}, high={qh}. Must satisfy 0 <= low < high <= 1.")
    return ql, qh


def _instance_colors(labels: np.ndarray, id_to_color: dict | None = None, salt: int = 7919):
    labels = labels.astype(np.int64)
    colors = np.zeros((len(labels), 4), dtype=np.float32)
    colors[:, :] = np.array([0.8, 0.8, 0.8, 0.05], dtype=np.float32)

    uniq = np.unique(labels)
    for inst_id in uniq:
        if inst_id <= 0:
            continue
        m = labels == inst_id
        if id_to_color is not None and int(inst_id) in id_to_color:
            col = np.asarray(id_to_color[int(inst_id)], dtype=np.float32)
        else:
            col = _color_for_id(int(inst_id), salt=salt)
        colors[m, :3] = col
        colors[m, 3] = 1.0
    return colors


def _iter_instance_bboxes_2d(
    xyz: np.ndarray,
    labels: np.ndarray,
    min_points: int = 20,
    bbox_q_low: float = 0.02,
    bbox_q_high: float = 0.98,
):
    q_low, q_high = _validate_bbox_quantiles(bbox_q_low, bbox_q_high)
    labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    for inst_id in np.unique(labels):
        if inst_id <= 0:
            continue
        m = labels == inst_id
        if int(m.sum()) < min_points:
            continue
        pts = xyz[m]
        x0 = float(np.quantile(pts[:, 0], q_low))
        x1 = float(np.quantile(pts[:, 0], q_high))
        z0 = float(np.quantile(pts[:, 2], q_low))
        z1 = float(np.quantile(pts[:, 2], q_high))
        if (x1 - x0) <= 1e-8 or (z1 - z0) <= 1e-8:
            continue
        yield inst_id, x0, x1, z0, z1


def _draw_instance_bboxes_2d(
    ax,
    xyz: np.ndarray,
    labels: np.ndarray,
    id_to_color: dict | None = None,
    bbox_q_low: float = 0.02,
    bbox_q_high: float = 0.98,
):
    for inst_id, x0, x1, z0, z1 in _iter_instance_bboxes_2d(
        xyz,
        labels,
        bbox_q_low=bbox_q_low,
        bbox_q_high=bbox_q_high,
    ):
        if id_to_color is not None and int(inst_id) in id_to_color:
            col = np.asarray(id_to_color[int(inst_id)], dtype=np.float32)
        else:
            col = _color_for_id(int(inst_id), salt=7919)
        rect = Rectangle(
            (x0, z0),
            (x1 - x0),
            (z1 - z0),
            fill=False,
            linewidth=1.2,
            edgecolor=col,
            alpha=0.95,
        )
        ax.add_patch(rect)


def _get_view_axes(view: str):
    v = (view or "front").lower()
    if v == "front":
        return 0, 2, 1, 1.0, 1.0, -1.0, "Front X-Z"
    if v == "back":
        return 0, 2, 1, -1.0, 1.0, 1.0, "Back X-Z"
    if v == "left":
        return 1, 2, 0, 1.0, 1.0, -1.0, "Left Y-Z"
    if v == "right":
        return 1, 2, 0, -1.0, 1.0, 1.0, "Right Y-Z"
    if v == "side":
        return 1, 2, 0, 1.0, 1.0, -1.0, "Side Y-Z"
    if v == "top":
        return 0, 1, 2, 1.0, 1.0, -1.0, "Top X-Y"
    if v == "bottom":
        return 0, 1, 2, 1.0, -1.0, 1.0, "Bottom X-Y"
    raise ValueError("Unsupported view: {0}. Use one of: front, back, left, right, top, bottom".format(view))


def _project_uvd(xyz: np.ndarray, view: str):
    ax_u, ax_v, ax_d, flip_u, flip_v, depth_sign, view_title = _get_view_axes(view)
    u = xyz[:, ax_u] * float(flip_u)
    v = xyz[:, ax_v] * float(flip_v)
    d = xyz[:, ax_d] * float(depth_sign)
    return u, v, d, view_title


def _zbuffer_visible_indices(
    xyz: np.ndarray,
    view: str,
    image_size: int = 900,
):
    if len(xyz) == 0:
        return np.zeros((0,), dtype=np.int64)

    u, v, d, _ = _project_uvd(xyz, view=view)

    u_min = float(np.min(u))
    u_max = float(np.max(u))
    v_min = float(np.min(v))
    v_max = float(np.max(v))
    du = max(u_max - u_min, 1e-8)
    dv = max(v_max - v_min, 1e-8)

    px = np.clip(((u - u_min) / du * (image_size - 1)).astype(np.int64), 0, image_size - 1)
    py = np.clip(((v - v_min) / dv * (image_size - 1)).astype(np.int64), 0, image_size - 1)
    pix = py * image_size + px

    # z-buffer equivalent for point clouds: keep nearest (minimum depth) per pixel.
    order = np.lexsort((d, pix))
    pix_sorted = pix[order]
    first = np.ones(len(order), dtype=bool)
    if len(order) > 1:
        first[1:] = pix_sorted[1:] != pix_sorted[:-1]
    return order[first]


def _iter_instance_bboxes_by_axes(
    xyz: np.ndarray,
    labels: np.ndarray,
    ax_u: int,
    ax_v: int,
    flip_u: float = 1.0,
    flip_v: float = 1.0,
    min_points: int = 20,
    bbox_q_low: float = 0.02,
    bbox_q_high: float = 0.98,
):
    q_low, q_high = _validate_bbox_quantiles(bbox_q_low, bbox_q_high)
    labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    for inst_id in np.unique(labels):
        if inst_id <= 0:
            continue
        m = labels == inst_id
        if int(m.sum()) < min_points:
            continue
        pts = xyz[m]
        pu = pts[:, ax_u] * float(flip_u)
        pv = pts[:, ax_v] * float(flip_v)
        u0 = float(np.quantile(pu, q_low))
        u1 = float(np.quantile(pu, q_high))
        v0 = float(np.quantile(pv, q_low))
        v1 = float(np.quantile(pv, q_high))
        if (u1 - u0) <= 1e-8 or (v1 - v0) <= 1e-8:
            continue
        yield inst_id, u0, u1, v0, v1


def _draw_instance_bboxes_2d_view(
    ax,
    xyz: np.ndarray,
    labels: np.ndarray,
    ax_u: int,
    ax_v: int,
    flip_u: float = 1.0,
    flip_v: float = 1.0,
    id_to_color: dict | None = None,
    bbox_q_low: float = 0.02,
    bbox_q_high: float = 0.98,
):
    for inst_id, u0, u1, v0, v1 in _iter_instance_bboxes_by_axes(
        xyz,
        labels,
        ax_u=ax_u,
        ax_v=ax_v,
        flip_u=flip_u,
        flip_v=flip_v,
        bbox_q_low=bbox_q_low,
        bbox_q_high=bbox_q_high,
    ):
        if id_to_color is not None and int(inst_id) in id_to_color:
            col = np.asarray(id_to_color[int(inst_id)], dtype=np.float32)
        else:
            col = _color_for_id(int(inst_id), salt=7919)
        rect = Rectangle(
            (u0, v0),
            (u1 - u0),
            (v1 - v0),
            fill=False,
            linewidth=1.2,
            edgecolor=col,
            alpha=0.95,
        )
        ax.add_patch(rect)


def _format_scene_metrics(scene_metrics):
    if not scene_metrics:
        return ""
    keys = [
        ("num_gt_instances", "GT#"),
        ("num_pred_instances", "Pred#"),
        ("f1_50", "F1@50"),
        ("precision50", "P@50"),
        ("recall50", "R@50"),
        ("mean_matched_iou", "MeanMatchedIoU"),
        ("mean_best_iou", "MeanBestIoU"),
        ("count_error_abs", "|Kpred-Kgt|"),
    ]
    parts = []
    for key, name in keys:
        if key in scene_metrics:
            value = scene_metrics[key]
            if isinstance(value, (int, np.integer)):
                parts.append(f"{name}: {int(value)}")
            else:
                parts.append(f"{name}: {float(value):.4f}")
    return " | ".join(parts)


def _proposal_to_point_instance(pred_masks: np.ndarray, pred_scores: np.ndarray, n_points: int):
    if pred_masks.size == 0:
        return np.zeros((n_points,), dtype=np.int64)

    pred_masks = np.asarray(pred_masks, dtype=bool)
    pred_scores = np.asarray(pred_scores, dtype=np.float32).reshape(-1)
    if pred_masks.ndim != 2:
        raise ValueError(f"pred_masks must be 2D [K, N], got {pred_masks.shape}")
    if pred_masks.shape[0] != pred_scores.shape[0]:
        raise ValueError(f"mask/score mismatch: masks={pred_masks.shape}, scores={pred_scores.shape}")
    if pred_masks.shape[1] != n_points:
        raise ValueError(f"mask length mismatch: mask N={pred_masks.shape[1]} vs points N={n_points}")

    order = np.argsort(-pred_scores)
    point_scores = np.full((n_points,), -np.inf, dtype=np.float32)
    point_inst = np.zeros((n_points,), dtype=np.int64)
    for rank, idx in enumerate(order, start=1):
        m = pred_masks[int(idx)]
        s = float(pred_scores[int(idx)])
        overwrite = np.logical_and(m, s > point_scores)
        point_scores[overwrite] = s
        point_inst[overwrite] = int(rank)
    return point_inst


def save_instance_visualization(
    xyz,
    rgb,
    gt_instance,
    pred_instance,
    save_path,
    max_pts=6000,
    point_size=3.0,
    scene_metrics=None,
    view="front",
    matched_pred_to_gt: dict | None = None,
    bbox_q_low: float = 0.02,
    bbox_q_high: float = 0.98,
):
    import matplotlib.pyplot as plt

    orig_xyz = np.asarray(xyz, dtype=np.float32)
    orig_gt = np.asarray(gt_instance, dtype=np.int64)
    orig_pred = np.asarray(pred_instance, dtype=np.int64)
    _, _, _, view_title = _project_uvd(orig_xyz, view=view)
    gt_count = int(np.sum(np.unique(orig_gt) > 0))
    pred_count = int(np.sum(np.unique(orig_pred) > 0))

    gt_color_map = {}
    for gid in np.unique(orig_gt):
        gid = int(gid)
        if gid <= 0:
            continue
        gt_color_map[gid] = _color_for_id(gid, salt=7919)

    pred_color_map = {}
    for pid in np.unique(orig_pred):
        pid = int(pid)
        if pid <= 0:
            continue
        mapped_gid = None
        if matched_pred_to_gt is not None:
            mapped_gid = matched_pred_to_gt.get(pid)
        if mapped_gid is not None and int(mapped_gid) in gt_color_map:
            pred_color_map[pid] = gt_color_map[int(mapped_gid)]
        else:
            pred_color_map[pid] = _color_for_id(pid, salt=15485863)

    if len(xyz) > max_pts:
        idx = np.random.choice(len(xyz), max_pts, replace=False)
        xyz = xyz[idx]
        rgb = rgb[idx]
        gt_instance = gt_instance[idx]
        pred_instance = pred_instance[idx]

    vis_idx = _zbuffer_visible_indices(np.asarray(xyz, dtype=np.float32), view=view, image_size=900)
    xyz = xyz[vis_idx]
    rgb = rgb[vis_idx]
    gt_instance = gt_instance[vis_idx]
    pred_instance = pred_instance[vis_idx]
    u, v, _, _ = _project_uvd(np.asarray(xyz, dtype=np.float32), view=view)

    fig = plt.figure(figsize=(16, 5.5))

    # 2D projection with z-buffer visibility filtering.
    ax1 = fig.add_subplot(131)
    ax1.scatter(u, v, c=rgb, s=point_size)
    ax1.set_title(f"Input RGB ({view_title})")
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax1.set_aspect("equal", adjustable="box")

    ax2 = fig.add_subplot(132)
    ax2.scatter(u, v, c=_instance_colors(gt_instance, id_to_color=gt_color_map, salt=7919), s=point_size)
    ax_u, ax_v, _, flip_u, flip_v, _, _ = _get_view_axes(view)
    _draw_instance_bboxes_2d_view(
        ax2,
        orig_xyz,
        orig_gt,
        ax_u=ax_u,
        ax_v=ax_v,
        flip_u=flip_u,
        flip_v=flip_v,
        id_to_color=gt_color_map,
        bbox_q_low=bbox_q_low,
        bbox_q_high=bbox_q_high,
    )
    ax2.set_title(f"GT Instances + BBox ({view_title}) | count={gt_count}")
    ax2.set_xticks([])
    ax2.set_yticks([])
    ax2.set_aspect("equal", adjustable="box")

    ax3 = fig.add_subplot(133)
    ax3.scatter(u, v, c=_instance_colors(pred_instance, id_to_color=pred_color_map, salt=15485863), s=point_size)
    _draw_instance_bboxes_2d_view(
        ax3,
        orig_xyz,
        orig_pred,
        ax_u=ax_u,
        ax_v=ax_v,
        flip_u=flip_u,
        flip_v=flip_v,
        id_to_color=pred_color_map,
        bbox_q_low=bbox_q_low,
        bbox_q_high=bbox_q_high,
    )
    ax3.set_title(f"Pred Instances + BBox ({view_title}) | count={pred_count}")
    ax3.set_xticks([])
    ax3.set_yticks([])
    ax3.set_aspect("equal", adjustable="box")

    metric_text = _format_scene_metrics(scene_metrics)
    count_text = f"GT#: {gt_count} | Pred#: {pred_count}"
    if metric_text:
        fig.suptitle(f"{count_text} | {metric_text}", fontsize=11)
    else:
        fig.suptitle(count_text, fontsize=11)

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight", dpi=120)
    plt.close()


def visualize_from_files(
    data_npy_path: str,
    pred_npy_path: str,
    output_path: str,
    max_points: int = 6000,
    point_size: float = 3.0,
    views: tuple = ("front", "back", "left", "right", "top", "bottom"),
    bbox_q_low: float = 0.02,
    bbox_q_high: float = 0.98,
):
    data = _load_npy_dict(data_npy_path)

    xyz = np.asarray(data["xyz"], dtype=np.float32)
    rgb = np.asarray(data["rgb"], dtype=np.float32)
    if rgb.max() > 1.0:
        rgb = rgb / 255.0

    if "instance_labels" in data:
        gt_instance = np.asarray(data["instance_labels"], dtype=np.int64)
    else:
        gt_instance = np.asarray(data["is_mesh"], dtype=np.int64)

    loaded_pred = np.load(pred_npy_path, allow_pickle=True)
    if isinstance(loaded_pred, np.lib.npyio.NpzFile):
        if "masks" not in loaded_pred.files:
            raise ValueError(f"Prediction npz must contain 'masks': {pred_npy_path}")
        pred_masks = np.asarray(loaded_pred["masks"]).astype(bool)
        if "scores" in loaded_pred.files:
            pred_scores = np.asarray(loaded_pred["scores"], dtype=np.float32)
        else:
            pred_scores = np.ones((pred_masks.shape[0],), dtype=np.float32)
        pred_instance = _proposal_to_point_instance(pred_masks, pred_scores, n_points=len(xyz))
    else:
        pred_instance = np.asarray(loaded_pred, dtype=np.int64).reshape(-1)

    if len(pred_instance) != len(xyz):
        raise ValueError(
            f"Length mismatch: pred={len(pred_instance)} vs points={len(xyz)} | "
            f"data={data_npy_path}, pred={pred_npy_path}"
        )

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    stem, ext = os.path.splitext(output_path)
    for view in views:
        out_path = f"{stem}_{view}{ext}" if len(views) > 1 else output_path
        save_instance_visualization(
            xyz,
            rgb,
            gt_instance,
            pred_instance,
            out_path,
            max_pts=max_points,
            point_size=point_size,
            view=view,
            bbox_q_low=bbox_q_low,
            bbox_q_high=bbox_q_high,
        )


def main():
    parser = argparse.ArgumentParser(description="Visualize one scene and one prediction result.")
    parser.add_argument("--data-npy", type=str, required=True, help="Path to one *_aug.npy scene file")
    parser.add_argument("--pred-npy", type=str, required=True, help="Path to one *_pred.npz or *_pred.npy file")
    parser.add_argument("--output", type=str, required=True, help="Output image path")
    parser.add_argument("--max-points", type=int, default=6000, help="Max points to draw")
    parser.add_argument("--point-size", type=float, default=3.0, help="Scatter point size")
    parser.add_argument("--bbox-q-low", type=float, default=0.02, help="Lower quantile for bbox outlier trimming")
    parser.add_argument("--bbox-q-high", type=float, default=0.98, help="Upper quantile for bbox outlier trimming")
    parser.add_argument(
        "--views",
        type=str,
        default="front,back,left,right,top,bottom",
        help="Comma-separated: front,back,left,right,top,bottom (or use '6')",
    )
    args = parser.parse_args()

    if args.views.strip().lower() in {"6", "six", "6view", "6views", "all"}:
        views = ("front", "back", "left", "right", "top", "bottom")
    else:
        views = tuple(v.strip().lower() for v in args.views.split(",") if v.strip())
    visualize_from_files(
        args.data_npy,
        args.pred_npy,
        args.output,
        max_points=args.max_points,
        point_size=args.point_size,
        views=views,
        bbox_q_low=args.bbox_q_low,
        bbox_q_high=args.bbox_q_high,
    )
    print(f"Saved visualization to: {args.output}")


if __name__ == "__main__":
    main()
