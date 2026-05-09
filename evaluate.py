import argparse
import json
import os
import time
from datetime import datetime

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import InstancePointCloudDataset
from model import initialize_model, run_inference

MAX_MODEL_PARAMS = 50_000_000
NO_VIS_LOOP_TIME_LIMIT_SECONDS = 300.0
MAX_ALLOWED_INSTANCE_ID = 100


def _labels_to_masks(labels: np.ndarray):
    ids = [int(x) for x in np.unique(labels) if int(x) > 0]
    masks = [(labels == i) for i in ids]
    return ids, masks


def _pairwise_iou_masks(pred_masks: list, gt_masks: list):
    k = len(pred_masks)
    m = len(gt_masks)
    if k == 0 or m == 0:
        return np.zeros((k, m), dtype=np.float32)
    iou = np.zeros((k, m), dtype=np.float32)
    for i in range(k):
        pm = pred_masks[i]
        for j in range(m):
            gm = gt_masks[j]
            inter = np.logical_and(pm, gm).sum()
            union = np.logical_or(pm, gm).sum()
            iou[i, j] = (inter / union) if union > 0 else 0.0
    return iou


def _hungarian_match(iou_mat: np.ndarray):
    if iou_mat.size == 0:
        return np.zeros((0,), dtype=np.int64), np.zeros((0,), dtype=np.int64), np.zeros((0,), dtype=np.float32)
    cost = 1.0 - iou_mat
    row_ind, col_ind = linear_sum_assignment(cost)
    matched_ious = iou_mat[row_ind, col_ind].astype(np.float32)
    return row_ind.astype(np.int64), col_ind.astype(np.int64), matched_ious


def _tp_fp_fn_from_matched(matched_ious: np.ndarray, num_pred: int, num_gt: int, thr: float):
    tp = int(np.sum(matched_ious >= float(thr)))
    fp = int(num_pred - tp)
    fn = int(num_gt - tp)
    return tp, fp, fn


def _prf(tp: int, fp: int, fn: int):
    precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    f1 = float(2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    return precision, recall, f1

def evaluate(args):
    os.makedirs(args.output_dir, exist_ok=True)
    pred_save_dir = os.path.join(args.output_dir, "predictions")
    os.makedirs(pred_save_dir, exist_ok=True)

    vis_save_dir = None
    if args.visualize:
        from visualize import save_instance_visualization
        vis_save_dir = os.path.join(args.output_dir, "visualizations")
        os.makedirs(vis_save_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    dataset = InstancePointCloudDataset(args.test_data_dir, split="all")
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=1,
        pin_memory=(device.type == "cuda"),
        persistent_workers=True,
    )
    model = initialize_model(args.ckpt_path, device=device)
    total_params = int(sum(p.numel() for p in model.parameters()))
    print(f"Model parameter count: {total_params:,}")
    if total_params > MAX_MODEL_PARAMS:
        raise RuntimeError(
            f"[EVAL-ERROR] Model parameter limit exceeded: {total_params:,} > {MAX_MODEL_PARAMS:,}. "
            "Please reduce model size and retry."
        )
    # Only measure F1@0.25 and F1@0.50.
    all_thresholds = [0.25, 0.50]
    agg = {thr: {"tp": 0, "fp": 0, "fn": 0} for thr in all_thresholds}

    per_scene_metrics = []

    with torch.inference_mode():
        if args.vis_views.strip().lower() in {"6", "six", "6view", "6views", "all"}:
            vis_views = ("front", "back", "left", "right", "top", "bottom")
        else:
            vis_views = tuple(v.strip().lower() for v in args.vis_views.split(",") if v.strip())
        if len(vis_views) == 0:
            vis_views = ("front",)
        loop_start_time = time.monotonic()

        for i, batch in enumerate(tqdm(loader, desc="Evaluating Hungarian Instance Segmentation")):
            if (not args.visualize) and (time.monotonic() - loop_start_time > NO_VIS_LOOP_TIME_LIMIT_SECONDS):
                raise RuntimeError(
                    f"[EVAL-ERROR] Main inference loop exceeded {NO_VIS_LOOP_TIME_LIMIT_SECONDS:.0f} seconds "
                    "with --no-visualize."
                )
            features = batch["features"].to(device, non_blocking=(device.type == "cuda"))
            gt_instance = batch["instance_labels"].squeeze(0).cpu().numpy().astype(np.int64)
            scene_path = batch["scene_path"][0]

            pred_instance = run_inference(
                model,
                features,
            ).squeeze(0).cpu().numpy().astype(np.int64)
            # Only instance ids in [1, 100] are allowed. Others are remapped to background.
            pred_instance[pred_instance > MAX_ALLOWED_INSTANCE_ID] = 0

            # Save official prediction format: point-wise instance id.
            stem = os.path.splitext(os.path.basename(scene_path))[0]
            np.save(os.path.join(pred_save_dir, f"{stem}_pred.npy"), pred_instance)

            # Instance Hungarian matching.
            pred_ids, pred_masks = _labels_to_masks(pred_instance)
            gt_ids, gt_masks = _labels_to_masks(gt_instance)
            iou_mat = _pairwise_iou_masks(pred_masks, gt_masks)
            row_ind, col_ind, matched_ious = _hungarian_match(iou_mat)

            matched_pred_to_gt = {}
            if len(row_ind) > 0 and len(pred_ids) > 0 and len(gt_ids) > 0:
                for rr, cc, miou in zip(row_ind, col_ind, matched_ious):
                    if float(miou) <= 0.0:
                        continue
                    matched_pred_to_gt[int(pred_ids[int(rr)])] = int(gt_ids[int(cc)])

            for thr in all_thresholds:
                tp, fp, fn = _tp_fp_fn_from_matched(
                    matched_ious=matched_ious,
                    num_pred=len(pred_masks),
                    num_gt=len(gt_masks),
                    thr=thr,
                )
                agg[thr]["tp"] += tp
                agg[thr]["fp"] += fp
                agg[thr]["fn"] += fn

            tp25, fp25, fn25 = _tp_fp_fn_from_matched(
                matched_ious=matched_ious,
                num_pred=len(pred_masks),
                num_gt=len(gt_masks),
                thr=0.25,
            )
            _, _, f1_25 = _prf(tp25, fp25, fn25)
            tp50, fp50, fn50 = _tp_fp_fn_from_matched(
                matched_ious=matched_ious,
                num_pred=len(pred_masks),
                num_gt=len(gt_masks),
                thr=0.5,
            )
            _, _, f1_50 = _prf(tp50, fp50, fn50)
            scene_metric_entry = {
                "scene": stem,
                "num_gt_instances": int(len(gt_masks)),
                "num_pred_instances": int(len(pred_masks)),
                "f1_25": f1_25,
                "f1_50": f1_50,
            }
            per_scene_metrics.append(scene_metric_entry)

            if args.visualize and (args.vis_limit is None or i < args.vis_limit):
                xyz = features.squeeze(0)[:3, :].T.cpu().numpy()
                rgb = np.clip(features.squeeze(0)[3:6, :].T.cpu().numpy(), 0.0, 1.0)
                for view in vis_views:
                    save_instance_visualization(
                        xyz=xyz,
                        rgb=rgb,
                        gt_instance=gt_instance,
                        pred_instance=pred_instance,
                        save_path=os.path.join(vis_save_dir, f"{stem}_vis_{view}.png"),
                        max_pts=args.vis_max_points,
                        point_size=args.vis_point_size,
                        scene_metrics=scene_metric_entry,
                        view=view,
                        matched_pred_to_gt=matched_pred_to_gt,
                        bbox_q_low=args.vis_bbox_q_low,
                        bbox_q_high=args.vis_bbox_q_high,
                    )

    # Global F1 scores by threshold.
    f1_by_threshold = {}
    for thr in all_thresholds:
        tp = int(agg[thr]["tp"])
        fp = int(agg[thr]["fp"])
        fn = int(agg[thr]["fn"])
        _, _, f1 = _prf(tp, fp, fn)
        f1_by_threshold[str(thr)] = float(f1)

    f1_25 = float(f1_by_threshold[str(0.25)])
    f1_50 = float(f1_by_threshold[str(0.5)])

    metrics = {
        "evaluated_at": datetime.utcnow().isoformat() + "Z",
        "test_data_dir": os.path.abspath(args.test_data_dir),
        "checkpoint_path": os.path.abspath(args.ckpt_path),
        "num_scenes": len(dataset),
        "model_param_count": total_params,
        "model_param_limit": int(MAX_MODEL_PARAMS),
        "max_allowed_instance_id": int(MAX_ALLOWED_INSTANCE_ID),
        "no_visualize_time_limit_seconds": float(NO_VIS_LOOP_TIME_LIMIT_SECONDS),
        "instance_f1_25": f1_25,
        "instance_f1_50": f1_50,
        "instance_f1_by_threshold": f1_by_threshold,
    }

    metrics_path = os.path.join(args.output_dir, args.metrics_file)
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    scene_metrics_path = os.path.join(args.output_dir, "metrics_per_scene.json")
    with open(scene_metrics_path, "w", encoding="utf-8") as f:
        json.dump(per_scene_metrics, f, indent=2)

    print(f"Instance F1 -> @25: {f1_25:.4f}, @50: {f1_50:.4f}")
    print(f"Predictions saved to: {pred_save_dir}")
    if args.visualize:
        print(f"Visualizations saved to: {vis_save_dir}")
    print(f"Metrics saved to: {metrics_path}")
    print(f"Per-scene metrics saved to: {scene_metrics_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hungarian-matching instance evaluator (single category).")
    parser.add_argument("--test-data-dir", type=str, required=True)
    parser.add_argument("--ckpt-path", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)

    parser.add_argument("--visualize", dest="visualize", action="store_true")
    parser.add_argument("--vis-limit", type=int, default=5)
    parser.add_argument("--vis-max-points", type=int, default=50000)
    parser.add_argument("--vis-point-size", type=float, default=3.0)
    parser.add_argument("--vis-bbox-q-low", type=float, default=0.0)
    parser.add_argument("--vis-bbox-q-high", type=float, default=1.0)
    parser.add_argument("--vis-views", type=str, default="front,back,left,right,top,bottom")
    parser.add_argument("--metrics-file", type=str, default="metrics.json")

    evaluate(parser.parse_args())
