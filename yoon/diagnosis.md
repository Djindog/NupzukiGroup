# Model Diagnosis: SoftGroup Nubzuki Segmentation

**Date:** 2026-05-06  
**Current F1@25 (official `evaluate.py`):** 0.062  
**Goal:** ≥ 0.90  
**Training validation F1@25 (in `train.py`):** ~0.26 (misleading — see §5)

---

## Quick Summary

64 out of 82 test scenes receive **zero predicted instances**. The few predictions that
exist are almost always off. The core failure is a compound of: (a) the proposal
scoring heads being **completely untrained** due to a training crash, which causes a
confidence filter to silently discard every proposal; (b) the BFS clustering radius
being **15× too small**; and (c) severe class imbalance making the semantic head
predict mostly background. These are fixable. The sections below document each issue
with evidence, then list improvements in priority order.

---

## 1. The Proposal Heads Were Never Trained (Critical)

### What happened
Training was killed mid-epoch-3 by an OOM error (see `train_log.txt`):
```
Train E3:   1%|  | 5/870 [00:51<2:44:59, 11.44s/it]
line 14: 487895 Killed  python yoon/train.py ...
```

Proposal loss only activates at `epoch >= 3` (`train.py:136`). The kill happened at
the very beginning of epoch 3, meaning **`cls_head`, `score_head`, and `seg_head`
received zero gradient updates and remain at random initialization**.

### Why this causes 0 predictions
In `predict()` (`model.py:258–263`):
```python
conf = torch.softmax(cls_l, dim=0)[1] * torch.sigmoid(score.squeeze())
if conf < 0.3 or inst_id > 5:
    break
```
With random weights:
- `torch.softmax(random_2vec, dim=0)[1]` ≈ 0.50 (random logits → near-uniform)
- `torch.sigmoid(random_scalar)` ≈ 0.50

So `conf ≈ 0.25`, which is **below the hardcoded 0.3 threshold**. Every proposal,
regardless of quality, is rejected before being assigned an instance ID. This
mechanically explains the 64/82 zero-prediction scenes.

### Evidence
- `metrics_per_scene.json`: 64/82 scenes have `num_pred_instances: 0`; the remaining
  18 scenes have at most 2 predictions
- `train_log.txt`: only 3 epochs completed (0, 1, 2); epoch 3 OOM killed
- The 3 completed epochs used only semantic + offset loss — no proposal supervision

---

## 2. BFS Clustering Radius — Nuanced, Not the Primary Problem (Low–Medium)

### Clarification
`model.py:190` calls `_cluster(..., radius=0.02, ...)`. `CLAUDE.md` mentions `eps=0.3`
but that appears to be a stale description of an earlier design; 0.02 is not
inherently wrong.

### Why `radius=0.02` is actually fine for connectivity
The BFS radius governs local connectivity *within a converged cluster in offset-shifted
space*. If offset predictions are working, all N points of an instance converge toward
their shared center. The inter-point gap in that converged cluster is:

```
gap ≈ ε × (4π / 3N)^(1/3)
```

where ε is the residual spread radius and N is point count. For a small Nubzuki
(N=2500, ε=0.05 normalized units, i.e. 50% offset error on a 0.1-unit object):

```
gap ≈ 0.05 × (4π / 7500)^(1/3) ≈ 0.005
```

`radius=0.02` is 4× larger than this gap — more than enough for connectivity. With
denser instances (N=10000+), the gap shrinks further. So **the radius is not too small
to connect a well-formed cluster**.

### When radius *does* become the bottleneck
The radius only fails if offset predictions are so poor that shifted points scatter
broadly and randomly rather than converging at all — in which case no radius would
help. This is an offset-quality problem (§4), not a radius problem.

### What remains valid to check
The `VOXEL_SIZE=0.02` equals the cluster radius. Points near voxel boundaries are
feature-averaged across a 0.02-wide cell. After de-voxelization, these points share
the same feature vector and the same predicted offset, so their shifted positions are
identical — they trivially cluster together. This is actually benign: it does not
*hurt* cluster formation, it slightly aids it.

---

## 3. Class Imbalance Without Compensation (Critical)

### The ratio
A typical generated scene has:
- Background: ~100k–225k points
- Nubzuki foreground: ~3k–27k points per instance × 1–5 instances
  → typically 5k–80k total foreground

Background-to-foreground ratio: commonly **5:1 to 30:1**, sometimes worse.

### Effect on training
`train.py:127` uses `F.cross_entropy(sem_logits, sem_gt)` with no class weights.
Plain cross-entropy on imbalanced data makes the model converge to predicting the
majority class (background). The semantic head learns "predict background everywhere"
because this minimizes loss early and the gradient from rare foreground points is
overwhelmed.

### Downstream effect
`_cluster()` gates everything on `fg_mask = sem_prob[:, 1] >= fg_threshold`. If the
semantic head is biased toward background, `fg_mask` is nearly empty → no proposals →
no predictions.

This also explains the OOM crash: if the semantic head is NOT well-calibrated and
assigns moderate foreground probability to many background points, `fg_mask` becomes
very large (>>30k), the BFS processes the full `max_pts=30000` cap, which combined
with proposal scoring during epoch 3 exhausts GPU memory.

---

## 4. Training Ran for Only 3 Epochs Total (High)

### Consequence
- Epochs 0–2: only semantic + offset loss
- Epoch 3: OOM crash, no checkpoint saved for this epoch

The best checkpoint (`best_model.pth`) is from epoch 1 (val_f1@25=0.2660), trained
only on semantic + offset losses. The `last_model.pth` is from epoch 2. Neither has
ever had proposal supervision.

### Loss trajectory
```
Epoch 0 | loss=0.1283 | val_f1@25=0.2586
Epoch 1 | loss=0.0688 | val_f1@25=0.2660  ← saved as best
Epoch 2 | loss=0.0563 | val_f1@25=0.2571
```
Loss is still decreasing at epoch 2; the model was far from converged when training
died. Even the backbone (semantic + offset heads) is undertrained.

---

## 5. Training F1 vs Official Test F1 Gap (High)

### The numbers
- `train.py` validation reports: F1@25 ≈ 0.26
- `evaluate.py` on test data: F1@25 = **0.062**

This is a 4× gap. Several sources contribute:

**a) Different data splits**  
`train.py` validates on `yoon/generated_data/val` (generated data). `evaluate.py`
runs on `yoon/generated_data/test` (a different generated split). The splits share the
same generation pipeline, so this alone shouldn't cause a 4× gap.

**b) Different F1 computation**  
`train.py` computes micro-averaged F1 (global TP/FP/FN pooled across all scenes).
`evaluate.py` computes scene-level F1 and then aggregates. With many zero-prediction
scenes, the official evaluator sees many FN per scene; micro-averaging in `train.py`
can hide this by pooling across scenes where predictions happen to match.

**c) Checkpoint selection**  
The `train.py` validator selects the checkpoint with best `val_f1@25`. If the val set
is small or easier than the test set, the selected checkpoint is overfit to val.

**d) Inference mode differences**  
`validate()` in `train.py` calls `model.predict(feat_N9)` directly; `evaluate.py`
calls `run_inference(model, features)`. These should be equivalent, but `run_inference`
transposes the feature tensor (`features[b].T`) before passing to `predict`. Both do
this correctly, but worth double-checking.

### Key point
The training validation F1 of 0.26 is an overestimate. The true current model
performance is **0.062**, meaning the model produces meaningful predictions in very few
scenes.

---

## 6. Confidence Threshold Hardcoded at 0.3 (High)

`model.py:259`: `if conf < 0.3 or inst_id > 5: break`

This threshold is applied even when proposal heads are at random initialization
(§1). Even after proper training, 0.3 may be too aggressive. A reasonable threshold
should be tuned on validation data rather than hardcoded. Currently it acts as a
complete blocker.

Additionally, the instance cap `inst_id > 5` is hardcoded to 5 (matching max GT
instances). This is correct for the problem but means any extra proposals are always
dropped even if they correspond to valid instances.

---

## 7. No Non-Maximum Suppression (Medium)

`model.py:258–263`: proposals are sorted by confidence and assigned sequentially.
There is no overlap check. Two high-confidence proposals that cover the same Nubzuki
will both get assigned distinct instance IDs, incorrectly **splitting** one GT instance
into two predicted instances. This creates extra FP and a duplicate match that wastes
a prediction slot.

In the few scenes where predictions ARE generated, this likely hurts F1 further.

---

## 8. `fg_threshold` Inconsistency Between Training and Inference (Medium)

- During proposal loss computation (`train.py:55`): `fg_threshold=0.5`
- During inference (`model.py:243`): `fg_threshold=0.2` (the default)

The model is trained to use 0.5 as the decision boundary for "is this point
foreground enough to cluster?", but inference uses 0.2. This means:
- Inference includes many more low-confidence-foreground points
- These additional points are noisier, potentially fragmenting clusters or forming
  spurious proposals
- The BFS processes a much larger set of points at inference, making it slower

The semantic head was never supervised with the 0.2 threshold in mind.

---

## 9. BFS Clustering Is Slow (Medium — Inference Speed)

### Why inference is slow
`_cluster()` (`model.py:207–228`) implements a manual BFS using `cKDTree`:
```python
tree = cKDTree(pts)
for i in range(n):
    if visited[i]: continue
    # BFS with tree.query_ball_point at each step
```

For `n` foreground points, this runs `n` ball queries. Each `query_ball_point` call
is O(log n + k) where k is the number of neighbors. With dense Nubzuki point clouds
and a large foreground set (up to 30k points), the total complexity is O(n × k).

**Validation timing from train_log.txt**: 84 val scenes took ~3 minutes = ~2.1s/scene.
With 82 test scenes and a 300s limit, the budget is ~3.7s/scene — marginal. Edge cases
(large scenes with many predicted foreground points) could exceed the time limit.

### Root cause of slowness
The main bottleneck is the BFS loop in pure Python calling into cKDTree for each
point individually. `sklearn.cluster.DBSCAN` performs the same operation but batches
all neighborhood queries and runs them through optimized C/Cython code, typically
10–50× faster.

Additionally, if the semantic head produces many false-positive foreground points
(due to class imbalance), the foreground set approaches `max_pts=30000` cap on
every scene, maximizing the BFS work.

---

## 10. `min_cluster_size=20` May Filter Small Instances (Low)

`_cluster(..., min_cluster_size=20)` drops any cluster with fewer than 20 points
after BFS. Small Nubzukis (scale_ratio ≈ 0.025–0.035) can have as few as 500–2500
points total, but after offset-shifting and considering prediction errors, the
effective cluster in shifted space might be fragmented. If a small instance's offset
predictions are scattered, no individual BFS cluster may reach 20 points even if the
instance has thousands of points overall.

---

## 11. Generation Pipeline Potential Mismatch (Low)

The `generation.py` parameters (density constant, color jitter ranges) were
calibrated from the 5 provided test cases. These test cases come from scenes with
diagonal ~3.9–4.6 m. The full MultiScan training set has larger scenes (mean diagonal
~7.2 m per `findings.md`). This means:
- Objects in training data are **smaller relative to the scene** (same `scale_range`
  but larger denominator `scene_diagonal`)
- The model sees more extreme scale ratios in training than at test time

Additionally, the generated data uses random augmentations that may not exactly
match the true test generation pipeline, creating a domain gap.

---

## Summary Table

| # | Issue | Severity | Root Effect |
|---|-------|----------|-------------|
| 1 | Proposal heads untrained (OOM crash) | **Critical** | All proposals rejected → 0 predictions |
| 2 | BFS radius 0.02 — fine for connectivity; only hurts if offsets are completely wrong | **Low–Medium** | Cluster formation depends on offset quality, not radius |
| 3 | Class imbalance, no weighting | **Critical** | Semantic head biased to background |
| 4 | Only 3 training epochs, backbone undertrained | **High** | Offset predictions inaccurate |
| 5 | Train vs test F1 gap (0.26 → 0.062) | **High** | Misleading validation; real perf much lower |
| 6 | Hardcoded confidence threshold 0.3 | **High** | Blocks all proposals post-crash |
| 7 | No NMS between proposals | **Medium** | Instance splitting, wasted predictions |
| 8 | fg_threshold 0.2 (infer) vs 0.5 (train) | **Medium** | Noisy fg set, slow inference |
| 9 | BFS clustering is slow (pure Python) | **Medium** | Risk of hitting 300s time limit |
| 10 | min_cluster_size=20 too aggressive | **Low** | Small instances missed |
| 11 | Generation domain gap | **Low** | Generalization slightly degraded |

---

## Proposed Improvements (Prioritized)

### Priority 1: Fix the Confidence Threshold / Untrained Proposal Heads

**Option A (quick patch):** Lower `conf < 0.3` to `conf < 0.0` or remove it entirely.
Accept all proposals and only cap by `inst_id > 5`. This will let untrained proposals
through and improve recall immediately, at the cost of precision. Given the current
recall is ~0, any improvement in recall is a net gain.

**Option B (correct fix):** Actually train the proposal heads. Fix the OOM issue first
(see below), then run training for 30 epochs. The proposal heads should receive
gradient for at least 10–15 epochs.

### Priority 2: Verify Offset Quality Before Adjusting Radius

`radius=0.02` is sufficient for local connectivity in a well-formed cluster (see §2).
The clustering will only fail if offset predictions are so poor that shifted points
don't form a cluster at all — in that case, fix the offset head (more training, class
weighting) rather than enlarging the radius. Consider logging the mean per-point
offset prediction error on validation to confirm whether offsets are converging.

### Priority 3: Fix Class Imbalance

Add class weighting to the semantic loss:
```python
# Compute weight inversely proportional to class frequency
fg_count = (sem_gt == 1).sum().float()
bg_count = (sem_gt == 0).sum().float()
weight = torch.tensor([1.0, bg_count / fg_count.clamp(min=1)], device=device)
L_sem = F.cross_entropy(sem_logits, sem_gt, weight=weight)
```
Alternatively, use **focal loss** (`alpha=0.25, gamma=2`) which automatically down-
weights easy negatives and focuses training on hard, rare positives.

### Priority 4: Fix the OOM Crash During Training

The epoch-3 OOM is caused by the proposal loss computation on large foreground
sets. Mitigations:
- Reduce `max_proposals` in `compute_proposal_losses` from 20 to 5–10
- Reduce `max_pts` in `_cluster` during training from 30000 to 5000–10000
- Clear intermediate tensors after each proposal's backward (use `torch.no_grad()`
  where gradients aren't needed, as currently done for the clustering step)
- Consider accumulating proposal losses and doing one backward per N proposals to
  bound memory

### Priority 5: Train for More Epochs

30 epochs as configured is the minimum. Target at least 15 epochs WITH proposal
training (so epochs 3–17 or more). Use the `--resume` flag once the OOM is fixed.
Monitor both backbone losses and proposal head losses separately in the log.

### Priority 6: Align fg_threshold Between Training and Inference

Use the same threshold (0.5) in both `_cluster()` during inference and during
training's `compute_proposal_losses`. If 0.5 produces too few foreground points
at inference, tune it, but keep training and inference consistent.

### Priority 7: Add Non-Maximum Suppression

After sorting proposals by confidence, before assigning instance IDs, compute pairwise
overlap between proposals and skip any proposal whose IoU with an already-accepted
proposal exceeds a threshold (e.g., 0.3). This prevents the same instance from being
predicted twice.

### Priority 8: Replace BFS with DBSCAN for Inference Speed

```python
from sklearn.cluster import DBSCAN
labels = DBSCAN(eps=0.08, min_samples=10, algorithm='ball_tree', n_jobs=1).fit_predict(pts)
```
This batches all neighborhood queries and runs in C, making it 10–50× faster than
the custom Python BFS. Also remove the `max_pts=30000` cap or increase it, since
DBSCAN scales better.

### Priority 9: Reduce `min_cluster_size` or Make It Adaptive

Lower `min_cluster_size` from 20 to 5–10, or make it proportional to the expected
point count for a given scale. Alternatively, don't filter by cluster size at this
stage and rely on proposal scoring to reject small noisy clusters.

### Priority 10: Consider Focal Loss for Long-Term Training

For the semantic head, focal loss:
```python
L_sem = sigmoid_focal_loss(sem_logits, sem_gt_onehot, alpha=0.25, gamma=2).mean()
```
is significantly more robust than weighted cross-entropy for extreme class imbalance.
`torchvision.ops.sigmoid_focal_loss` provides a ready implementation.

### Priority 11: Fix the Training Validation Metric

The `validate()` function in `train.py` uses micro-averaged F1 which can mask per-
scene failures. Modify it to compute mean-of-per-scene F1 to match `evaluate.py`'s
behavior. This ensures the training validation signal is calibrated against the
official metric.

### Priority 12: Experiment with Augmentation During Training

Add point-level augmentations during training to improve generalization:
- Random subsampling of background points (reduce background density to match Nubzuki
  density)
- Jitter on xyz coordinates (Gaussian noise, σ ≈ 0.002 in normalized space)
- Random global rotation around the vertical axis

---

## Expected Impact of Fixes (Rough Estimate)

| Fixes Applied | Expected F1@25 Range |
|---|---|
| None (current) | 0.062 |
| #1 threshold only (quick patch) | 0.10–0.20 |
| #1 + #3 (threshold + class weights) | 0.25–0.45 |
| #1 + #3 + more training | 0.40–0.60 |
| #1–#5 (+ OOM fix + more training) | 0.50–0.75 |
| #1–#7 (+ NMS + alignment) | 0.65–0.85 |
| All fixes + extended training | potentially ≥ 0.90 |

These are order-of-magnitude estimates based on the severity of each identified bug.
The largest single gains come from the threshold fix (#1) and clustering radius (#2).
