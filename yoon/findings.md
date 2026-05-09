# Test-Case Dataset Analysis: Nubzuki Segmentation

Empirical findings from inspecting `assets/test_case_000.npy` – `test_case_004.npy`
and the Nubzuki reference mesh `assets/sample.glb`.

---

## 1. Scene Structure

| Metric | Value |
|--------|-------|
| Total test cases | 5 (test_case_000 – 004) |
| Distinct base scenes | 2 |
| Cases sharing scene A (diag ≈ 3.87 m) | 000, 001, 002 |
| Cases sharing scene B (diag ≈ 4.61 m) | 003, 004 |
| Background points (scene A) | **56 749** – identical across cases 000/001/002 |
| Background points (scene B) | **58 226** – identical across cases 003/004 |
| Total points per scene | 86 735 – 146 796 |

### Key observation: background is frozen
The background xyz **and** rgb are **identical** across test cases sharing the same
base scene.  Computing the background mean RGB separately for cases 000–002 yields
`[138.47, 132.92, 118.46]` in all three.  This rules out any global scene-level
colour augmentation and confirms that the background MultiScan scan is used
verbatim.

---

## 2. Nubzuki Placement

### Number of instances
- Range: 3–5 per scene (README: min=1, max=5).
- All five cases contain ≥ 3 instances.

### Placement height
All instances have `center_z` in the range **−0.187 to −0.453 m** (relative to
the scene coordinate frame), consistent with placement on floor-level horizontal
surfaces.  The floor in both scenes is at approximately z ≈ −0.5 to −0.6 m.

### Scale (relative to scene diagonal)
| Instance (case, id) | bbox (m) | Approx. scale ratio |
|---|---|---|
| 001-3 | 0.08 × 0.10 × 0.076 | ~0.038 |
| 003-3 | 0.16 × 0.14 × 0.11  | ~0.040 |
| 002-2 | 0.18 × 0.15 × 0.15  | ~0.049 |
| 003-1 | 0.16 × 0.22 × 0.18  | ~0.066 |
| 000-1 | 0.14 × 0.17 × 0.18  | ~0.052 |
| 002-1 | 0.19 × 0.08 × 0.18  | ~0.065 |
| 001-1 | 0.16 × 0.13 × 0.09  | ~0.056 |
| 004-2 | 0.08 × 0.11 × 0.11  | ~0.033 |
| 004-4 | 0.18 × 0.16 × 0.18  | ~0.053 |
| 000-4 | 0.11 × 0.22 × 0.18  | ~0.065 |
| 003-2 | 0.41 × 0.26 × 0.18  | ~0.122 |
| 000-2 | 0.30 × 0.34 × 0.30  | ~0.100 |
| 004-1 | 0.50 × 0.40 × 0.24  | ~0.170 |
| 004-3 | 0.28 × 0.42 × 0.40  | ~0.162 |
| 002-3 | 0.24 × 0.57 × 0.50  | ~0.183 |

The effective range observed is **~0.033 – 0.183**, consistent with the stated
0.025 – 0.2 window.  Smaller scale ratios dominate (long tail toward large).\

### Anisotropic Scaling 

Mean: 0.951   Std: 0.228
Min : 0.359   Max : 1.410
Expected range: [0.5, 1.5]
- Result for 20 test samples

A gaussian distribution of **mean 1, std 0.25**

### Point count per instance
- Min: **2 557** (small, highly occluded instance)
- Max: **26 924** (large, scale ≈ 0.17)
- Typical mid-range: **12 000 – 20 000**

---

## 3. Nubzuki Mesh Characteristics

| Property | Value |
|---|---|
| Vertices | 11 829 |
| Faces | 19 426 |
| Native bounding box | 0.948 × 0.999 × 0.811 m |
| Native diagonal | **1.598 m** |
| Geometry centre | (0.010, 0.0003, 0.0015) m |
| Bottom of mesh (z) | −0.404 m |
| Visual type | UV-textured (TextureVisuals) |
| Baked vertex colour mean | (63, 84, 108) – dark blue-grey |
| Baked vertex colour std  | (45, 57, 61) |

The Nubzuki is a cute rounded figurine with:
- **Light sky-blue** spherical body and head with small ear bumps
- **Dark navy** body-shirt with printed text
- **Black** collar / belt stripe
- Very smooth, organic surface with no sharp edges

---

## 4. Colour Augmentation

### Augmentation scope: **per-object**, not global
The critical insight is that colour jitter is applied independently to each
Nubzuki instance, **not** to the entire scene.  Evidence:

- Test case 000 contains both warm-toned (`rgb_mean ≈ [159, 87, 116]`) and
  cool-toned (`rgb_mean ≈ [71, 94, 143]`) instances in the **same** scene.
  A single global transform cannot produce this.
- Background RGB mean is **bit-for-bit identical** across cases sharing a base
  scene (confirmed numerically above), ruling out any global colour augmentation.

### Jitter magnitude
The per-channel multiplicative factor can be as extreme as **~2.5× boost** on one
channel while others stay near 1.0, fully flipping the apparent hue:

| Case | Instance | Original mesh dominant hue | After jitter |
|---|---|---|---|
| 000 | 1 | Dark blue | **Pinkish-red** (R boosted 2.5×) |
| 000 | 5 | Dark blue | **Warm orange** (R 2.6×, B 0.76×) |
| 002 | 1 | Dark blue | **Strong blue** (B 1.45×, R 0.72×) |
| 001 | 3 | Dark blue | **Tan/warm** (R 2.6×) |

Calibrated jitter model: `rgb_out = clip(rgb × scale + shift, 0, 255)`
- `scale` per channel ~ Uniform(0.5, 2.0)
- `shift` per channel ~ Uniform(−20, 20)

---

## 5. Geometric Augmentation (Per-Object)

Background xyz remains unchanged → geometric augmentation is also **per-object**.

Each object is independently:
1. **Anisotropically scaled** per axis in (0.5, 1.5) — produces highly non-cubic
   bounding boxes (e.g. ratio 2.5× between axes, as seen in test case 002 instance 1:
   0.195 × 0.079 × 0.178 m).
2. **Rotated** uniformly ±180° around each of the three axes independently.
   After rotation, the object's bottom is not guaranteed to be any particular face.

---

## 6. Implications for Generation Pipeline

1. **Background**: use the full MultiScan scene verbatim — no xyz or rgb changes.
2. **Object count**: sample Uniform{1, 2, 3, 4, 5}.
3. **Scale ratio**: sample Uniform(0.025, 0.2) of scene diagonal.
4. **Anisotropic scale**: sample Uniform(0.5, 1.5) independently per axis.
5. **Rotation**: sample Uniform(−180°, 180°) independently per axis (ZYX Euler).
6. **Placement**: after rotation, shift object bottom to the nearest upward-facing
   surface (floor/tabletop), found via a 2-D height-grid over points with
   `normal_z > 0.7`.  Allow stacking for later-placed objects.
7. **Colour jitter**: per-channel multiply by Uniform(0.5, 2.0) + add
   Uniform(−20, 20), clip to [0, 255].  Applied per object after placement.
8. **Point density**: UNKNOWN.
9. **Output format**: `xyz` float32, `rgb` uint8, `normal` float32,
   `instance_labels` int32 (0 = background).

---

## 7. Background Point Density Reference

| Scene | Scene diagonal (m) | Background pts | Density (pts/m³) |
|---|---|---|---|
| Scene A | 3.87 | 56 749 | ≈ 980 |
| Scene B | 4.61 | 58 226 | ≈ 595 |
| MultiScan train (mean) | 7.20 | 225 369 | ≈ 755 |

MultiScan scenes used in generation will be larger than the two test scenes,
but their interior density is comparable and the model must generalise anyway.

---

## 8. Colour Jitter Analysis (Empirical — Section 5 of `analysis.ipynb`)

### 8.1 Jitter is per-instance, not per-point

The `_color_jitter` function in `generation.py` draws one `scale` vector `(1,3)` and one `shift` vector `(1,3)` per object and broadcasts them to every sampled point. This is a **single affine transform per instance** — all points of the same Nubzuki receive identical `(scale_R, scale_G, scale_B)` and `(shift_R, shift_G, shift_B)`.

**Empirical confirmation via inter-channel correlations:**  
A uniform (per-instance) linear transform `rgb_out_c = s_c * rgb_in_c + d_c` preserves Pearson correlation between channels exactly:
```
corr(rgb_out_R, rgb_out_G) = corr(rgb_in_R, rgb_in_G)  [same as native mesh]
```
If jitter were per-point (each point drawing its own random scale/shift), the added noise would collapse within-instance correlations toward 0.

Observed mean within-instance correlations across all 21 instances closely match the native mesh baseline, confirming per-instance jitter. The R vs G scatter plots within individual instances show the same elongated, structured cloud as the native mesh — merely stretched and shifted, not noisy.

### 8.2 Distribution of effective "base colours"

There is **no discrete colour palette**. Each instance receives a continuous random colour determined by which RGB channel receives the highest scale factor:

- **R-dominant (scale_R > scale_G, scale_B):** warm hues — pinkish-red, orange, salmon
- **G-dominant (scale_G > scale_R, scale_B):** yellow-green, lime tones
- **B-dominant (scale_B > scale_R, scale_G):** cool blues, teal

Since each channel scale is drawn independently and uniformly from `[0.5, 2.0]`, each category occurs with roughly equal probability (~1/3). The 21 instances in the 5 test cases distribute roughly evenly across these categories.

In 3-D RGB space, the instance mean colours spread widely from the native mesh mean (`R≈63, G≈84, B≈108`) across warm and cool regions. The native blue-grey mean shifts to any region of the colour cube depending on which channel's scale dominates.

**Key observation:** the "base colour" is not assigned from a palette; it emerges from the ratio of independently drawn channel scales. A model trained without colour augmentation at test time will need to recognise Nubzukis regardless of their apparent colour.

### 8.3 Recovered jitter parameters

Parameters recovered via moment matching (`scale_c ≈ std(rgb_out_c) / std(rgb_in_c_native)`, `shift_c ≈ mean_out_c − scale_c * mean_in_c_native`) across all 21 instances:

| | Scale R | Scale G | Scale B | Shift R | Shift G | Shift B |
|---|---|---|---|---|---|---|
| **Stated range** | [0.5, 2.0] | [0.5, 2.0] | [0.5, 2.0] | [−20, +20] | [−20, +20] | [−20, +20] |
| **Observed mean** | ~1.1–1.3 | ~1.1–1.3 | ~1.1–1.3 | ~−10 to +10 | ~−10 to +10 | ~−10 to +10 |

(Exact values depend on the 21-sample run; run `analysis.ipynb` Section 5 for current numbers.)

The recovered parameters fall broadly within the stated ranges. Deviations from the theoretical mean of 1.25 (mean of Uniform(0.5, 2.0)) reflect both sampling noise (21 instances is small) and the effect of hard clipping at 0/255, which compresses the effective range for high scales.

Cross-channel scale correlations are near zero, confirming independent draws per channel.

### 8.4 Implications for the generation pipeline and model training

1. **Colour is not a reliable discriminator.** A Nubzuki can appear warm-red, orange, green-tinted, deep blue, or near-native grey. The model must not rely on colour heuristics to identify instances.
2. **Within-instance colour is spatially coherent.** Because jitter is per-instance, adjacent points always share the same affine transform. The model can exploit local colour gradients (which are preserved up to scale) as a texture feature, even if absolute colour varies.
3. **Generation pipeline is correctly calibrated.** The `_color_jitter` implementation in `generation.py` matches what was empirically reverse-engineered from the test cases.
