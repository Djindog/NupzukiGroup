

## 3D Point Cloud Segmentation Challenge

**Mid-Term Evaluation Submission Due**: April 30 (Thursday), 23:59 KST  
**Final Submission Due**: May 9 (Saturday), 23:59 KST  
**Where to submit**: KLMS  

![Dataset](assets/figures/nubjuki.png)

## What to Do — Find Nubzukis!

In this competition, your mission is to detect and segment **Nubzukis** in 3D scenes. You'll need to train a 3D segmentation neural network that takes a point cloud with color information as input and predicts indices of points for one or multiple instances of Nubzukis. It is a **single-category 3D point cloud instance segmentation** task.

For each scene point cloud, predict a point-wise instance label:

$$
\hat y_i \in \{0,1,2,\dots\}
$$

- `0`: background
- `1,2,...`: predicted object instances (the order of instance IDs is arbitrary)

Ground truth is also provided as point-wise instance labels with the same convention.

This means:
- the key task is to correctly separate background and multiple target instances,
- matching between predictions and ground truth is permutation-invariant (handled in the evaluator).

We provide the dataset format and a fixed evaluator. Your job is to improve the model in `model.py`.

**Important Notes**

PLEASE READ THE FOLLOWING CAREFULLY! Any violation of the rules, or failure to properly cite any existing code, models, or papers used in the project in your write-up, will result in a zero score.

### What You CANNOT Do

- ❌ **DO NOT** use any pretrained network.
- ❌ **DO NOT** exceed the total model parameter limit (50M).
- ❌ **DO NOT** use any extra dataset for training other than the provided training split (`train`, `val`) and reference objects (`sample.glb`).
- ❌ **DO NOT** modify the provided `evaluate.py` in the official submission.
- ❌ **DO NOT** use any CUDA version other than the provided one (default: 12.4).
- ❌ **DO NOT** exceed the main inference loop time limit (300 seconds).
- ❌ **DO NOT** exceed the VRAM limit (24GB).

### What You CAN Do

- ✅ **Modify `model.py`** to implement your own model.
- ✅ **Implement your own dataset loader** based on the MultiScan dataset and the provided reference objects.
- ✅ **Implement your own training pipeline** to train your model.
- ✅ **Create new files**: Add any additional implementation files you need
- ✅ **Use open-source implementations**: As long as they are clearly mentioned and cited in your write-up
- ✅ **Use extra libraries**: If your implementation requires an additional library, please upload to the Slack channel with the library name and a brief justification. The TAs will review each request and approve or reject it. Only approved libraries may be used.

## Environment Setup

```shell
conda create -n 3d-seg python=3.10 -y
conda activate 3d-seg

# Install torch torch==2.7.1
pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu126

# Install other packages
pip install numpy scipy tqdm matplotlib
```

### Allowed Python Packages

```shell
ninja, trimesh, pytorch-cluster, pytorch-scatter, pytorch-sparse, spconv, scikit-learn, pointcept
```

**Note:** We will add packages upon request. Once a package is requested, it will be reviewed, and only approved packages will be allowed for use.

## Challenge Structure

```
CS479-Assignment-3DPC-Segmentation-Challenge/
├── assets/
│   ├── sample.glb         # Object mesh data
│   └── test_0000.npy      # Test data example
│
├── dataset.py             # Code for loading test dataset (provided, DO NOT MODIFY)
├── evaluate.py            # Evaluation script (provided, DO NOT MODIFY)
├── model.py               # Model definition (students SHOULD modify)
└── visualize.py           # Visualization script (provided, CAN modify)
```

**Legend:**
- **DO NOT MODIFY**: Keep these files as-is (for fair comparison)
- **SHOULD modify**: Main files where you implement your solution
- **CAN modify**: Optional modifications to improve your model

**Model interface constraint (must keep):**
- `initialize_model(ckpt_path, device, ...) -> your_model: torch.nn.Module`: load model from checkpoint path
- `run_inference(your_model, features, ...) -> [B, N]`: return point-wise instance labels

You may freely change the model architecture, training strategy, and `model.py`, as long as the interface above remains compatible. You can also generate additional codes, but note that your code will be evaluated using the provided evaluator. The evaluation will be conducted using the provided `evaluate.py` without modification.

## MultiScan Dataset

You are required to use the **MultiScan benchmark dataset** for training. We will use Object Instance Segmentation benchmark dataset.

**Please note that downloading the dataset may take some time, so we recommend preparing it as early as possible.**

We provide Google Drive link with reference 3D objects in `.glb` format:
[Link]({https://drive.google.com/drive/folders/1guo68JlVkeqAzX7nR3DOfOB6SfhL3XC9?usp=sharing})


You can also download the MultiScan benchmark dataset by following the instructions in the original GitHub repository:
[MultiScan Dataset README](https://github.com/smartscenes/multiscan/blob/main/dataset/README.md)


## Generation Pipeline for Test Data

Each test scene will be generated using the following procedure. For each scene:
- a random number of objects is inserted (`min=1`, `max=5`)
- mesh placement is attempted with multiple scale ratios (range: `0.025` to `0.2` of the scene diagonal)
- an object may be placed on top of another object, with partial overhang allowed

Once the object is placed, the point cloud is extracted with the following augmentations:
- anisotropic scaling: each of the x, y, and z axes is independently scaled within the range `(0.5, 1.5)`
- affine transform: rotation around the x,y,z-axis in the range `(-180, 180)`
- color map jittering

Note that simulating similar object insertion and augmentation procedures during training is allowed.

## Test Data Format

Each generated scenes are saved as `.npy` files with the following format:

Saved dictionary keys:
- `xyz`: `float32`, shape `(N, 3)`
- `rgb`: `uint8`, shape `(N, 3)`
- `normal`: `float32`, shape `(N, 3)`
- `instance_labels`: `int32`, shape `(N,)` (`0` for background, positive IDs for inserted instances)

We will also provide example test data. 

## Evaluation

Instance segmentation results are evaluated on the generated test set using two metrics: `F1@0.25`($F1_{0.25}$) and `F1@0.50`($F1_{0.50}$). 

For each scene, we first convert the predicted instance labels and ground-truth instance labels into binary instance masks, one mask per object instance. We then compute the pairwise IoU matrix between all predicted and ground-truth instances.

Using this IoU matrix, we perform Hungarian matching (1-to-1 assignment) with cost `1 - IoU`. For a given IoU threshold $\tau$, we define:

- $TP_\tau$: matched pairs with $IoU \ge \tau$
- $FP_\tau$: predicted instances not counted as TP
- $FN_\tau$: GT instances not counted as TP

For each threshold, TP, FP, and FN are aggregated over all scenes. Then, the F1 score is computed for all scenes as follows:

```math
\text{F1}_{\tau} = \frac{2 \text{TP}_{\tau}}{2 \text{TP}_{\tau} + \text{FP}_{\tau} + \text{FN}_{\tau}}
```

The F1 scores with two different thresholds, 0.25 and 0.50, must be reported: `F1@0.25`($F1_{0.25}$) and `F1@0.50`($F1_{0.50}$).

**Note**: Predicted instance IDs are valid only in `1..100`. Any predicted ID greater than `100` is remapped to background (`0`) before scoring.

More details about the evaluation metrics are provided in `evaluate.py`.

- Additionally, to help everyone gauge progress, there will be a [Mid-Term Evaluation](#mid-term-evaluation-submission-optional) where teams can submit intermediate results.
- **Final grading will be determined relative to the best score achieved for each metric (`F1@0.25` and `F1@0.50`).** Specifically, the score for each metric is computed as follows:

  ```math
  \mathrm{Score} = \max\left(\cfrac{\mathrm{Your\,Score}}{\mathrm{Highest\,Score}} \times 8, 0\right)
  ```

- If your score equals the highest score, you receive 8 points for that task.

- Bonus credits for each task:
  - **Mid-Term Evaluation Bonus**: Top-k teams for each metric will receive +1.0 point toward the final grade.
  - **Winner Bonus**: If your team achieves the highest score for a task, you receive +1.0 point for that task.

- In total, the 3D Point Cloud Segmentation Challenge is worth a maximum of 20 points.

## Mid-Term Evaluation Submission (Optional)

The purpose of the mid-term evaluation is to help teams gauge their progress. Participation is optional, but the top-k teams for each metric will receive bonus credit toward the final grade.


- **What to submit**
  1. **Self-contained source code**
      - Your submission must include the complete codebase necessary to run end-to-end in the TA environment.
      - The TAs will run your code in their environment without additional modifications.
      - For consistent evaluation, `evaluate.py` will be replaced with the official version.
  2. **A model checkpoint** (and an optional config file)

- **Grading Procedure**
  - The TAs will run your submitted code in their Python environment.
  - The scores measured by the TAs will be published on the leaderboard.
  - Submissions that fail to run in the TA environment will be marked as failed on the leaderboard.
  - Among the submissions that outperform the TAs’ scores, the top-k teams will receive bonus credit.

## Final Submission

- **What to submit**:
  1. **Self-contained source code**
  2. **A model checkpoint** (and an optional config file)
  3. **A 2-page write-up**
      - No template is provided.
      - Maximum of two A4 pages, excluding references.
      - All of the following must be included:
          - **Technical details**: A one-paragraph description of your implementation, including the architecture design, hyperparameters, and other relevant details.
          - **Training details**: Training logs (e.g., training loss curves) and the total training time.
          - **Qualitative evidence**: Approximately four rendered sample images with segmentation results.
          - **Citations**: All external code and papers used must be properly cited.
      - Missing any of these items will result in a penalty.
      - If the write-up exceeds two pages, any content beyond the second page will be ignored, which may result in missing required items.

## Grading

**There are no late days. Submit on time.**  

**Late submission**: Zero score.  
**Missing any required item in the final submission (qualitative results, code/checkpoint, or write-up)**: Zero score.  
**Missing items in the write-up**: 10% penalty for each. 

## Recommended Readings
[1] [Mao et al., MultiScan: Scalable RGBD scanning for 3D environments with articulated objects, NeurIPS 2022.](https://proceedings.neurips.cc/paper_files/paper/2022/hash/3b3a83a5d86e1d424daefed43d998079-Abstract-Conference.html) [[Github]](https://github.com/smartscenes/multiscan) [[Benchmark Docs]](https://3dlg-hcvc.github.io/multiscan/read-the-docs/benchmark/dataset.html#object-instance-segmentation)  
[2] [Jiang et al., PointGroup: Dual-Set Point Grouping for 3D Instance Segmentation, CVPR 2020.](https://arxiv.org/abs/2004.01658)  
[3] [Liang et al., Instance Segmentation in 3D Scenes using Semantic Superpoint Tree Networks, ICCV 2021.](https://arxiv.org/abs/2108.07478)  
[4] [Chen et al., Hierarchical Aggregation for 3D Instance Segmentation, ICCV 2021.](https://arxiv.org/abs/2108.02350)

<br />

[Back to top](#)
<br /># NupzukiGroup
