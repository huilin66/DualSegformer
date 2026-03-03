# Segformer for Mars Landslide Segmentation Challenge

This repository is built based on [M3LSNet](https://github.com/tingul4/M3LSNet), and contains the solution for the **1st Mars Landslide Segmentation (Mars-LS) Challenge**. It implements Dual_Segformer with ConvNext backbonefor the challenges of Martian terrain segmentation.

## 📂 Repository Structure

```
DualSegformerConvNext/
├── losses/                         # Loss Functions Definition
├── networks/                       # Model Definition
├── augmentations.py                # Online Data Augmentation
├── data_prepare.py                 # Data Preparation
├── dataset.py                      # Dataset Definition
├── generate_submission_aug.py      # Inference & Submission Zipping
├── generate_submission_voting.py   # Inference & Submission Zipping
├── generate_submission.py          # Inference & Submission Zipping
├── log_write.py                    # Training Log Writing
├── result_ana.py                   # Result Visualization and Analysis
├── train.py                        # Main Training Loop (A100 Optimized)
└── outputs/                        # Structured Logs & Checkpoints
    └── YYYYMMDD_HHMMSS/
        ├── checkpoints/            # best.pth, last.pth
        ├── logs/                   # Training logs
        └── tensorboard/            # Visualization   
```

---

## 🛠️ Usage

### 1. Requirements
Ensure you have the following installed:
- PyTorch 2.x (CUDA recommended)
- `tifffile` (for 7-channel images)
- `tensorboard`, `tqdm`

### 2. Training
To start training with the optimized settings (Batch Size 32, Combined Loss):

```bash
python train.py
```
*Outputs will be saved to `outputs/CurrentTimestamp/`.*

### 3. Generating Submission
To generate the `submission.zip` for the challenge leaderboard:

```bash
# Uses the 'best.pth' from the specified run
python generate_submission.py
```
or:
```bash
python generate_submission_aug.py
```
or:
```bash
python generate_submission_voting.py
```
---

