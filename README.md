# Segformer for Mars Landslide Segmentation Challenge

This repository is built based on [M3LSNet](https://github.com/tingul4/M3LSNet), and contains the solution for the **1st Mars Landslide Segmentation (Mars-LS) Challenge**. It implements Segformer with ConvNext backbonefor the challenges of Martian terrain segmentation.

## 📂 Repository Structure

```
SegformerConvNext/
├── networks/              # Model Definition
├── losses/                # Loss Functions Definition
├── dataset.py             # 7-Channel Tiff Loader with Smart Normalization
├── train.py               # Main Training Loop (A100 Optimized)
├── generate_submission.py # Inference & Submission Zipping
├── augmentations.py       # Online Data Augmentation
├── utils_loss.py          # Combined Loss Implementation
└── outputs/               # Structured Logs & Checkpoints
    └── YYYYMMDD_HHMMSS/
        ├── checkpoints/   # best.pth, last.pth
        ├── logs/          # Training logs
        └── tensorboard/   # Visualization
```

---

## 🛠️ Usage

### 1. Requirements
Ensure you have the following installed:
- PyTorch 2.x (CUDA recommended)
- `tifffile` (for 7-channel images)
- `mamba_ssm` (Optional, for acceleration)
- `tensorboard`, `tqdm`

### 2. Training
To start training with the optimized settings (Batch Size 32, Combined Loss):

```bash
python train.py
```
*Outputs will be saved to `outputs/CurrentTimestamp/`.*

### 3. Monitoring
Track training progress:
```bash
tensorboard --logdir outputs/
```

### 4. Generating Submission
To generate the `submission.zip` for the challenge leaderboard:

```bash
# Uses the 'best.pth' from the specified run
python generate_submission.py --checkpoint outputs/YOUR_RUN_DIR/checkpoints/best.pth
```
or use the helper script:
```bash
sh generate_submission.sh
```

---

## 📊 Performance Notes

- **Validation mIoU**: May reach high values (e.g., >0.90) due to spatial correlation in the dataset.
- **Test Generalization**: The **Per-Image Normalization** and **Augmentation** strategies are specifically designed to maximize performance on the hidden Test Set by forcing the model to learn invariant physical features.
