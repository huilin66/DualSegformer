import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import logging
from datetime import datetime
from tqdm import tqdm
import numpy as np

from dataset import MarsSegDataset, MosaicCastDataset
from networks import get_model
from losses import UnetFormerLoss
import random


RANDOM_SEED = 42


def set_seed(seed=42):
    # 1. 设置 Python 环境变量
    os.environ['PYTHONHASHSEED'] = str(seed)
    # 对于 PyTorch 1.8+，强制 cuBLAS 使用确定性算法
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8' 
    
    # 2. 设置 Python 内置随机种子
    random.seed(seed)
    
    # 3. 设置 Numpy 随机种子
    np.random.seed(seed)
    
    # 4. 设置 PyTorch 随机种子
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # 如果使用多GPU
    
    # 5. 配置 cuDNN
    # 禁用 cuDNN 自动寻找最快算法的机制（因为最快算法往往有随机性）
    torch.backends.cudnn.benchmark = False 
    # 强制 cuDNN 使用确定性算法
    torch.backends.cudnn.deterministic = True 
    
    # 6. 强制 PyTorch 使用确定性算法 (可选，可能会导致程序报错如果使用了不支持确定性的操作)
    # torch.use_deterministic_algorithms(True)
    torch.use_deterministic_algorithms(True, warn_only=True)


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
# ==========================================
# Utils
# ==========================================
def setup_logger(save_dir):
    log_format = '%(asctime)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_format)
    logger = logging.getLogger()
    
    # File handler
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = os.path.join(save_dir, f'train_log_{timestamp}.txt')
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(logging.Formatter(log_format))
    logger.addHandler(file_handler)
    return logger, timestamp

def calculate_metrics(preds, targets):
    """
    Calculate metrics for binary segmentation (Class 1: Landslide, Class 0: Background)
    preds: [B, H, W] (0 or 1)
    targets: [B, H, W] (0 or 1)
    """
    preds = preds.view(-1)
    targets = targets.view(-1)
    
    tp = (preds * targets).sum().float()
    fp = ((preds == 1) & (targets == 0)).sum().float()
    fn = ((preds == 0) & (targets == 1)).sum().float()
    tn = ((preds == 0) & (targets == 0)).sum().float()
    
    # Precision, Recall, F1 for Foreground
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    
    # IoU
    iou_fg = tp / (tp + fp + fn + 1e-8)
    iou_bg = tn / (tn + fp + fn + 1e-8)
    miou = (iou_fg + iou_bg) / 2
    
    return {
        'miou': miou.item(),
        'precision': precision.item(),
        'recall': recall.item(),
        'f1': f1.item(),
        'iou_fg': iou_fg.item(),
        'iou_bg': iou_bg.item()
    }

# ==========================================
# Part E: Training Loop
# 參數參照論文 Section III.B [cite: 165-172]
# ==========================================
def train_pipeline(model_name, conduct_val=False):
    # Seed setting
    set_seed(RANDOM_SEED)
    # 1. Hyperparameters
    if model_name.startswith('dual_'):
        BATCH_SIZE = 16
    elif model_name.startswith('ocrnet'):
        BATCH_SIZE = 8
    elif model_name.startswith('mask2former'):
        BATCH_SIZE = 8
    elif model_name.startswith('oneformer'):
        BATCH_SIZE = 8
    else:
        BATCH_SIZE = 32
    LR = 1e-4                 # [cite: 167]
    WEIGHT_DECAY = 5e-4       # [cite: 171]
    EPOCHS = 100
    VAL_INTERVAL = 1
    IN_CHANNELS = 7

    # Paths
    DATASET_ROOT = '/scrinvme/huilin/bdd/cp_data/mars_seg/Mars_LSc_2025_dataset_1st_phase_updateB2'
    
    # Output Directory Structure: outputs/YYYYMMDD_HHMMSS/{checkpoints, logs, tensorboard}
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    EXPERIMENT_DIR = os.path.join('outputs', timestamp)
    
    CHECKPOINT_DIR = os.path.join(EXPERIMENT_DIR, 'checkpoints')
    LOG_DIR = os.path.join(EXPERIMENT_DIR, 'logs')
    TB_DIR = os.path.join(EXPERIMENT_DIR, 'tensorboard')
    
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(TB_DIR, exist_ok=True)

    # Logging & Tensorboard
    logger, _ = setup_logger(LOG_DIR)
    writer = SummaryWriter(log_dir=TB_DIR)
    logger.info(f"Model: {model_name}")
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Experiment started at {timestamp}")
    logger.info(f"Outputs will be saved to: {EXPERIMENT_DIR}")
    logger.info(f"Using device: {device}")

    # 2. Model & Data
    model = get_model(model_name, in_channels=IN_CHANNELS, num_classes=2).to(device)
    
    train_dataset_raw = MarsSegDataset(root_dir=DATASET_ROOT, split='train')
    train_dataset = MosaicCastDataset(train_dataset_raw)
    val_dataset = MarsSegDataset(root_dir=DATASET_ROOT, split='val')
    
    gt = torch.Generator()
    gt.manual_seed(RANDOM_SEED)
    gv = torch.Generator()
    gv.manual_seed(RANDOM_SEED)

    if len(train_dataset) == 0:
        logger.warning("Dataset not found. Creating dummy data.")
        # Mocking for test
        train_loader = []
        val_loader = []
    else:
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True, drop_last=True,
        worker_init_fn=seed_worker, generator=gt)
        # Validate logic for val_dataset
        if len(val_dataset) > 0:
            val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, 
            worker_init_fn=seed_worker, generator=gv)
        else:
            val_loader = None
            logger.warning("Validation set is empty.")

    # 3. Optimizer & Scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY, betas=(0.9, 0.999))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    # criterion = CombinedLoss(ce_weight=0.5, dice_weight=0.5)
    criterion = UnetFormerLoss()

    best_miou = 0.0
    
    # 4. Loop
    logger.info("Start Training...")
    
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0
        
        # Progress Bar for Training
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Train]", unit="batch")
        
        for batch_idx, (data, target) in enumerate(pbar):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            
            # Loss
            if isinstance(output, tuple):
                final_output, aux_output = output
                loss_main = criterion(final_output, target)
                loss_aux = criterion(aux_output, target)
                
                loss = loss_main + 0.4 * loss_aux
                
                output = final_output
                
            else:
                loss = criterion(output, target)
            
            loss.backward()
            
            # Gradient Clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            loss_val = loss.item()
            train_loss += loss_val
            
            # Update pbar
            pbar.set_postfix({'loss': f"{loss_val:.4f}"})
            
            # Log step loss
            writer.add_scalar('Train/Batch_Loss', loss_val, epoch * len(train_loader) + batch_idx)
            
        avg_train_loss = train_loss / len(train_loader) if len(train_loader) > 0 else 0
        logger.info(f"Epoch [{epoch+1}/{EPOCHS}] Train Loss: {avg_train_loss:.4f}")
        writer.add_scalar('Train/Epoch_Loss', avg_train_loss, epoch)
        
        # ==========================================
        # Validation
        # ==========================================
        if conduct_val and val_loader and (epoch + 1) % VAL_INTERVAL == 0:
            model.eval()
            val_loss = 0
            
            # Metrics accumulators
            total_matches = {'miou': [], 'precision': [], 'recall': [], 'f1': [], 'iou_fg': [], 'iou_bg': []}
            
            with torch.no_grad():
                pbar_val = tqdm(val_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Val]", unit="batch", leave=False)
                for val_data, val_target in pbar_val:
                    val_data, val_target = val_data.to(device), val_target.to(device)
                    
                    output = model(val_data)
                    
                    # 核心修复：安全处理 UnetFormerLoss 与不同模型输出的兼容性
                    if isinstance(output, tuple):
                        final_output, aux_output = output
                        # 如果损失函数是 UnetFormerLoss，直接调用其内部的 main_loss 和 aux_loss，防止底层拆包崩溃
                        if hasattr(criterion, 'main_loss') and hasattr(criterion, 'aux_loss'):
                            loss_main = criterion.main_loss(final_output, val_target)
                            loss_aux = criterion.aux_loss(aux_output, val_target)
                        else:
                            loss_main = criterion(final_output, val_target)
                            loss_aux = criterion(aux_output, val_target)
                            
                        loss = loss_main + 0.4 * loss_aux
                        output = final_output
                        
                    else:
                        # 针对 SegFormer 等只输出单个 Tensor 的模型
                        if hasattr(criterion, 'main_loss'):
                            # 绕过 UnetFormerLoss 的 forward，直接用 main_loss 计算，杜绝维度拆包错误！
                            loss = criterion.main_loss(output, val_target)
                        else:
                            loss = criterion(output, val_target)

                    val_loss += loss.item()
                    
                    # Get predictions
                    preds = torch.argmax(output, dim=1) # [B, H, W]
                    
                    # Calculate metrics for batch
                    metrics = calculate_metrics(preds, val_target)
                    for k, v in metrics.items():
                        total_matches[k].append(v)
            
            # 平均 Validation Loss
            avg_val_loss = val_loss / len(val_loader)
            
            # Aggregate metrics
            epoch_metrics = {k: np.mean(v) for k, v in total_matches.items()}
            
            logger.info(f"Epoch [{epoch+1}/{EPOCHS}] Val Loss: {avg_val_loss:.4f} | "
                        f"mIoU: {epoch_metrics['miou']:.4f} | "
                        f"F1: {epoch_metrics['f1']:.4f} | "
                        f"IoU (FG): {epoch_metrics['iou_fg']:.4f} | "
                        f"IoU (BG): {epoch_metrics['iou_bg']:.4f}")
            
            # Tensorboard
            writer.add_scalar('Val/Loss', avg_val_loss, epoch)
            writer.add_scalar('Val/mIoU', epoch_metrics['miou'], epoch)
            writer.add_scalar('Val/F1', epoch_metrics['f1'], epoch)
            writer.add_scalar('Val/Recall', epoch_metrics['recall'], epoch)
            writer.add_scalar('Val/Precision', epoch_metrics['precision'], epoch)
            writer.add_scalar('Val/IoU_FG', epoch_metrics['iou_fg'], epoch)
            writer.add_scalar('Val/IoU_BG', epoch_metrics['iou_bg'], epoch)
            
            # Checkpoint - Best Model
            if epoch_metrics['miou'] > best_miou:
                best_miou = epoch_metrics['miou']
                torch.save(model.state_dict(), os.path.join(CHECKPOINT_DIR, 'best.pth'))
                logger.info(f"New Best Model Saved! (mIoU: {best_miou:.4f})")
        scheduler.step()
        
        # Checkpoint - Last Model
        torch.save(model.state_dict(), os.path.join(CHECKPOINT_DIR, 'last.pth'))
        
    writer.close()
    logger.info("Training Completed.")

if __name__ == '__main__':
    model_names = [
            'dual_segformer_convnexttiny_chv1_add',
            'dual_segformer_convnextlarge_chv1_add',
        ]
    for model_name in model_names:
        train_pipeline(model_name, conduct_val=False)
