import torch
import torch.nn as nn
import torch.nn.functional as F

class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, pred, target):
        # pred: [B, C, H, W] (raw logits)
        # target: [B, H, W] (long)
        
        # Apply Softmax to get probabilities
        probs = torch.softmax(pred, dim=1)
        
        # Get Foreground probabilities (Class 1)
        # Assuming binary segmentation: Class 0=BG, Class 1=FG
        pred_fg = probs[:, 1]
        
        # Create one-hot like mask for target or just use where target==1
        target_fg = (target == 1).float()
        
        # Flatten for calculation
        pred_flat = pred_fg.contiguous().view(-1)
        target_flat = target_fg.contiguous().view(-1)
        
        intersection = (pred_flat * target_flat).sum()
        union = pred_flat.sum() + target_flat.sum()
        
        dice = (2. * intersection + self.smooth) / (union + self.smooth)
        
        return 1 - dice

class CombinedLoss(nn.Module):
    def __init__(self, ce_weight=0.5, dice_weight=0.5):
        super(CombinedLoss, self).__init__()
        self.ce = nn.CrossEntropyLoss()
        self.dice = DiceLoss()
        self.w_ce = ce_weight
        self.w_dice = dice_weight
        
    def forward(self, pred, target):
        loss_ce = self.ce(pred, target)
        loss_dice = self.dice(pred, target)
        return self.w_ce * loss_ce + self.w_dice * loss_dice
