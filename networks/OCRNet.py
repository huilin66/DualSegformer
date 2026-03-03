import torch
import torch.nn as nn
import torch.nn.functional as F
import segmentation_models_pytorch as smp
from segmentation_models_pytorch.base import SegmentationModel

# ==========================================
# Part 1: OCR 核心模块 (Attention Logic)
# ==========================================

class SpatialGatherModule(nn.Module):
    """根据粗糙分割结果(Soft Mask)聚合特征，得到 Object Region Representations"""
    def __init__(self, scale=1):
        super(SpatialGatherModule, self).__init__()
        self.scale = scale

    def forward(self, feats, probs):
        # feats: [B, C, H, W]
        # probs: [B, K, H, W] (K=类别数)
        batch_size, c, h, w = feats.size()
        probs = probs.view(batch_size, -1, h * w)
        feats = feats.view(batch_size, -1, h * w)
        
        # Soft-assignment via matrix multiplication
        # Result: [B, K, C] -> 每个类别的特征中心
        probs = F.softmax(self.scale * probs, dim=2) 
        ocr_context = torch.bmm(probs, feats.permute(0, 2, 1))
        
        # Output: [B, C, K] -> 转置以便后续计算
        return ocr_context.permute(0, 2, 1).unsqueeze(3)

class ObjectAttentionBlock(nn.Module):
    """计算像素特征与 Object Region 特征之间的注意力"""
    def __init__(self, in_channels, key_channels, scale=1):
        super(ObjectAttentionBlock, self).__init__()
        self.scale = scale
        self.in_channels = in_channels
        self.key_channels = key_channels

        self.f_pixel = nn.Sequential(
            nn.Conv2d(in_channels, key_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(key_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(key_channels, key_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(key_channels),
            nn.ReLU(inplace=True),
        )
        
        self.f_object = nn.Sequential(
            nn.Conv2d(in_channels, key_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(key_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(key_channels, key_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(key_channels),
            nn.ReLU(inplace=True),
        )
        
        self.f_down = nn.Sequential(
            nn.Conv2d(in_channels, key_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(key_channels),
            nn.ReLU(inplace=True),
        )

        self.f_up = nn.Sequential(
            nn.Conv2d(key_channels, in_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x, proxy):
        batch_size, h, w = x.size(0), x.size(2), x.size(3)
        
        # 1. 变换空间 query (pixels)
        query = self.f_pixel(x).view(batch_size, self.key_channels, -1)
        query = query.permute(0, 2, 1) # [B, HW, KeyC]
        
        # 2. 变换物体 key (object regions)
        key = self.f_object(proxy).view(batch_size, self.key_channels, -1) # [B, KeyC, K]
        
        # 3. 计算注意力
        value = self.f_down(proxy).view(batch_size, self.key_channels, -1) # [B, KeyC, K]
        value = value.permute(0, 2, 1) # [B, K, KeyC]

        sim_map = torch.matmul(query, key) # [B, HW, K]
        sim_map = (self.key_channels ** -0.5) * sim_map
        sim_map = F.softmax(sim_map, dim=-1) # 每个像素对 K 个类别的关注度
        
        # 4. 聚合上下文
        context = torch.matmul(sim_map, value) # [B, HW, KeyC]
        context = context.permute(0, 2, 1).contiguous()
        context = context.view(batch_size, self.key_channels, h, w)
        
        # 5. 映射回原始维度并残差连接
        context = self.f_up(context)
        return x + context

class OCRHead(nn.Module):
    def __init__(self, in_channels, num_classes, ocr_channels=256):
        super(OCRHead, self).__init__()
        
        # 1. Auxiliary Head (粗糙分割)
        self.aux_head = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, num_classes, 1)
        )
        
        # 2. OCR 模块组件
        self.spatial_gather = SpatialGatherModule()
        self.object_attention = ObjectAttentionBlock(in_channels, ocr_channels)
        
        # 3. 最终分类头
        self.cls_head = nn.Conv2d(in_channels, num_classes, 1)

    def forward(self, x):
        # x: HRNet 聚合后的高分辨率特征 [B, C, H, W]
        
        # Step 1: 生成粗糙分割 (Auxiliary Output)
        # 用作 Soft Mask 来提取物体区域
        aux_out = self.aux_head(x) 
        
        # Step 2: 提取 Object Region Features
        # 输入: 像素特征 x, 粗糙预测 aux_out
        proxy_feats = self.spatial_gather(x, aux_out)
        
        # Step 3: OCR Attention
        # 增强像素特征
        ocr_feats = self.object_attention(x, proxy_feats)
        
        # Step 4: 最终分类
        final_out = self.cls_head(ocr_feats)
        
        return final_out, aux_out

# ==========================================
# Part 2: 组装 OCRNet (SMP HRNet + OCRHead)
# ==========================================

class SMP_OCRNet(nn.Module):
    def __init__(self, encoder_name="tu-hrnet_w18", in_channels=3, classes=2, ocr_mid_channels=256):
        super().__init__()
        
        # 1. 使用 SMP 加载 Encoder (timm backend)
        # 注意: depth=4 会返回 HRNet 的4个阶段特征
        self.encoder = smp.encoders.get_encoder(
            encoder_name, 
            in_channels=in_channels, 
            depth=4,
            weights="imagenet"
        )
        
        # 2. 自动计算 HRNet 输出的总通道数
        # HRNet 输出是 4 个不同尺度的特征图，OCRNet 做法是把它们 Upsample 到同一尺寸后 Concat
        # 例如 w18: [18, 36, 72, 144] -> sum = 270
        self.encoder_out_channels = sum(self.encoder.out_channels)
        
        # 3. OCR Head
        self.head = OCRHead(in_channels=self.encoder_out_channels, num_classes=classes, ocr_channels=ocr_mid_channels)

    def forward(self, x):
        input_shape = x.shape[-2:]
        
        # 1. Encoder Forward
        features = self.encoder(x) 
        # features 是一个列表，包含不同 stride 的特征:
        # F0 (s=4), F1 (s=8), F2 (s=16), F3 (s=32)
        
        # 2. HRNet Feature Aggregation (Upsample + Concat)
        # 将所有特征图上采样到 F0 (stride=4) 的尺寸
        target_h, target_w = features[0].shape[2], features[0].shape[3]
        
        upsampled_feats = [features[0]]
        for i in range(1, len(features)):
            upsampled_feats.append(
                F.interpolate(features[i], size=(target_h, target_w), mode='bilinear', align_corners=True)
            )
            
        # 拼接 -> 得到高分辨率强语义特征
        feats = torch.cat(upsampled_feats, dim=1)
        
        # 3. OCR Head Forward
        final_logits, aux_logits = self.head(feats)
        
        # 4. Upsample to original input size (128x128)
        final_logits = F.interpolate(final_logits, size=input_shape, mode='bilinear', align_corners=True)
        aux_logits = F.interpolate(aux_logits, size=input_shape, mode='bilinear', align_corners=True)
        
        # 训练时通常需要 Aux Loss，但在 Inference 时只需要 final
        if self.training:
            return final_logits, aux_logits
        else:
            return final_logits