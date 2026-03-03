import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Mask2FormerForUniversalSegmentation

class HF_Mask2Former(nn.Module):
    def __init__(self, in_channels=7, classes=2, backbone="swin-tiny"):
        super().__init__()
        
        # 1. 选择配置
        if "swin" in backbone:
            model_repo = "facebook/mask2former-swin-tiny-cityscapes-semantic"
        else:
            model_repo = "facebook/mask2former-swin-tiny-cityscapes-semantic"

        # 2. 加载模型
        self.model = Mask2FormerForUniversalSegmentation.from_pretrained(
            model_repo,
            num_labels=classes,
            ignore_mismatched_sizes=True
        )
        
        # 3. 修改第一层卷积适配 7 通道 (保持你之前的逻辑)
        patch_embed = self.model.model.pixel_level_module.encoder.embeddings.patch_embeddings.projection
        if patch_embed.in_channels != in_channels:
            new_conv = nn.Conv2d(
                in_channels=in_channels,
                out_channels=patch_embed.out_channels,
                kernel_size=patch_embed.kernel_size,
                stride=patch_embed.stride,
                padding=patch_embed.padding
            )
            with torch.no_grad():
                new_conv.weight[:, :3, :, :] = patch_embed.weight
                new_conv.weight[:, 3:6, :, :] = patch_embed.weight
                new_conv.weight[:, 6:7, :, :] = patch_embed.weight[:, 0:1, :, :]
                if patch_embed.bias is not None:
                    new_conv.bias = patch_embed.bias
            self.model.model.pixel_level_module.encoder.embeddings.patch_embeddings.projection = new_conv

    def forward(self, x):
        # x shape: [Batch, 7, 128, 128]
        
        # 1. 前向传播
        outputs = self.model(pixel_values=x)
        
        # 2. 提取 Raw Outputs (修复 AttributeError)
        # class_queries_logits: [Batch, Num_Queries, Num_Classes+1] (最后一个是 'no object')
        # masks_queries_logits: [Batch, Num_Queries, H/4, W/4]
        class_logits = outputs.class_queries_logits
        mask_logits = outputs.masks_queries_logits
        
        # 3. 【关键步骤】将 Mask Classification 转换为 Semantic Segmentation map
        # 这是一个手动解码过程，为了适配你的 CrossEntropyLoss
        
        # A. 移除 "No Object" 类 (最后一类)
        # shape 变为 [B, Q, Num_Classes]
        class_prob = class_logits[..., :-1].softmax(dim=-1)
        
        # B. Mask 激活 (Sigmoid)
        # shape: [B, Q, H/4, W/4]
        mask_prob = mask_logits.sigmoid()
        
        # C. 矩阵乘法融合: Semantic Map = sum(Class_Prob * Mask_Prob)
        # Einsum 公式: "bqc, bqhw -> bchw"
        # b: batch, q: query, c: class, h: height, w: width
        sem_seg = torch.einsum("bqc, bqhw -> bchw", class_prob, mask_prob)
        
        # 4. 上采样回原始尺寸 (128x128)
        # 因为 Mask2Former 内部输出是 1/4 分辨率
        sem_seg = F.interpolate(
            sem_seg, 
            size=x.shape[-2:], 
            mode="bilinear", 
            align_corners=False
        )
        
        return sem_seg
    

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Mask2FormerForUniversalSegmentation, Mask2FormerConfig, ConvNextModel

class HF_Mask2Former_ConvNeXt(nn.Module):
    def __init__(self, in_channels=7, classes=2, backbone="convnext-tiny"):
        super().__init__()
        
        print(f"🚀 Initializing Mask2Former with Public Meta ConvNeXt Backbone...")

        # =========================================================
        # 1. 手动构建 Mask2Former 配置
        # =========================================================
        config = Mask2FormerConfig(
            num_labels=classes,
            ignore_mismatched_sizes=True,
            
            # Backbone 设置 (对应 ConvNeXt-Tiny)
            backbone_config={
                "model_type": "convnext",
                "num_channels": in_channels,          # 让模型原生初始化为 7 通道
                "depths": [3, 3, 9, 3],               
                "hidden_sizes": [96, 192, 384, 768],  
                "out_indices": [0, 1, 2, 3],          
                "drop_path_rate": 0.1,
            }
        )

        # =========================================================
        # 2. 创建模型 (此时第一层直接就是 7 通道的尺寸)
        # =========================================================
        self.model = Mask2FormerForUniversalSegmentation(config)

        # =========================================================
        # 3. 加载公开权重，并在喂给模型前「扩建」通道数
        # =========================================================
        public_convnext_repo = "facebook/convnext-tiny-224"
        print(f"📥 Loading public weights from: {public_convnext_repo}")
        
        # 拉取 3 通道的原始模型
        public_convnext = ConvNextModel.from_pretrained(
            public_convnext_repo,
            revision="refs/pr/4"
        )
        
        # 提取字典
        pretrained_state_dict = public_convnext.state_dict()
        
        # 找到冲突的 3 通道权重
        old_weight = pretrained_state_dict["embeddings.patch_embeddings.weight"]
        
        if old_weight.shape[1] != in_channels:
            print(f"🔧 Adapting checkpoint weights from {old_weight.shape[1]} to {in_channels} channels...")
            
            # 创建一个空的 7 通道权重张量
            new_weight = torch.zeros(
                (old_weight.shape[0], in_channels, old_weight.shape[2], old_weight.shape[3]), 
                dtype=old_weight.dtype, 
                device=old_weight.device
            )
            
            # 复制策略: 扩充到 7 通道
            with torch.no_grad():
                new_weight[:, :3, :, :] = old_weight
                new_weight[:, 3:6, :, :] = old_weight
                new_weight[:, 6:7, :, :] = old_weight[:, 0:1, :, :]
            
            # 用扩建好的 7 通道权重替换字典里原本的 3 通道权重
            pretrained_state_dict["embeddings.patch_embeddings.weight"] = new_weight

        # 现在字典和模型的形状完全匹配，安全加载！
        msg = self.model.model.pixel_level_module.encoder.load_state_dict(
            pretrained_state_dict, 
            strict=False
        )
        print(f"✅ Backbone weights loaded. Missing keys (expected for decoder/head): {len(msg.missing_keys)}")
        
        # 注意：原代码中的“4. 修改第一层卷积适配 7 通道”已经被删除，不需要了。

    def forward(self, x):
        # =========================================================
        # A. 尺寸对齐 (Padding 到 32 的倍数)
        # =========================================================
        h, w = x.shape[-2:]
        target_h = ((h - 1) // 32 + 1) * 32
        target_w = ((w - 1) // 32 + 1) * 32
        
        if target_h != h or target_w != w:
            pad_h = target_h - h
            pad_w = target_w - w
            # 右侧和下方填充
            x_padded = F.pad(x, (0, pad_w, 0, pad_h))
        else:
            x_padded = x

        # =========================================================
        # B. 前向传播
        # =========================================================
        outputs = self.model(pixel_values=x_padded)
        
        # 提取 Raw Outputs
        class_logits = outputs.class_queries_logits
        mask_logits = outputs.masks_queries_logits
        
        # =========================================================
        # C. 解码为语义分割图
        # =========================================================
        # 移除 "No Object" 类并激活
        class_prob = class_logits[..., :-1].softmax(dim=-1)
        mask_prob = mask_logits.sigmoid()
        
        # 矩阵乘法融合
        sem_seg = torch.einsum("bqc, bqhw -> bchw", class_prob, mask_prob)
        
        # =========================================================
        # D. 裁剪并上采样回原始尺寸
        # =========================================================
        # 此时的 sem_seg 是 padded_size / 4 的大小。
        # 直接 interpolate 回原始输入的 h, w，自动完成裁边和上采样
        sem_seg = F.interpolate(
            sem_seg, 
            size=(h, w), 
            mode="bilinear", 
            align_corners=False
        )
        
        return sem_seg