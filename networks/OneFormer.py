import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import OneFormerForUniversalSegmentation, OneFormerConfig, SwinConfig, SwinModel, CLIPTokenizer

class HF_OneFormer(nn.Module):
    def __init__(self, in_channels=7, num_classes=2, backbone="swin-tiny"):
        super().__init__()
        
        print(f"🚀 Initializing OneFormer with Public Microsoft Swin Backbone...")

        # =========================================================
        # 1. 手动构建 OneFormer 配置 (完全避开 shi-labs 的 config)
        # =========================================================
        # 我们参考 Swin-Tiny 的标准参数
        config = OneFormerConfig(
            # 基础设置
            num_labels=num_classes,
            ignore_mismatched_sizes=True,
            # Backbone 设置 (对应 Swin-Tiny)
            backbone_config={
                "model_type": "swin",
                "image_size": 224,
                "patch_size": 4,
                "embed_dim": 96,
                "depths": [2, 2, 6, 2],
                "num_heads": [3, 6, 12, 24],
                "window_size": 7,
                "mlp_ratio": 4.0,
                "out_indices": [1, 2, 3, 4], # OneFormer 需要多尺度特征
            },
            # Head 设置 (OneFormer 标准)
            encoder_feed_forward_network_dim=1024,
            dim_feedforward=2048,
        )

        # =========================================================
        # 2. 创建一个“空白”的 OneFormer 模型 (随机权重)
        # =========================================================
        self.model = OneFormerForUniversalSegmentation(config)

        # =========================================================
        # 3. 加载 Microsoft 公开的 Swin 权重 (移花接木)
        # =========================================================
        # 这是 Microsoft 官方公开的，无权限限制
        public_swin_repo = "microsoft/swin-tiny-patch4-window7-224"
        print(f"📥 Loading public weights from: {public_swin_repo}")
        
        public_swin = SwinModel.from_pretrained(public_swin_repo)
        
        # 将 Microsoft 的权重加载到 OneFormer 的 Encoder 部分
        # 注意：strict=False 是必须的，因为我们只加载 Backbone，不加载 Head
        msg = self.model.model.pixel_level_module.encoder.load_state_dict(
            public_swin.state_dict(), 
            strict=False
        )
        print(f"✅ Backbone weights loaded. Missing keys (expected for Head): {len(msg.missing_keys)}")

        # =========================================================
        # 4. 初始化 Tokenizer (用于生成任务文本)
        # =========================================================
        # 使用 OpenAI 公开的 CLIP Tokenizer，完全兼容 OneFormer
        self.tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
        
        # 预生成 "语义分割" 的任务 Token
        task_tokens = self.tokenizer(
            ["the task is semantic"], 
            padding="max_length", 
            max_length=77, 
            return_tensors="pt"
        )
        self.register_buffer("task_token", task_tokens["input_ids"])

        # =========================================================
        # 5. 修改第一层卷积适配 7 通道
        # =========================================================
        patch_embed = self.model.model.pixel_level_module.encoder.embeddings.patch_embeddings.projection
        
        if patch_embed.in_channels != in_channels:
            print(f"🔧 Adapting input from {patch_embed.in_channels} to {in_channels} channels...")
            new_conv = nn.Conv2d(
                in_channels=in_channels,
                out_channels=patch_embed.out_channels,
                kernel_size=patch_embed.kernel_size,
                stride=patch_embed.stride,
                padding=patch_embed.padding
            )
            # 复制权重: 前3通道用训练好的，后4通道复用
            with torch.no_grad():
                new_conv.weight[:, :3, :, :] = patch_embed.weight
                new_conv.weight[:, 3:6, :, :] = patch_embed.weight
                new_conv.weight[:, 6:7, :, :] = patch_embed.weight[:, 0:1, :, :]
                if patch_embed.bias is not None:
                    new_conv.bias = patch_embed.bias
            
            self.model.model.pixel_level_module.encoder.embeddings.patch_embeddings.projection = new_conv

    def forward(self, x):
        # Swin 尺寸适配 (128 -> 140)
        h, w = x.shape[-2:]
        target_h = ((h - 1) // 28 + 1) * 28
        target_w = ((w - 1) // 28 + 1) * 28
        
        if target_h != h or target_w != w:
            pad_h = target_h - h
            pad_w = target_w - w
            x_padded = F.pad(x, (0, pad_w, 0, pad_h))
        else:
            x_padded = x

        # 扩展 task token
        current_task_inputs = self.task_token.expand(x.size(0), -1)
        
        # Forward
        outputs = self.model(pixel_values=x_padded, task_inputs=current_task_inputs)
        
        # 解码
        class_logits = outputs.class_queries_logits
        mask_logits = outputs.masks_queries_logits
        
        class_prob = class_logits[..., :-1].softmax(dim=-1)
        mask_prob = mask_logits.sigmoid()
        
        sem_seg = torch.einsum("bqc, bqhw -> bchw", class_prob, mask_prob)
        
        # 裁剪回原尺寸
        sem_seg = F.interpolate(sem_seg, size=(h, w), mode="bilinear", align_corners=False)
        
        return sem_seg


import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import OneFormerForUniversalSegmentation, OneFormerConfig, ConvNextModel, CLIPTokenizer

class HF_OneFormer_ConvNeXt(nn.Module):
    def __init__(self, in_channels=7, num_classes=2, backbone="convnext-tiny"):
        super().__init__()
        
        print(f"🚀 Initializing OneFormer with Public Meta ConvNeXt Backbone...")

        # =========================================================
        # 1. 手动构建 OneFormer 配置 (添加 num_channels 绕过输入检测)
        # =========================================================
        config = OneFormerConfig(
            num_labels=num_classes,
            ignore_mismatched_sizes=True,
            
            backbone_config={
                "model_type": "convnext",
                "num_channels": in_channels,          # [新增] 防止 forward 时报错 7 通道不匹配
                "depths": [3, 3, 9, 3],               
                "hidden_sizes": [96, 192, 384, 768],  
                "out_indices": [0, 1, 2, 3],          
                "drop_path_rate": 0.1,
            },
            
            encoder_feed_forward_network_dim=1024,
            dim_feedforward=2048,
        )

        self.model = OneFormerForUniversalSegmentation(config)

        # =========================================================
        # 2. 加载权重并提前“扩建”到 7 通道，防止 size mismatch
        # =========================================================
        public_convnext_repo = "facebook/convnext-tiny-224"
        print(f"📥 Loading public weights from: {public_convnext_repo}")
        
        # [修改] 使用 revision="refs/pr/4" 加载 safetensors 绕过安全报错
        public_convnext = ConvNextModel.from_pretrained(
            public_convnext_repo,
            revision="refs/pr/4" 
        )
        
        pretrained_state_dict = public_convnext.state_dict()
        old_weight = pretrained_state_dict["embeddings.patch_embeddings.weight"]
        
        if old_weight.shape[1] != in_channels:
            print(f"🔧 Adapting checkpoint weights from {old_weight.shape[1]} to {in_channels} channels...")
            new_weight = torch.zeros(
                (old_weight.shape[0], in_channels, old_weight.shape[2], old_weight.shape[3]), 
                dtype=old_weight.dtype, 
                device=old_weight.device
            )
            with torch.no_grad():
                new_weight[:, :3, :, :] = old_weight
                new_weight[:, 3:6, :, :] = old_weight
                new_weight[:, 6:7, :, :] = old_weight[:, 0:1, :, :]
                
            pretrained_state_dict["embeddings.patch_embeddings.weight"] = new_weight

        msg = self.model.model.pixel_level_module.encoder.load_state_dict(
            pretrained_state_dict, 
            strict=False
        )
        print(f"✅ Backbone weights loaded. Missing keys (expected for Head): {len(msg.missing_keys)}")

        # =========================================================
        # 3. 初始化 Tokenizer
        # =========================================================
        self.tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
        task_tokens = self.tokenizer(
            ["the task is semantic"], 
            padding="max_length", 
            max_length=77, 
            return_tensors="pt"
        )
        self.register_buffer("task_token", task_tokens["input_ids"])

    def forward(self, x):
        h, w = x.shape[-2:]
        target_h = ((h - 1) // 32 + 1) * 32
        target_w = ((w - 1) // 32 + 1) * 32
        
        if target_h != h or target_w != w:
            pad_h = target_h - h
            pad_w = target_w - w
            x_padded = F.pad(x, (0, pad_w, 0, pad_h))
        else:
            x_padded = x

        current_task_inputs = self.task_token.expand(x.size(0), -1)
        outputs = self.model(pixel_values=x_padded, task_inputs=current_task_inputs)
        
        class_logits = outputs.class_queries_logits
        mask_logits = outputs.masks_queries_logits
        
        class_prob = class_logits[..., :-1].softmax(dim=-1)
        mask_prob = mask_logits.sigmoid()
        
        sem_seg = torch.einsum("bqc, bqhw -> bchw", class_prob, mask_prob)
        sem_seg = F.interpolate(sem_seg, size=(h, w), mode="bilinear", align_corners=False)
        
        return sem_seg