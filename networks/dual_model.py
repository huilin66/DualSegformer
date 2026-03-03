import torch
import torch.nn as nn
import segmentation_models_pytorch as smp
import torch.nn.functional as F

# --- 1. 双分支通道注意力模块 ---
class DualBranchAttention(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.attention = nn.Sequential(
            nn.Conv2d(channels * 2, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels * 2, kernel_size=1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, f1, f2):
        concat_f = torch.cat([f1, f2], dim=1)
        weights = self.attention(self.avg_pool(concat_f))
        w1, w2 = torch.split(weights, f1.size(1), dim=1)
        return f1 * w1 + f2 * w2

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.SiLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)

class MoE_Fusion(nn.Module):
    def __init__(self, channels, num_experts=4, top_k=2, ):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        in_ch = channels * 2
        self.experts = nn.ModuleList([
            ConvBlock(in_ch, channels) for _ in range(num_experts)
        ])
        self.router = nn.Sequential(
            nn.Linear(channels * 2, channels, bias=False),
            nn.SiLU(inplace=True),
            nn.Linear(channels, num_experts, bias=False)
        )

    def forward(self, E, D):
        """
        E, D: [B, C, H, W]
        return: [B, C, H, W]
        """
        B, C, H, W = E.shape

        e_pool = F.adaptive_avg_pool2d(E, 1).view(B, C)
        d_pool = F.adaptive_avg_pool2d(D, 1).view(B, C)
        router_inp = torch.cat([e_pool, d_pool], dim=1)

        logits = self.router(router_inp)
        gate = F.softmax(logits, dim=-1)

        if self.top_k is not None and self.top_k < self.num_experts:
            topk_val, topk_idx = torch.topk(gate, self.top_k, dim=-1)
            mask = torch.zeros_like(gate)
            mask.scatter_(1, topk_idx, 1.0)
            gate = gate * mask
            gate = gate / (gate.sum(dim=-1, keepdim=True) + 1e-9)

        x = torch.cat([E, D], dim=1)
        out = 0.0
        for i, expert in enumerate(self.experts):
            w = gate[:, i].view(B, 1, 1, 1)
            out = out + w * expert(x)

        return out



class VisionDimReductionExpert(nn.Module):
    """
    视觉专家网络：对单个像素/Patch的通道维度进行降维。
    在展平的特征上使用 Linear，等效于在 NCHW 上使用 1x1 卷积。
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # 使用 MLP 结构增加非线性拟合能力
        self.net = nn.Sequential(
            nn.Linear(in_channels, in_channels),
            nn.GELU(),
            nn.Linear(in_channels, out_channels)
        )

    def forward(self, x):
        return self.net(x)

class MoE_FusionV2(nn.Module):
    def __init__(self, channels, num_experts=4, top_k=2):
        super().__init__()
        self.in_channels = channels
        self.out_channels = channels // 2
        self.num_experts = num_experts
        self.top_k = top_k

        # 路由网络：输入通道数，输出各个专家的得分
        self.router = nn.Linear(self.in_channels, self.num_experts)

        # 专家列表
        self.experts = nn.ModuleList([
            VisionDimReductionExpert(self.in_channels, self.out_channels) 
            for _ in range(self.num_experts)
        ])

    def forward(self, x1, x2):
        """
        输入 x: 形状为 (N, C, H, W)
        输出 : 形状为 (N, C/2, H, W)
        """
        x = torch.cat([x1, x2], dim=1)
        N, C, H, W = x.shape
        
        # 1. 维度置换与展平: (N, C, H, W) -> (N, H, W, C) -> (N*H*W, C)
        # 这样就把特征图变成了 (样本数, 特征维度) 的标准格式
        x_reshaped = x.permute(0, 2, 3, 1).contiguous()
        x_flat = x_reshaped.view(-1, C)
        
        # 2. 计算路由打分
        router_logits = self.router(x_flat) # (N*H*W, num_experts)
        
        # 3. Top-K 路由与权重计算
        # routing_weights 形状: (N*H*W, top_k)
        # selected_experts 形状: (N*H*W, top_k)
        routing_weights, selected_experts = torch.topk(router_logits, self.top_k, dim=-1)
        routing_weights = F.softmax(routing_weights, dim=-1)
        
        # 初始化输出张量 (N*H*W, C/2)
        final_output = torch.zeros(
            (x_flat.size(0), self.out_channels), 
            device=x.device, 
            dtype=x.dtype
        )
        
        # 4. 专家处理与加权融合
        for i, expert in enumerate(self.experts):
            # 找出哪些像素/Patch 选择了第 i 个专家
            expert_mask = (selected_experts == i)
            
            if expert_mask.any():
                # 获取实际选中的行索引 (N*H*W 中的位置)
                batch_idx = expert_mask.any(dim=-1)
                
                # 获取这些像素的特征并交由专家降维: C -> C/2
                expert_inputs = x_flat[batch_idx]
                expert_outputs = expert(expert_inputs)
                
                # 获取对应的路由权重并扩展维度对齐
                expert_weights = routing_weights[expert_mask].unsqueeze(-1)
                
                # 加权累加到对应位置
                final_output[batch_idx] += expert_outputs * expert_weights
                
        # 5. 还原回图像特征格式: (N*H*W, C/2) -> (N, H, W, C/2) -> (N, C/2, H, W)
        final_output = final_output.view(N, H, W, self.out_channels)
        final_output = final_output.permute(0, 3, 1, 2).contiguous()
        
        return final_output

# --- 2. 全能异构双主干封装类 ---
class UniversalDualWrapper(nn.Module):
    def __init__(self, main_model, aux_model, channels_1=(0, 1, 2), channels_2=(3, 4, 5, 6), fusion_type='att'):
        '''
        参数:
        - main_model: 实例化好的主模型
        - aux_model: 实例化好的辅模型
        - channels_1: 分配给主模型的波段索引
        - channels_2: 分配给辅模型的波段索引
        - fusion_type: 融合方式，支持 'att' (注意力), 'add' 或 '+', 'cat' (拼接)
        '''
        super().__init__()
        self.channels_1 = list(channels_1)
        self.channels_2 = list(channels_2)
        self.fusion_type = fusion_type.lower()
        
        if self.fusion_type == '+': 
            self.fusion_type = 'add'
            
        assert self.fusion_type in ['add', 'cat', 'att', 'moe', 'moev2'], "fusion_type 必须是 'add', 'cat', 'att' 或 'moe'"
        
        self.main_model = main_model
        self.encoder2 = aux_model.encoder 
        
        out_channels_1 = self.main_model.encoder.out_channels
        out_channels_2 = self.encoder2.out_channels
        
        assert len(out_channels_1) == len(out_channels_2), "主辅 Backbone 的特征层数必须一致"

        # 初始化可能用到的模块列表
        self.align_convs = nn.ModuleList()       # 用于 add 和 att 时的通道对齐
        self.attention_fusions = nn.ModuleList() # 用于 att 的注意力机制
        self.moe_fusions = nn.ModuleList() # 用于 moe 的注意力机制
        self.cat_convs = nn.ModuleList()         # 用于 cat 的降维机制
        
        for ch1, ch2 in zip(out_channels_1, out_channels_2):
            # 1. 为 'add' 和 'att' 构建对齐卷积
            if self.fusion_type in ['add', 'att', 'moe', 'moev2']:
                if ch1 != ch2 and ch2 > 0 and ch1 > 0:
                    self.align_convs.append(nn.Conv2d(ch2, ch1, kernel_size=1))
                else:
                    self.align_convs.append(nn.Identity())
                    
            # 2. 为 'att' 构建注意力模块
            if self.fusion_type == 'att':
                if ch1 > 0:
                    self.attention_fusions.append(DualBranchAttention(channels=ch1))
                else:
                    self.attention_fusions.append(nn.Identity())
            
            if self.fusion_type == 'moe':
                if ch1 > 0:
                    self.moe_fusions.append(MoE_Fusion(channels=ch1))
                else:
                    self.moe_fusions.append(nn.Identity())
                        
            if self.fusion_type == 'moev2':
                if ch1 > 0:
                    self.moe_fusions.append(MoE_FusionV2(channels=ch1+ch2))
                else:
                    self.moe_fusions.append(nn.Identity()) 
            # 3. 为 'cat' 构建降维卷积 (直接拼接 ch1 和 ch2，然后用 1x1 卷积降维回 ch1)
            if self.fusion_type == 'cat':
                if ch1 > 0 and ch2 > 0:
                    self.cat_convs.append(nn.Conv2d(ch1 + ch2, ch1, kernel_size=1))
                else:
                    self.cat_convs.append(nn.Identity())

        # 初始化新增层的权重
        self._initialize_weights()

    def _initialize_weights(self):
        # 统一的 Kaiming 初始化函数
        def init_kaiming(m):
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
                
        # 对用到哪种算子就初始化哪种
        if self.fusion_type in ['add', 'att', 'moe', 'moev2']:
            self.align_convs.apply(init_kaiming)
        if self.fusion_type == 'att':
            self.attention_fusions.apply(init_kaiming)
        if self.fusion_type == 'cat':
            self.cat_convs.apply(init_kaiming)
        if self.fusion_type == 'moe':
            self.moe_fusions.apply(init_kaiming)
        if self.fusion_type == 'moev2':
            self.moe_fusions.apply(init_kaiming)

    def forward(self, x):
        x1 = x[:, self.channels_1, :, :]  
        x2 = x[:, self.channels_2, :, :]  
        
        features1 = self.main_model.encoder(x1)
        features2 = self.encoder2(x2)
        
        fused_features = []
        for i, (f1, f2) in enumerate(zip(features1, features2)):
            # 第0层通常是原始图像层或未下采样层，直接保留主干特征避免冲突
            if i == 0:
                fused_features.append(f1)
                continue
                
            if self.fusion_type == 'add':
                f2_aligned = self.align_convs[i](f2)
                fused_features.append(f1 + f2_aligned)
            elif self.fusion_type == 'moe' or self.fusion_type == 'moev2':
                f2_aligned = self.align_convs[i](f2)
                
                # 【修复核心】检查当前模块是不是 nn.Identity
                if isinstance(self.moe_fusions[i], nn.Identity):
                    # 如果是 Identity（通常是无通道的特殊层），如果形状一致就相加，不一致就直接保留主干特征
                    fused = f1 + f2_aligned if f1.shape == f2_aligned.shape else f1
                else:
                    # 正常的注意力机制融合
                    fused = self.moe_fusions[i](f1, f2_aligned)
                    
                fused_features.append(fused)

            elif self.fusion_type == 'att':
                f2_aligned = self.align_convs[i](f2)
                
                # 【修复核心】检查当前模块是不是 nn.Identity
                if isinstance(self.attention_fusions[i], nn.Identity):
                    # 如果是 Identity（通常是无通道的特殊层），如果形状一致就相加，不一致就直接保留主干特征
                    fused = f1 + f2_aligned if f1.shape == f2_aligned.shape else f1
                else:
                    # 正常的注意力机制融合
                    fused = self.attention_fusions[i](f1, f2_aligned)
                    
                fused_features.append(fused)
                
            elif self.fusion_type == 'cat':
                # 直接在通道维度拼接，然后过卷积降维到主干期望的通道数
                fused = torch.cat([f1, f2], dim=1)
                fused = self.cat_convs[i](fused)
                fused_features.append(fused)
                

        decoder_output = self.main_model.decoder(fused_features)
        masks = self.main_model.segmentation_head(decoder_output)
        
        return masks


class DualUNetFormerWrapper(nn.Module):
    def __init__(self, main_model, aux_model, channels_1=(0, 1, 2), channels_2=(3, 4, 5, 6), fusion_type='add'):
        super().__init__()
        self.channels_1 = list(channels_1)
        self.channels_2 = list(channels_2)
        self.fusion_type = fusion_type.lower()
        if self.fusion_type == '+': 
            self.fusion_type = 'add'
            
        self.main_model = main_model
        self.aux_model = aux_model
        
        # UNetFormer 的特征通道数提取方式
        out_channels = self.main_model.backbone.feature_info.channels()
        
        self.attention_fusions = nn.ModuleList()
        self.moe_fusions = nn.ModuleList()
        self.cat_convs = nn.ModuleList()
        
        # 为 4 个层级的特征构建融合模块 (因为是同构网络，通道数完全一致，无需 align_conv)
        for ch in out_channels:
            if self.fusion_type == 'att':
                self.attention_fusions.append(DualBranchAttention(channels=ch))
            elif self.fusion_type == 'moe':
                self.moe_fusions.append(MoE_Fusion(channels=ch))
            elif self.fusion_type == 'cat':
                self.cat_convs.append(nn.Conv2d(ch * 2, ch, kernel_size=1))
                
        self._initialize_weights()

    def _initialize_weights(self):
        def init_kaiming(m):
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        
        if self.fusion_type == 'att':
            self.attention_fusions.apply(init_kaiming)
        elif self.fusion_type == 'cat':
            self.cat_convs.apply(init_kaiming)
        elif self.fusion_type == 'moe':
            self.moe_fusions.apply(init_kaiming)

    def forward(self, x):
        h_img, w_img = x.size()[-2:]
        x1 = x[:, self.channels_1, :, :]  
        x2 = x[:, self.channels_2, :, :]  
        
        # UNetFormer 提取特征 (返回 4 个 stage 的元组)
        features1 = self.main_model.backbone(x1)
        features2 = self.aux_model.backbone(x2)
        
        fused_features = []
        for i, (f1, f2) in enumerate(zip(features1, features2)):
            if self.fusion_type == 'add':
                fused = f1 + f2
            elif self.fusion_type == 'att':
                fused = self.attention_fusions[i](f1, f2)
            elif self.fusion_type == 'moe':
                fused = self.moe_fusions[i](f1, f2)
            elif self.fusion_type == 'cat':
                fused = self.cat_convs[i](torch.cat([f1, f2], dim=1))
            fused_features.append(fused)
            
        # 解包融合后的特征
        res1, res2, res3, res4 = fused_features
        
        # UNetFormer 特有的 Decoder 调用方式 (自带分割头)
        out = self.main_model.decoder(res1, res2, res3, res4, h_img, w_img)
        

        return out