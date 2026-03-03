import torch
import torch.nn as nn
import torch.nn.functional as F
import math

try:
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
except ImportError:
    selective_scan_fn = None


# ==========================================
# Part A: Multimodal Feature Fusion (MFF)
# 論文 Fig. 3 [cite: 133]
# ==========================================
class MFFModule(nn.Module):
    def __init__(self, in_channels):
        super(MFFModule, self).__init__()
        # 根據論文：先將通道縮減為原來的 1/5 [cite: 146]
        self.reduced_channels = in_channels // 5
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, self.reduced_channels, kernel_size=1),
            nn.GroupNorm(1, self.reduced_channels), # Replaced BatchNorm with GroupNorm (G=1 -> LayerNorm-like) to avoid conflict with Per-Image Norm
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x):
        # x: F_MF (Multimodal Features)
        
        # 1. 降維 (Channel Reduction) -> F_MMF
        f_mmf = self.conv(x)
        
        # 2. Global Information Representation 
        # AvgPool 分支
        avg_pool = torch.mean(f_mmf, dim=(2, 3), keepdim=True) # Global Avg Pooling
        # MaxPool 分支
        max_pool = torch.amax(f_mmf, dim=(2, 3), keepdim=True) # Global Max Pooling
        
        # 3. Excitation (類似 SE-Block 的操作，論文公式 1) 
        # F_FF = F_MFF * I_avg + F_MFF * I_max
        # 這裡假設 I_avg 和 I_max 是經過 pooling 後的權重，直接廣播相乘
        out_avg = f_mmf * avg_pool
        out_max = f_mmf * max_pool
        
        f_ff = out_avg + out_max
        return f_ff


class SSMCore(nn.Module):
    def __init__(self, d_inner, d_state=16, dt_rank="auto"):
        super().__init__()
        self.d_inner = d_inner
        self.d_state = d_state
        self.dt_rank = math.ceil(d_inner / 16) if dt_rank == "auto" else dt_rank

        self.x_proj = nn.Linear(d_inner, self.dt_rank + self.d_state * 2, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, d_inner, bias=True)

        # S4D real initialization
        A = torch.arange(1, d_state + 1, dtype=torch.float32).repeat(d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(d_inner))
        
        # Init dt_proj
        dt_init_std = 2**-4
        nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
        dt = torch.exp(
            torch.rand(d_inner) * (math.log(0.1) - math.log(0.001))
            + math.log(0.001)
        )
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)

    def forward(self, x):
        # x: [B, H, W, C]
        B, H, W, C = x.shape
        x_flat = x.view(B, -1, C) # [B, L, C]
        
        x_dbl = self.x_proj(x_flat) 
        (delta, B_param, C_param) = x_dbl.split([self.dt_rank, self.d_state, self.d_state], dim=-1)
        
        delta = self.dt_proj(delta) 
        
        u = x_flat.transpose(1, 2).contiguous()
        delta = delta.transpose(1, 2).contiguous()
        B_param = B_param.transpose(1, 2).contiguous()
        C_param = C_param.transpose(1, 2).contiguous()
        
        y = selective_scan_fn(
            u, delta, 
            -torch.exp(self.A_log.float()), 
            B_param, C_param, 
            self.D.float(), 
            z=None,
            delta_bias=self.dt_proj.bias.float(),
            delta_softplus=True
        )
        
        y = y.transpose(1, 2).view(B, H, W, C)
        return y

# ==========================================
# Part B: Visual State Space Duality (VSSD) Block
# 論文 Fig. 2 [cite: 108]
# ==========================================
class VSSDBlock(nn.Module):
    def __init__(self, d_model, d_state=16, d_conv=3, expand=2):
        super(VSSDBlock, self).__init__()
        self.d_model = d_model
        self.d_inner = int(expand * d_model)
        
        # Input Layer Norm [cite: 100]
        self.ln = nn.LayerNorm(d_model)
        
        # 分支 1: Linear (Gate)
        self.linear_gate = nn.Linear(d_model, self.d_inner)
        
        # 分支 2: Main process
        self.linear_x = nn.Linear(d_model, self.d_inner)
        
        # 深度卷積 (Depth-wise Conv) [cite: 137]
        # 用於捕捉局部特徵 (Local Perception)
        self.conv = nn.Conv2d(self.d_inner, self.d_inner, kernel_size=d_conv, 
                              padding=(d_conv-1)//2, groups=self.d_inner)
        self.act = nn.SiLU()
        
        # --- Mamba / SSM Core ---
        # 如果有安裝 mamba_ssm，則使用真正的 Selective Scan
        if selective_scan_fn is not None:
            self.ssm_core = SSMCore(self.d_inner, d_state=d_state)
            self.use_mamba = True
        else:
            self.use_mamba = False
            # 這裡使用一個輕量級的 Gated Conv 來模擬 SSM 的"選擇性"特性，
            # 確保使用者不需安裝複雜環境即可執行程式碼。
            self.ssm_simulator = nn.Sequential(
                nn.Linear(self.d_inner, self.d_inner),
                nn.SiLU()
            )
        # ------------------------

        self.out_norm = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, d_model)
        
        # Feed Forward Network (FFN) [cite: 138]
        self.ffn = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model)
        )

    def forward(self, x):
        # x shape: [Batch, Channels, Height, Width] -> 需轉為 [B, H, W, C] 做 Linear
        B, C, H, W = x.shape
        resid = x
        
        # Permute for LayerNorm and Linear: [B, C, H, W] -> [B, H, W, C]
        x_in = x.permute(0, 2, 3, 1)
        x_in = self.ln(x_in)
        
        # VSSD Flow (Simplified based on Fig 2)
        
        # Branch 1 & 2
        z = self.linear_gate(x_in) # Gate branch
        x_feat = self.linear_x(x_in) # Main branch
        
        # DW Conv (需轉回 [B, C, H, W] 進行 Conv2d)
        x_feat = x_feat.permute(0, 3, 1, 2)
        x_feat = self.conv(x_feat)
        x_feat = self.act(x_feat)
        
        # SSM / VSSD Core Process
        x_feat = x_feat.permute(0, 2, 3, 1) # Back to [B, H, W, C]
        if self.use_mamba:
            x_feat = self.ssm_core(x_feat)
        else:
            x_feat = self.ssm_simulator(x_feat) # 模擬 NC-SSD [cite: 116]
        
        # Normalization
        x_feat = self.out_norm(x_feat)
        
        # Multiply with Gate (Element-wise product)
        out = x_feat * F.silu(z)
        
        # Output Projection
        out = self.out_proj(out)
        
        # Skip Connection 1 (Resid)
        out = out.permute(0, 3, 1, 2) # [B, C, H, W]
        out = out + resid
        
        # FFN with Skip Connection 2
        out_ffn = out.permute(0, 2, 3, 1)
        out_ffn = self.ffn(out_ffn)
        out_ffn = out_ffn.permute(0, 3, 1, 2)
        
        return out + out_ffn


class InputChannelDropout(nn.Module):
    def __init__(self, p=0.1, min_drop=1, max_drop=3):
        """
        Args:
            p (float): 触发丢弃操作的概率 (默认 0.1，即 10%)
            min_drop (int): 最少丢弃几个通道
            max_drop (int): 最多丢弃几个通道
        """
        super().__init__()
        self.p = p
        self.min_drop = min_drop
        self.max_drop = max_drop

    def forward(self, x):
        # 1. 如果不是训练模式 (val/test)，或者概率没命中，直接返回原图
        if not self.training or x.shape[1] <= 1:
            return x
        
        # 使用 torch.rand 确保受 torch.manual_seed 控制 (可复现性)
        if torch.rand(1, device=x.device).item() > self.p:
            return x

        # 2. 只有命中概率 (10%) 后才执行以下逻辑
        # x shape: [Batch, Channel, Height, Width]
        B, C, H, W = x.shape
        
        # 克隆 x，避免原地修改影响梯度或原始数据
        x_aug = x.clone()

        # 策略：为了效率和模拟"传感器故障"，通常对整个 Batch 丢弃相同的通道
        # 随机决定要丢弃的数量 k (范围: min_drop ~ max_drop)
        # 使用 torch.randint 保证可复现性
        k = torch.randint(self.min_drop, self.max_drop + 1, (1,), device=x.device).item()
        
        # 确保 k 不超过实际通道数 (防止报错)
        k = min(k, C)

        # 随机选择 k 个通道索引
        # torch.randperm 生成 0 到 C-1 的随机排列，取前 k 个
        drop_indices = torch.randperm(C, device=x.device)[:k]

        # 3. 将选中的通道置为 0 (Zero-out)
        # 广播机制：x_aug[:, indices, :, :] 会将所有样本的对应通道置 0
        x_aug[:, drop_indices, :, :] = 0.0

        return x_aug

# ==========================================
# Part C: M3LSNet Main Architecture
# 論文 Fig. 1 [cite: 89, 92]
# ==========================================
class M3LSNet(nn.Module):
    def __init__(self, input_channels=7, num_classes=2, img_size=128):
        super(M3LSNet, self).__init__()
        
        # 假設輸入是 5 個模態的 Concatenation
        # RGB(3) + DEM(1) + Thermal(1) + Slope(1) + Grayscale(1) = 7 Channels
        # 註：論文中未明確說明輸入是 5 個獨立 encoder 還是 concat，
        # 但 Fig 1 左側只有一個 "Encoder" 柱狀圖，且 Section II.A 提到 "merge... through channel-wise concatenation" [cite: 41]。
        # input_channels = 3 + 1 + 1 + 1 + 1 
        
        # --- Encoder (Downsampling Path) ---
        # Stage 1
        self.stem = nn.Sequential(
            nn.Conv2d(input_channels, 48, kernel_size=3, padding=1),
            nn.BatchNorm2d(48),
            nn.ReLU()
        )
        self.enc1 = nn.Sequential(*[VSSDBlock(48) for _ in range(2)]) # L1 blocks
        self.mff1 = MFFModule(48) # Skip connection processing
        
        # Stage 2
        self.down1 = nn.Conv2d(48, 96, kernel_size=2, stride=2)
        self.enc2 = nn.Sequential(*[VSSDBlock(96) for _ in range(2)])
        self.mff2 = MFFModule(96)
        
        # Stage 3
        self.down2 = nn.Conv2d(96, 192, kernel_size=2, stride=2)
        self.enc3 = nn.Sequential(*[VSSDBlock(192) for _ in range(2)])
        self.mff3 = MFFModule(192)
        
        # Stage 4 (Bottleneck with Attention)
        self.down3 = nn.Conv2d(192, 384, kernel_size=2, stride=2)
        self.enc4 = nn.Sequential(*[VSSDBlock(384) for _ in range(2)])
        # 論文提到 Stage 4 包含 Multi-Head Attention Block [cite: 81, 143]
        self.attn = nn.MultiheadAttention(embed_dim=384, num_heads=8, batch_first=True)
        self.mff4 = MFFModule(384)
        
        # --- Decoder (Upsampling Path) ---
        # Decoder 4 -> 3
        self.up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec_conv1 = nn.Conv2d(384 + 192//5, 192, kernel_size=1) # Concat MFF output (reduced dim)
        self.dec1 = VSSDBlock(192)
        
        # Decoder 3 -> 2
        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec_conv2 = nn.Conv2d(192 + 96//5, 96, kernel_size=1)
        self.dec2 = VSSDBlock(96)
        
        # Decoder 2 -> 1
        self.up3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec_conv3 = nn.Conv2d(96 + 48//5, 48, kernel_size=1)
        self.dec3 = VSSDBlock(48)
        
        # Output Head
        self.final_conv = nn.Conv2d(48, num_classes, kernel_size=1)
        self.channel_dropout = InputChannelDropout(p=0.1, min_drop=1, max_drop=3)

    def forward(self, x):
        # x: [B, 7, 128, 128]
        x = self.channel_dropout(x)
        # --- Encoder ---
        # Stage 1
        x1 = self.stem(x)
        x1 = self.enc1(x1)
        f1 = self.mff1(x1) # Feature for skip
        
        # Stage 2
        x2 = self.down1(x1)
        x2 = self.enc2(x2)
        f2 = self.mff2(x2)
        
        # Stage 3
        x3 = self.down2(x2)
        x3 = self.enc3(x3)
        f3 = self.mff3(x3)
        
        # Stage 4 (Bottleneck)
        x4 = self.down3(x3)
        x4 = self.enc4(x4)
        
        # Apply Multi-Head Attention at bottleneck [cite: 143]
        B, C, H, W = x4.shape
        x4_flat = x4.flatten(2).transpose(1, 2) # [B, H*W, C]
        attn_out, _ = self.attn(x4_flat, x4_flat, x4_flat)
        x4 = attn_out.transpose(1, 2).reshape(B, C, H, W) + x4 # Residual
        
        f4 = self.mff4(x4)
        
        # --- Decoder ---
        # 論文 Fig. 1 右側顯示 Cross-Scale Feature Fusion (F圈圈)
        # 這裡將 MFF 處理後的 Encoder 特徵與 Upsampled Decoder 特徵結合
        
        # Block 1
        d1 = self.up1(x4)
        # Concat with F3 (MFF output of Stage 3)
        # 注意: MFF 輸出通道是原本的 1/5，需在此對齊
        d1 = torch.cat([d1, f3], dim=1) 
        d1 = self.dec_conv1(d1)
        d1 = self.dec1(d1)
        
        # Block 2
        d2 = self.up2(d1)
        d2 = torch.cat([d2, f2], dim=1)
        d2 = self.dec_conv2(d2)
        d2 = self.dec2(d2)
        
        # Block 3
        d3 = self.up3(d2)
        d3 = torch.cat([d3, f1], dim=1)
        d3 = self.dec_conv3(d3)
        d3 = self.dec3(d3)
        
        # Output
        out = self.final_conv(d3)
        return out