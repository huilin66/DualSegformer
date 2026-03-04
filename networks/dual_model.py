import torch
import torch.nn as nn
import segmentation_models_pytorch as smp
import torch.nn.functional as F

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
    def __init__(self, in_channels, out_channels):
        super().__init__()
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

        self.router = nn.Linear(self.in_channels, self.num_experts)
        self.experts = nn.ModuleList([
            VisionDimReductionExpert(self.in_channels, self.out_channels) 
            for _ in range(self.num_experts)
        ])

    def forward(self, x1, x2):
        x = torch.cat([x1, x2], dim=1)
        N, C, H, W = x.shape

        x_reshaped = x.permute(0, 2, 3, 1).contiguous()
        x_flat = x_reshaped.view(-1, C)
        
        router_logits = self.router(x_flat) # (N*H*W, num_experts)
        
        routing_weights, selected_experts = torch.topk(router_logits, self.top_k, dim=-1)
        routing_weights = F.softmax(routing_weights, dim=-1)

        final_output = torch.zeros(
            (x_flat.size(0), self.out_channels), 
            device=x.device, 
            dtype=x.dtype
        )

        for i, expert in enumerate(self.experts):
            expert_mask = (selected_experts == i)
            
            if expert_mask.any():
                batch_idx = expert_mask.any(dim=-1)
                
                expert_inputs = x_flat[batch_idx]
                expert_outputs = expert(expert_inputs)
                expert_weights = routing_weights[expert_mask].unsqueeze(-1)
        
                final_output[batch_idx] += expert_outputs * expert_weights
                
        final_output = final_output.view(N, H, W, self.out_channels)
        final_output = final_output.permute(0, 3, 1, 2).contiguous()
        
        return final_output


class UniversalDualWrapper(nn.Module):
    def __init__(self, main_model, aux_model, channels_1=(0, 1, 2), channels_2=(3, 4, 5, 6), fusion_type='att'):
        super().__init__()
        self.channels_1 = list(channels_1)
        self.channels_2 = list(channels_2)
        self.fusion_type = fusion_type.lower()
        
        if self.fusion_type == '+': 
            self.fusion_type = 'add'
            
        assert self.fusion_type in ['add', 'cat', 'att', 'moe', 'moev2'], "fusion_type must be ['add', 'cat', 'att', 'moe', 'moev2']"
        
        self.main_model = main_model
        self.encoder2 = aux_model.encoder 
        
        out_channels_1 = self.main_model.encoder.out_channels
        out_channels_2 = self.encoder2.out_channels
        
        assert len(out_channels_1) == len(out_channels_2), "Backbone misalign"


        self.align_convs = nn.ModuleList()       
        self.attention_fusions = nn.ModuleList() 
        self.moe_fusions = nn.ModuleList() 
        self.cat_convs = nn.ModuleList()        
        
        for ch1, ch2 in zip(out_channels_1, out_channels_2):
            if self.fusion_type in ['add', 'att', 'moe', 'moev2']:
                if ch1 != ch2 and ch2 > 0 and ch1 > 0:
                    self.align_convs.append(nn.Conv2d(ch2, ch1, kernel_size=1))
                else:
                    self.align_convs.append(nn.Identity())
                    
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
            if self.fusion_type == 'cat':
                if ch1 > 0 and ch2 > 0:
                    self.cat_convs.append(nn.Conv2d(ch1 + ch2, ch1, kernel_size=1))
                else:
                    self.cat_convs.append(nn.Identity())
        self._initialize_weights()

    def _initialize_weights(self):
        def init_kaiming(m):
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

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
            if i == 0:
                fused_features.append(f1)
                continue
                
            if self.fusion_type == 'add':
                f2_aligned = self.align_convs[i](f2)
                fused_features.append(f1 + f2_aligned)
            elif self.fusion_type == 'moe' or self.fusion_type == 'moev2':
                f2_aligned = self.align_convs[i](f2)
                if isinstance(self.moe_fusions[i], nn.Identity):
                    fused = f1 + f2_aligned if f1.shape == f2_aligned.shape else f1
                else:
                    fused = self.moe_fusions[i](f1, f2_aligned)
                fused_features.append(fused)

            elif self.fusion_type == 'att':
                f2_aligned = self.align_convs[i](f2)
                if isinstance(self.attention_fusions[i], nn.Identity):
                    fused = f1 + f2_aligned if f1.shape == f2_aligned.shape else f1
                else:
                    fused = self.attention_fusions[i](f1, f2_aligned)
                fused_features.append(fused)
                
            elif self.fusion_type == 'cat':
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

        out_channels = self.main_model.backbone.feature_info.channels()
        
        self.attention_fusions = nn.ModuleList()
        self.moe_fusions = nn.ModuleList()
        self.cat_convs = nn.ModuleList()

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
            
        res1, res2, res3, res4 = fused_features
        out = self.main_model.decoder(res1, res2, res3, res4, h_img, w_img)
        return out