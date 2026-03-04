import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
import torch.nn.functional as F
from segmentation_models_pytorch.base import SegmentationModel


class SpatialGatherModule(nn.Module):
    def __init__(self, scale=1):
        super(SpatialGatherModule, self).__init__()
        self.scale = scale

    def forward(self, feats, probs):

        batch_size, c, h, w = feats.size()
        probs = probs.view(batch_size, -1, h * w)
        feats = feats.view(batch_size, -1, h * w)

        probs = F.softmax(self.scale * probs, dim=2)
        ocr_context = torch.bmm(probs, feats.permute(0, 2, 1))

        return ocr_context.permute(0, 2, 1).unsqueeze(3)


class ObjectAttentionBlock(nn.Module):
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

        query = self.f_pixel(x).view(batch_size, self.key_channels, -1)
        query = query.permute(0, 2, 1)  # [B, HW, KeyC]

        key = self.f_object(proxy).view(
            batch_size, self.key_channels, -1
        )  # [B, KeyC, K]

        value = self.f_down(proxy).view(
            batch_size, self.key_channels, -1
        )  # [B, KeyC, K]
        value = value.permute(0, 2, 1)  # [B, K, KeyC]

        sim_map = torch.matmul(query, key)  # [B, HW, K]
        sim_map = (self.key_channels**-0.5) * sim_map
        sim_map = F.softmax(sim_map, dim=-1)  # 每个像素对 K 个类别的关注度

        context = torch.matmul(sim_map, value)  # [B, HW, KeyC]
        context = context.permute(0, 2, 1).contiguous()
        context = context.view(batch_size, self.key_channels, h, w)

        context = self.f_up(context)
        return x + context


class OCRHead(nn.Module):
    def __init__(self, in_channels, num_classes, ocr_channels=256):
        super(OCRHead, self).__init__()

        self.aux_head = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, num_classes, 1),
        )

        self.spatial_gather = SpatialGatherModule()
        self.object_attention = ObjectAttentionBlock(in_channels, ocr_channels)

        self.cls_head = nn.Conv2d(in_channels, num_classes, 1)

    def forward(self, x):
        aux_out = self.aux_head(x)
        proxy_feats = self.spatial_gather(x, aux_out)
        ocr_feats = self.object_attention(x, proxy_feats)
        final_out = self.cls_head(ocr_feats)
        return final_out, aux_out


class SMP_OCRNet(nn.Module):
    def __init__(
        self,
        encoder_name="tu-hrnet_w18",
        in_channels=3,
        classes=2,
        ocr_mid_channels=256,
    ):
        super().__init__()

        self.encoder = smp.encoders.get_encoder(
            encoder_name, in_channels=in_channels, depth=4, weights="imagenet"
        )
        self.encoder_out_channels = sum(self.encoder.out_channels)

        self.head = OCRHead(
            in_channels=self.encoder_out_channels,
            num_classes=classes,
            ocr_channels=ocr_mid_channels,
        )

    def forward(self, x):
        input_shape = x.shape[-2:]

        features = self.encoder(x)
        target_h, target_w = features[0].shape[2], features[0].shape[3]

        upsampled_feats = [features[0]]
        for i in range(1, len(features)):
            upsampled_feats.append(
                F.interpolate(
                    features[i],
                    size=(target_h, target_w),
                    mode="bilinear",
                    align_corners=True,
                )
            )

        feats = torch.cat(upsampled_feats, dim=1)

        final_logits, aux_logits = self.head(feats)

        final_logits = F.interpolate(
            final_logits, size=input_shape, mode="bilinear", align_corners=True
        )
        aux_logits = F.interpolate(
            aux_logits, size=input_shape, mode="bilinear", align_corners=True
        )

        if self.training:
            return final_logits, aux_logits
        else:
            return final_logits
