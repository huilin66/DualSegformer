import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    CLIPTokenizer,
    ConvNextModel,
    OneFormerConfig,
    OneFormerForUniversalSegmentation,
    SwinModel,
)


class HF_OneFormer(nn.Module):
    def __init__(self, in_channels=7, num_classes=2, backbone="swin-tiny"):
        super().__init__()
        config = OneFormerConfig(
            num_labels=num_classes,
            ignore_mismatched_sizes=True,
            backbone_config={
                "model_type": "swin",
                "image_size": 224,
                "patch_size": 4,
                "embed_dim": 96,
                "depths": [2, 2, 6, 2],
                "num_heads": [3, 6, 12, 24],
                "window_size": 7,
                "mlp_ratio": 4.0,
                "out_indices": [1, 2, 3, 4],
            },
            encoder_feed_forward_network_dim=1024,
            dim_feedforward=2048,
        )

        self.model = OneFormerForUniversalSegmentation(config)

        public_swin_repo = "microsoft/swin-tiny-patch4-window7-224"

        public_swin = SwinModel.from_pretrained(public_swin_repo)

        msg = self.model.model.pixel_level_module.encoder.load_state_dict(
            public_swin.state_dict(), strict=False
        )

        self.tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")

        task_tokens = self.tokenizer(
            ["the task is semantic"],
            padding="max_length",
            max_length=77,
            return_tensors="pt",
        )
        self.register_buffer("task_token", task_tokens["input_ids"])

        patch_embed = self.model.model.pixel_level_module.encoder.embeddings.patch_embeddings.projection

        if patch_embed.in_channels != in_channels:
            new_conv = nn.Conv2d(
                in_channels=in_channels,
                out_channels=patch_embed.out_channels,
                kernel_size=patch_embed.kernel_size,
                stride=patch_embed.stride,
                padding=patch_embed.padding,
            )

            with torch.no_grad():
                new_conv.weight[:, :3, :, :] = patch_embed.weight
                new_conv.weight[:, 3:6, :, :] = patch_embed.weight
                new_conv.weight[:, 6:7, :, :] = patch_embed.weight[:, 0:1, :, :]
                if patch_embed.bias is not None:
                    new_conv.bias = patch_embed.bias

            self.model.model.pixel_level_module.encoder.embeddings.patch_embeddings.projection = new_conv

    def forward(self, x):

        h, w = x.shape[-2:]
        target_h = ((h - 1) // 28 + 1) * 28
        target_w = ((w - 1) // 28 + 1) * 28

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
        sem_seg = F.interpolate(
            sem_seg, size=(h, w), mode="bilinear", align_corners=False
        )

        return sem_seg


class HF_OneFormer_ConvNeXt(nn.Module):
    def __init__(self, in_channels=7, num_classes=2, backbone="convnext-tiny"):
        super().__init__()

        config = OneFormerConfig(
            num_labels=num_classes,
            ignore_mismatched_sizes=True,
            backbone_config={
                "model_type": "convnext",
                "num_channels": in_channels,
                "depths": [3, 3, 9, 3],
                "hidden_sizes": [96, 192, 384, 768],
                "out_indices": [0, 1, 2, 3],
                "drop_path_rate": 0.1,
            },
            encoder_feed_forward_network_dim=1024,
            dim_feedforward=2048,
        )

        self.model = OneFormerForUniversalSegmentation(config)

        public_convnext_repo = "facebook/convnext-tiny-224"

        public_convnext = ConvNextModel.from_pretrained(
            public_convnext_repo, revision="refs/pr/4"
        )

        pretrained_state_dict = public_convnext.state_dict()
        old_weight = pretrained_state_dict["embeddings.patch_embeddings.weight"]

        if old_weight.shape[1] != in_channels:
            new_weight = torch.zeros(
                (
                    old_weight.shape[0],
                    in_channels,
                    old_weight.shape[2],
                    old_weight.shape[3],
                ),
                dtype=old_weight.dtype,
                device=old_weight.device,
            )
            with torch.no_grad():
                new_weight[:, :3, :, :] = old_weight
                new_weight[:, 3:6, :, :] = old_weight
                new_weight[:, 6:7, :, :] = old_weight[:, 0:1, :, :]

            pretrained_state_dict["embeddings.patch_embeddings.weight"] = new_weight

        msg = self.model.model.pixel_level_module.encoder.load_state_dict(
            pretrained_state_dict, strict=False
        )
        self.tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
        task_tokens = self.tokenizer(
            ["the task is semantic"],
            padding="max_length",
            max_length=77,
            return_tensors="pt",
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
        sem_seg = F.interpolate(
            sem_seg, size=(h, w), mode="bilinear", align_corners=False
        )

        return sem_seg
