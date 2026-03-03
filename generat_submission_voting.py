import os
import gc
import glob
import shutil
import zipfile
import argparse
from datetime import datetime

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import tifffile
import numpy as np
import sys

# 添加当前目录到环境变量，以防找不到 networks 和 dataset
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from networks import get_model
from dataset import MarsSegDataset,MarsSegDatasetInferV1, MarsSegDatasetInferV2

def run_inference_for_model(model_name, checkpoint_path, data_root, temp_dir, augs, device):
    """
    阶段一：为单个模型运行推断（包含TTA），并保存 概率图(.npy) 和 掩码图(.tif)
    """
    print(f"\n{'='*50}")
    print(f"🚀 Starting inference for model: {model_name}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"{'='*50}")

    # 1. 初始化模型
    model = get_model(model_name, in_channels=7, num_classes=2).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    elif isinstance(checkpoint, dict):
        model.load_state_dict(checkpoint)
    else:
        model.load_state_dict(checkpoint.state_dict())
    model.eval()
    
    # 2. 准备数据
    test_dataset = MarsSegDatasetInferV2(root_dir=data_root, split='test')
    # test_dataset = MartianLandslideDatasetV3(root_dir=data_root, split='test')
    # test_dataset = MartianLandslideDatasetV4(root_dir=data_root, split='test')
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4)
    
    # 3. 创建临时文件夹（按照模型名区分）
    model_temp_dir = os.path.join(temp_dir, model_name)
    prob_dir = os.path.join(model_temp_dir, 'probs')
    mask_dir = os.path.join(model_temp_dir, 'masks')
    os.makedirs(prob_dir, exist_ok=True)
    os.makedirs(mask_dir, exist_ok=True)

    # 4. 推理
    with torch.no_grad():
        for i, (images, image_names) in enumerate(tqdm(test_loader, desc=f"Inferencing {model_name}")):
            images = images.to(device)
            orig_h, orig_w = images.shape[2], images.shape[3]
            filename = image_names[0]

            final_logits = []
            for scale in augs['ms_scale']:
                # 放缩
                if scale != 1.0:
                    h, w = int(orig_h * scale + 0.5), int(orig_w * scale + 0.5)
                    curr_images = F.interpolate(images, (h, w), mode='bilinear')
                else:
                    curr_images = images
                
                images_aug = [curr_images]
                
                # 正向 TTA
                if augs['flip_v']: images_aug.append(torch.flip(curr_images, dims=[2]))
                if augs['flip_h']: images_aug.append(torch.flip(curr_images, dims=[3]))
                if augs['flip_vh']: images_aug.append(torch.flip(curr_images, dims=[2, 3]))
                if augs['rotate_90']: images_aug.append(torch.rot90(curr_images, k=1, dims=[2, 3]))
                if augs['rotate_180']: images_aug.append(torch.rot90(curr_images, k=2, dims=[2, 3]))
                if augs['rotate_270']: images_aug.append(torch.rot90(curr_images, k=3, dims=[2, 3]))
                
                images_aug = torch.cat(images_aug, dim=0)
                logit = model(images_aug)

                # 还原尺寸
                if scale != 1.0:
                    logit = F.interpolate(logit, (orig_h, orig_w), mode='bilinear')
                
                logits = [logit[0]]
                pred_idx = 1
                
                # 逆向 TTA 还原 (已修复 3D Tensor 的维度问题 dims=[1, 2])
                if augs['flip_v']:
                    logits.append(torch.flip(logit[pred_idx], dims=[1])); pred_idx += 1
                if augs['flip_h']:
                    logits.append(torch.flip(logit[pred_idx], dims=[2])); pred_idx += 1
                if augs['flip_vh']:
                    logits.append(torch.flip(logit[pred_idx], dims=[1, 2])); pred_idx += 1
                if augs['rotate_90']:
                    logits.append(torch.rot90(logit[pred_idx], k=3, dims=[1, 2])); pred_idx += 1
                if augs['rotate_180']:
                    logits.append(torch.rot90(logit[pred_idx], k=2, dims=[1, 2])); pred_idx += 1
                if augs['rotate_270']:
                    logits.append(torch.rot90(logit[pred_idx], k=1, dims=[1, 2])); pred_idx += 1
                
                final_logits.extend(logits)

            # TTA 集成：求均值
            output = torch.mean(torch.stack(final_logits, dim=0), dim=0, keepdim=True)
            
            # --- 核心修改点 ---
            # 1. 计算概率图并保存为 float16 (用于软投票)
            prob_map = F.softmax(output, dim=1)[0, 1, :, :].cpu().numpy().astype(np.float16)
            np.save(os.path.join(prob_dir, filename.replace('.tif', '.npy')), prob_map)
            
            # 2. 计算 0/1 掩码并保存 (用于硬投票)
            pred_mask = torch.argmax(output, dim=1).squeeze(0).cpu().numpy().astype(np.uint8)
            tifffile.imwrite(os.path.join(mask_dir, filename), pred_mask)
            
    # 释放显存，防止下一个模型 OOM
    del model
    torch.cuda.empty_cache()
    gc.collect()

def generate_ensembles(temp_dir, model_names, output_base, weights=None):
    """
    阶段二：读取临时文件夹的结果，进行软投票和硬投票
    """
    print(f"\n{'='*50}")
    print(f"🧩 Starting Ensemble (Hard Voting & Soft Voting)...")
    print(f"{'='*50}")

    hard_vote_dir = os.path.join(output_base, 'hard_voting_masks')
    soft_vote_dir = os.path.join(output_base, 'soft_voting_masks')
    os.makedirs(hard_vote_dir, exist_ok=True)
    os.makedirs(soft_vote_dir, exist_ok=True)

    # 权重处理
    num_models = len(model_names)
    if weights is None:
        weights = [1.0] * num_models
    weights = np.array(weights) / np.sum(weights)

    # 获取所有图片文件名
    base_prob_dir = os.path.join(temp_dir, model_names[0], 'probs')
    filenames = [f.replace('.npy', '.tif') for f in os.listdir(base_prob_dir) if f.endswith('.npy')]

    for filename in tqdm(filenames, desc="Ensembling"):
        npy_name = filename.replace('.tif', '.npy')
        
        masks = []
        probs = []
        
        # 读取每个模型的结果
        for i, model_name in enumerate(model_names):
            # 读掩码
            mask_path = os.path.join(temp_dir, model_name, 'masks', filename)
            masks.append(tifffile.imread(mask_path))
            
            # 读概率
            prob_path = os.path.join(temp_dir, model_name, 'probs', npy_name)
            probs.append(np.load(prob_path) * weights[i])

        # --- 1. 硬投票逻辑 (Majority Vote) ---
        stacked_masks = np.stack(masks, axis=0)
        sum_mask = np.sum(stacked_masks, axis=0)
        hard_final = (sum_mask >= (num_models / 2.0)).astype(np.uint8)
        tifffile.imwrite(os.path.join(hard_vote_dir, filename), hard_final)

        # --- 2. 软投票逻辑 (Weighted Soft Vote) ---
        sum_prob = np.sum(probs, axis=0)
        soft_final = (sum_prob > 0.5).astype(np.uint8)
        tifffile.imwrite(os.path.join(soft_vote_dir, filename), soft_final)

    return hard_vote_dir, soft_vote_dir

def create_zip(source_dir, zip_filename):
    """将指定文件夹打包为 ZIP 文件"""
    print(f"📦 Creating zip archive: {zip_filename}")
    with zipfile.ZipFile(zip_filename, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, _, files in os.walk(source_dir):
            for file in files:
                if file.endswith('.tif'):
                    file_path = os.path.join(root, file)
                    zipf.write(file_path, arcname=file)
    print(f"✅ Saved: {zip_filename}")

def main():
    # ------------------ 基础配置区域 ------------------
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🖥️  Using device: {device}")

    # 数据集和输出路径
    # data_root = '/scrinvme/huilin/bdd/cp_data/mars_seg/Mars_LSc_2025_dataset_1st_phase_updateB2'
    data_root = '/scrinvme/huilin/bdd/cp_data/mars_seg/Mars_LSc_2025_test_data_2nd_phase_updateB2'
    outputs_dir = r'/localnvme/project/M3LSNet/outputs'
    
    # 临时文件和最终结果存放目录
    temp_dir = 'ensemble_temp_files'
    final_output_dir = 'submission_ensembles'
    os.makedirs(final_output_dir, exist_ok=True)

    # TTA 增强策略
    augs = {
        'flip_v': True,
        'flip_h': True,       # 建议开启，因为维度Bug已修复，翻转增强通常有效
        'flip_vh': True,
        'rotate_90': True,
        'rotate_180': True,
        'rotate_270': True,
        'ms_scale': [1.0, 1.5]     # 可以尝试 [0.75, 1.0, 1.25]
    }

    # 要参与集成的模型字典 {模型名: 实验时间戳}
    # 建议选取 3 个或 5 个模型，以防硬投票平局
    result_dict = {
        # # 'segformer_convnexttiny': '20260301_220437',
        # # 'upernet_convnexttiny': '20260301_220615',
        # 'segformer_convnextsmall': '20260301_221725',
        # # 'upernet_convnextsmall': '20260301_221923',
        # 'dual_segformer_convnexttiny_chv1_add': '20260301_223101',
        # 'dual_upernet_convnexttiny_chv1_add': '20260301_223250',
        # # 'dual_segformer_convnextsmall_chv1_add': '20260301_224425',
        # # 'dual_upernet_convnextsmall_chv1_add': '20260301_224618',
        # 'dual_segformer_convnexttiny_chv2_add': '20260301_225824',
        # # 'dual_upernet_convnexttiny_chv2_add': '20260301_230020',
        # 'dual_segformer_convnextsmall_chv2_add': '20260301_231148',
        # # 'dual_upernet_convnextsmall_chv2_add': '20260301_231357',

        # # # 'dual_unetformer_convnextsmall_chv1_add': "20260228_103424",

        # 'dual_segformer_convnextsmall_chv2_add': '20260302_121456',
        # 'dual_segformer_convnextsmall_chv3_add': '20260302_122655',
        # # 'dual_segformer_convnextsmall_chv1_cat': '20260302_123856',
        # # 'dual_segformer_convnextsmall_chv1_att': '20260302_125058',
        # # 'dual_segformer_convnextsmall_chv1_moe': '20260302_130304',
        # 'dual_segformer_convnextbase_chv1_add': '20260302_131628',
        # 'dual_segformer_convnextlarge_chv1_add': '20260302_133033',

        # # 'dual_segformer_convnexttiny_chv3_add': '20260302_150733',
        # # 'dual_segformer_convnextv2tiny_chv1_add': '20260302_152732',
        # # 'dual_segformer_convnexttiny_chv1_cat':'20260302_160613',
        # # 'dual_segformer_convnexttiny_chv1_att':'20260302_161817',
        # # 'dual_segformer_convnexttiny_chv1_moe':'20260302_160619',
        # 'dual_segformer_convnexttiny_chv1_moev2':'20260302_161951',


        # 'upernet_convnexttiny': '20260301_220615',

        # 'segformer_convnextsmall': '20260301_221725',
        'dual_segformer_convnexttiny_chv1_add': '20260301_223101',
        # 'dual_upernet_convnexttiny_chv1_add': '20260301_223250',
        # 'dual_segformer_convnexttiny_chv2_add': '20260301_225824',
        # 'dual_segformer_convnextsmall_chv2_add': '20260301_231148',


        # 'dual_segformer_convnextsmall_chv2_add': '20260302_121456',
        # 'dual_segformer_convnextbase_chv1_add': '20260302_131628',
        'dual_segformer_convnextlarge_chv1_add': '20260302_133033',
        # 'dual_segformer_convnexttiny_chv1_moev2':'20260302_161951',
    }

    # 针对软投票的权重设置（数量必须与上面的模型一致）W
    # 如果不知道怎么设，就写 [1.0, 1.0, 1.0]
    custom_weights = [1.0]*len(result_dict) 
    # --------------------------------------------------

    model_names = list(result_dict.keys())

    # 1. 串行执行所有模型的预测
    for model_name, timestamp in result_dict.items():
        checkpoint_path = os.path.join(outputs_dir, timestamp, 'checkpoints', 'last.pth')
        if not os.path.exists(checkpoint_path):
            print(f"⚠️ Warning: Checkpoint not found -> {checkpoint_path}")
            continue
        
        run_inference_for_model(model_name, checkpoint_path, data_root, temp_dir, augs, device)

    # 2. 生成集成结果 (软投票 & 硬投票)
    hard_dir, soft_dir = generate_ensembles(temp_dir, model_names, final_output_dir, weights=custom_weights)

    # 3. 分别打包成 Zip 供提交打榜
    time_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    hard_zip = os.path.join(final_output_dir, f"submission_HardVote_{time_str}.zip")
    soft_zip = os.path.join(final_output_dir, f"submission_SoftVote_{time_str}.zip")
    
    create_zip(hard_dir, hard_zip)
    create_zip(soft_dir, soft_zip)

    # 4. 自动清理临时文件 (可选)
    print("\n🧹 Cleaning up temporary files...")
    shutil.rmtree(temp_dir)
    print("🎉 All done! You can now submit the zip files.")

if __name__ == "__main__":
    main()