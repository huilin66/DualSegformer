from datetime import datetime
import os
import argparse
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import tifffile
import numpy as np
import zipfile
import shutil
import sys
from networks import get_model

from dataset import MartianLandslideDataset, MarsSegDataset, MartianLandslideDataset3Band
# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))



def generate_submission(model_name, checkpoint_path, data_root, output_zip_path='submission.zip', remove=True, augs=None):
    # 1. Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 2. Model
    print("Initializing model...")
    model = get_model(model_name, in_channels=7, num_classes=2).to(device)
    
    # 3. Load Weights
    print(f"Loading checkpoint: {checkpoint_path}")
    if not os.path.exists(checkpoint_path):
        print(f"Error: Checkpoint not found at {checkpoint_path}")
        print("Please train the model first using 'python train.py'")
        return
        
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Handle state dict
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    elif isinstance(checkpoint, dict):
        model.load_state_dict(checkpoint)
    else:
        # If it's the full model object
        model.load_state_dict(checkpoint.state_dict())
        
    model.eval()
    
    # 4. Data
    print(f"Loading test data from {data_root}...")
    # Check if test directory exists
    test_dir = os.path.join(data_root, 'test', 'images')
    if not os.path.exists(test_dir):
        print(f"Error: Test directory not found at {test_dir}")
        return

    test_dataset = MarsSegDataset(root_dir=data_root, split='test')
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4)
    
    # 5. Inference
    temp_dir = 'temp_submission_masks'
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)
    os.makedirs(temp_dir, exist_ok=True)
    
    print(f"Starting inference on {len(test_dataset)} images...")
    
    with torch.no_grad():
        for i, (images, image_names) in enumerate(tqdm(test_loader, desc="Inferencing")):
            images = images.to(device)
            orig_h, orig_w = images.shape[2], images.shape[3]

            final_logits = []
            for scale in augs['ms_scale']:
                if scale != 1.0:
                    h = int(orig_h * scale + 0.5)
                    w = int(orig_w * scale + 0.5)
                    curr_images = F.interpolate(images, (h, w), mode='bilinear')
                else:
                    curr_images = images
                images_aug = [curr_images]
                if augs['flip_v']:
                    images_aug.append(torch.flip(curr_images, dims=[2]))
                if augs['flip_h']:
                    images_aug.append(torch.flip(curr_images, dims=[3]))
                if augs['flip_vh']:
                    images_aug.append(torch.flip(curr_images, dims=[2, 3]))
                if augs['rotate_90']:
                    images_aug.append(torch.rot90(curr_images, k=1, dims=[2, 3]))
                if augs['rotate_180']:
                    images_aug.append(torch.rot90(curr_images, k=2, dims=[2, 3]))
                if augs['rotate_270']:
                    images_aug.append(torch.rot90(curr_images, k=3, dims=[2, 3]))
                images_aug = torch.cat(images_aug, dim=0)

                logit = model(images_aug)

                if scale != 1.0:
                    logit = F.interpolate(logit, (orig_h, orig_w), mode='bilinear')
                logits = [logit[0]]
                pred_idx = 1 # 修复Bug 1: 从 1 开始追踪增强结果的索引
                # 逆向 TTA 还原 (注意使用的是 logit[pred_idx] 而不是 logit[0])
                if augs['flip_v']:
                    logits.append(torch.flip(logit[pred_idx], dims=[1]))
                    pred_idx += 1
                if augs['flip_h']:
                    logits.append(torch.flip(logit[pred_idx], dims=[2]))
                    pred_idx += 1
                if augs['flip_vh']:
                    logits.append(torch.flip(logit[pred_idx], dims=[1, 2]))
                    pred_idx += 1
                if augs['rotate_90']:
                    # 原图是 k=1 转的，还原需要 k=3 (相当于转回来)
                    logits.append(torch.rot90(logit[pred_idx], k=3, dims=[1, 2]))
                    pred_idx += 1
                if augs['rotate_180']:
                    logits.append(torch.rot90(logit[pred_idx], k=2, dims=[1, 2]))
                    pred_idx += 1
                if augs['rotate_270']:
                    # 原图是 k=3 转的，还原需要 k=1
                    logits.append(torch.rot90(logit[pred_idx], k=1, dims=[1, 2]))
                    pred_idx += 1
                final_logits.extend(logits)
            output = torch.mean(torch.stack(final_logits, dim=0), dim=0, keepdim=True)
            pred_mask = torch.argmax(output, dim=1) # [1, 128, 128]
            pred_mask_np = pred_mask.squeeze(0).cpu().numpy().astype(np.uint8)
            
            # Debug: Check values for first image
            if i == 0:
                print(f"DEBUG: Sample output values: {np.unique(pred_mask_np)} (Should be [0, 1] for binary)")

            # Save
            filename = image_names[0]
            save_path = os.path.join(temp_dir, filename)
            tifffile.imwrite(save_path, pred_mask_np)

    # 6. Zip
    output_zip_path = datetime.now().strftime("submission_%Y%m%d_%H%M%S.zip")
    output_zip_path = os.path.join('submission', output_zip_path)
    os.makedirs('submission', exist_ok=True)
    print(f"Creating archive: {output_zip_path}")
    with zipfile.ZipFile(output_zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, _, files in os.walk(temp_dir):
            for file in files:
                if file.endswith('.tif'):
                    file_path = os.path.join(root, file)
                    zipf.write(file_path, arcname=file)
                
    # 7. Cleanup
    if remove:
        print("Cleaning up temporary files...")
        shutil.rmtree(temp_dir)
    print(f"Submission generated successfully: {os.path.abspath(output_zip_path)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate Submission Zip')
    # parser.add_argument('--model_name', type=str, default='segformer_convnexttiny', help='Model name')
    # parser.add_argument('--checkpoint', type=str, default='/localnvme/project/M3LSNet/outputs/20260206_120006/checkpoints/best.pth', help='Path to .pth checkpoint')
    # parser.add_argument('--root', type=str, default='/scrinvme/huilin/bdd/cp_data/mars_seg/Mars_LSc_2025_dataset_1st_phase', help='Path to Dataset root')
    
    # args = parser.parse_args()
    # generate_submission(args.model_name, args.checkpoint, args.root)
    root = '/scrinvme/huilin/bdd/cp_data/mars_seg/Mars_LSc_2025_test_data_2nd_phase_updateB2'
    outputs = r'/localnvme/project/M3LSNet/outputs'
    result_dict = {
        # 'segformer_convnexttiny': '20260209_172414',
        # 'mask2former-swintiny': '20260206_184314',
        # 'oneformer-swintiny': '20260206_191244',
        # 'segformer_mitb2': '20260206_202216',
        # 'deeplabv3p_convnexttiny': '20260206_215804',
        # 'ocrnet_hrnet_w48': '20260206_165946',
        # 'unet_resnest50': '20260206_222542',
        # 'unet_mitb2': '20260206_225236',
        # 'unet_pvtb2': '20260206_230604',
        # 'unet_convnexttinyv2': '20260223_201849',
        # 'dual_unet_convnexttiny_chv1_add': '20260223_205116',
        # 'dual_unet_convnexttiny_chv2_add': '20260223_210831',
        # 'segformer_convnexttiny': '20260226_203334',
        # 'upernet_convnexttiny': '20260224_100336',
        # 'unet_convnexttiny': '20260224_101937',
        # 'segformer_convnexttiny': '20260224_105811',
        # 'upernet_convnexttiny': '20260224_111536',
        # 'unet_convnexttiny': '20260224_121857',
        # 'dual_unetformer_convnexttiny_chv2_cat': '20260227_174943',
        # 'dual_unetformer_convnexttiny_chv2_att': '20260227_200723',
        # 'dual_unetformer_convnexttiny_chv2_moe': '20260227_211256',
        # 'segformer_convnexttinyv2': '20260227_213557',
        # 'segformer_convnextbase': '20260227_221840',
        # 'segformer_convnexttiny': '20260301_220437',
        # 'upernet_convnexttiny': '20260301_220615',
        # 'segformer_convnextsmall': '20260301_221725',
        # 'upernet_convnextsmall': '20260301_221923',
        # 'dual_segformer_convnexttiny_chv1_add': '20260301_223101',
        # 'dual_upernet_convnexttiny_chv1_add': '20260301_223250',
        # 'dual_segformer_convnextsmall_chv1_add': '20260301_224425',
        # 'dual_upernet_convnextsmall_chv1_add': '20260301_224618',
        # 'dual_segformer_convnexttiny_chv2_add': '20260301_225824',
        # 'dual_upernet_convnexttiny_chv2_add': '20260301_230020',
        # 'dual_segformer_convnextsmall_chv2_add': '20260301_231148',
        # 'dual_upernet_convnextsmall_chv2_add': '20260301_231357',

        # 'dual_segformer_convnextsmall_chv2_add': '20260302_121456',
        # 'dual_segformer_convnextsmall_chv3_add': '20260302_122655',
        # 'dual_segformer_convnextsmall_chv1_cat': '20260302_123856',
        # 'dual_segformer_convnextsmall_chv1_att': '20260302_125058',
        # 'dual_segformer_convnextsmall_chv1_moe': '20260302_130304',
        # 'dual_segformer_convnextbase_chv1_add': '20260302_131628',
        # 'dual_segformer_convnextlarge_chv1_add': '20260302_133033',

        # 'dual_segformer_convnexttiny_chv3_add': '20260302_150733',
        # 'dual_segformer_convnextv2tiny_chv1_add': '20260302_152732',
        'dual_segformer_convnexttiny_chv1_cat':'20260302_160613',
        'dual_segformer_convnexttiny_chv1_att':'20260302_161817',
        'dual_segformer_convnexttiny_chv1_moe':'20260302_160619',
        'dual_segformer_convnexttiny_chv1_moev2':'20260302_161951',
    }
    augs = {
        'flip_v': True,
        'flip_h': True,
        'flip_vh': True,
        'rotate_90': True,
        'rotate_180': True,
        'rotate_270': True,
        'ms_scale': [1.0,]
    }
    for k, v in result_dict.items():
        generate_submission(k, os.path.join(outputs, v, 'checkpoints', 'last.pth'), root, remove=False, augs=augs)
