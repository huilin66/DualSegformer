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

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from dataset import MartianLandslideDataset, MarsSegDataset, MartianLandslideDataset3Band

def generate_submission(model_name, checkpoint_path, data_root, output_zip_path='submission.zip', remove=True):
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
            
            output = model(images)
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
    print(f"[{checkpoint_path.split('/')[-3]}] {model_name}: {os.path.basename(output_zip_path)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate Submission Zip')
    # parser.add_argument('--model_name', type=str, default='segformer_convnexttiny', help='Model name')
    # parser.add_argument('--checkpoint', type=str, default='/localnvme/project/M3LSNet/outputs/20260206_120006/checkpoints/best.pth', help='Path to .pth checkpoint')
    # parser.add_argument('--root', type=str, default='/scrinvme/huilin/bdd/cp_data/mars_seg/Mars_LSc_2025_dataset_1st_phase', help='Path to Dataset root')
    
    # args = parser.parse_args()
    # generate_submission(args.model_name, args.checkpoint, args.root)
    root = '/scrinvme/huilin/bdd/cp_data/mars_seg/Mars_LSc_2025_dataset_1st_phase_updateB2'
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
        # 'segformer_convnexttiny': '20260224_094741',
        # 'upernet_convnexttiny': '20260224_100336',
        # 'unet_convnexttiny': '20260224_101937',
        # 'segformer_convnexttiny': '20260224_105811',
        # 'upernet_convnexttiny': '20260224_111536',
        # 'unet_convnexttiny': '20260224_121857',
        # 'segformer_convnexttiny': '20260225_105952', 
        # 'upernet_mitb2': '20260206_211719', 
        # 'ocrnet_hrnet_w48': '20260226_193709',
        # 'segformer_convnexttiny': '20260226_203334',
        # 'upernet_convnexttiny': '20260226_205406',
        # 'unet_convnexttiny': '20260226_211515',

        # 'dual_upernet_convnexttiny_chv2_add': '20260226_224952',
        # 'dual_segformer_convnexttiny_chv2_cat': '20260227_124411',
        # 'segformer_convnexttiny': '20260226_203334',
        'dual_segformer_convnexttiny_chv2_add': '20260227_102956',
        'dual_segformer_convnexttiny_chv2_moe': '20260227_171848',
    }
    for k, v in result_dict.items():
        generate_submission(k, os.path.join(outputs, v, 'checkpoints', 'best.pth'), root, remove=False)
        # generate_submission(k, os.path.join(outputs, v, 'checkpoints', 'last.pth'), root, remove=False)