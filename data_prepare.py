import os
os.environ["OPENCV_LOG_LEVEL"] = "ERROR"
import numpy as np
import tifffile as tiff
import cv2
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
from skimage.morphology import dilation, disk
import tifffile
import pandas as pd
from tqdm import tqdm
from pathlib import Path
import os
import numpy as np
import tifffile as tiff
import matplotlib.pyplot as plt
import cv2

# ================= 配置区域 =================

# 子集名称
SUBSETS = ['train', 'val', 'test']

# 文件夹结构名称
IMG_DIR_NAME = 'images'  # 你的图像文件夹名
MASK_DIR_NAME = 'masks'  # 你的标签文件夹名

# 异常值定义 (Float32 极小值)
ANOMALY_THRESHOLD = -1e30 
# 需要检测异常的波段索引 (0-based, 第2波段为1)
TARGET_BAND_IDX = 1 
# ===========================================
def normalize(img):
    """将图像归一化到 0-255"""
    img = np.nan_to_num(img)
    if img.max() - img.min() == 0:
        return img
    return ((img - img.min()) / (img.max() - img.min()) * 255).astype(np.uint8)

def get_mask_edges(mask):
    """提取Mask的轮廓边缘"""
    # 膨胀一下mask然后减去原mask得到边界，或者直接用Canny
    # 这里用Canny提取二值图的边缘
    edges = cv2.Canny(mask.astype(np.uint8) * 255, 100, 200)
    return edges > 0

def calculate_edge_overlap(band_img, mask_edges):
    """
    计算波段边缘与Mask边缘的重合度
    """
    # 1. 对波段进行 Canny 边缘检测
    # 由于不同波段动态范围不同，自适应计算阈值
    v = np.median(band_img)
    sigma = 0.33
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    
    band_edges = cv2.Canny(band_img, lower, upper)
    band_edges_bool = band_edges > 0
    
    # 2. 为了容忍轻微的像素偏移，对Mask边缘进行微小的膨胀 (1像素半径)
    # 意味着如果波段边缘落在Mask轮廓附近1像素内，也算匹配
    mask_edges_dilated = dilation(mask_edges, disk(1))
    
    # 3. 计算重合 (Intersection)
    # 只关心 Mask 边缘存在的地方，波段是否也检测到了边缘
    intersection = np.logical_and(band_edges_bool, mask_edges_dilated)
    
    # 4. 召回率风格的指标：Mask的边缘有多少被波段检测到了？
    if np.sum(mask_edges) == 0:
        return 0.0
    
    score = np.sum(intersection) / np.sum(mask_edges)
    return score

def visualize_band2_and_edges(img_dir, output_dir, num_samples=5):
    """
    可视化 Band 2 (Slope) 及其边缘特征
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    img_files = [f for f in os.listdir(img_dir) if f.endswith(('.tif', '.tiff'))]
    # img_files = img_files[:num_samples] # 只看前几张
    img_files = ['col00070_row00028.tif', 'col00071_row00028.tif', 'col00078_row00028.tif', 'col00079_row00028.tif', 'col00083_row00020.tif', 'col00083_row00021.tif']
    print(f"正在生成可视化结果到: {output_dir}")

    for img_file in img_files:
        img_path = os.path.join(img_dir, img_file)
        save_path = os.path.join(output_dir, f"vis_{img_file}.jpg")
        
        try:
            # 1. 读取数据
            img = tiff.imread(img_path)
            
            # 维度调整 (C, H, W) -> (H, W, C)
            if img.ndim == 3 and img.shape[0] < img.shape[2]: 
                img = np.transpose(img, (1, 2, 0))
            
            # 提取 Band 2 (索引 1)
            # 如果是单通道图，就取本身
            if img.ndim == 3 and img.shape[2] >= 2:
                band2 = img[:, :, 1]
            else:
                band2 = img
                print(f"Warning: {img_file} 似乎不是多波段图，直接使用第一通道")

            # 2. 数据清洗 (处理 -inf, nan, <0)
            # 创建掩膜：标记无效值
            invalid_mask = np.isinf(band2) | np.isnan(band2) | (band2 < -1e6)
            
            # 提取有效数据用于计算统计量
            valid_data = band2[~invalid_mask]
            
            if len(valid_data) == 0:
                print(f"Skipping {img_file}: No valid data in Band 2")
                continue
                
            vmin, vmax = np.percentile(valid_data, 1), np.percentile(valid_data, 99)
            
            # 准备绘图数据 (无效值设为 nan 以便 matplotlib 显示为空白/特定颜色)
            plot_data = band2.copy()
            plot_data[invalid_mask] = np.nan
            
            # 3. 生成边缘图 (模拟第8波段逻辑)
            # 先归一化到 0-255 uint8
            norm_img = np.clip((band2 - np.min(valid_data)) / (np.max(valid_data) - np.min(valid_data)) * 255, 0, 255)
            norm_img = np.nan_to_num(norm_img).astype(np.uint8)
            
            # 自适应 Canny
            v = np.median(norm_img)
            sigma = 0.33
            lower = int(max(0, (1.0 - sigma) * v))
            upper = int(min(255, (1.0 + sigma) * v))
            edges = cv2.Canny(norm_img, lower, upper)
            # 膨胀让显示更清晰
            edges = cv2.dilate(edges, np.ones((3,3), np.uint8))

            # 4. 绘图 (Matplotlib)
            fig = plt.figure(figsize=(15, 5))
            
            # 子图1：原始 Slope (伪彩色)
            ax1 = plt.subplot(1, 3, 1)
            # 使用 'magma' 或 'inferno' 配色，地形感强
            # set_bad('gray') 将 nan 显示为灰色
            cmap = plt.cm.magma
            cmap.set_bad('gray', 1.0)
            im = ax1.imshow(plot_data, cmap=cmap, vmin=vmin, vmax=vmax)
            plt.colorbar(im, ax=ax1, fraction=0.046, pad=0.04)
            ax1.set_title(f"Band 2: Slope\n(Range: {vmin:.1f} - {vmax:.1f})")
            ax1.axis('off')
            
            # 子图2：边缘特征 (第8波段预览)
            ax2 = plt.subplot(1, 3, 2)
            ax2.imshow(edges, cmap='gray')
            ax2.set_title("Generated Edge Band\n(Based on Slope)")
            ax2.axis('off')
            
            # 子图3：直方图 (检查数据分布)
            ax3 = plt.subplot(1, 3, 3)
            ax3.hist(valid_data.flatten(), bins=50, color='orange', alpha=0.7)
            ax3.set_title("Pixel Value Distribution\n(Valid Pixels Only)")
            ax3.set_xlabel("Slope Value")
            ax3.set_ylabel("Count")
            
            # 保存
            plt.tight_layout()
            plt.savefig(save_path)
            plt.close()
            
        except Exception as e:
            print(f"Error processing {img_file}: {e}")
import os
import numpy as np
import tifffile as tiff
import matplotlib.pyplot as plt
import cv2

def visualize_anomaly_location(img_dir, output_dir, max_samples=10):
    """
    专门查找并可视化 -3.4028235e+38 异常值的分布区域
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    img_files = [f for f in os.listdir(img_dir) if f.endswith(('.tif', '.tiff'))]
    
    # 异常值阈值 (Float32 最小值)
    # 我们用小于 -1e30 来安全地捕获这个值
    ANOMALY_THRESHOLD = -1e30 
    
    count = 0
    print(f"正在扫描异常值 (-3.4028235e+38) 分布...")
    
    for img_file in img_files:
        if count >= max_samples:
            break
            
        img_path = os.path.join(img_dir, img_file)
        
        try:
            # 读取
            img = tiff.imread(img_path)
            
            # 维度调整 (H, W, C)
            if img.ndim == 3 and img.shape[0] < img.shape[2]: 
                img = np.transpose(img, (1, 2, 0))
            
            # 获取 Band 2 (Slope)
            if img.ndim == 3:
                band2 = img[:, :, 1] # 假设 Band 2 是 Index 1
            else:
                band2 = img
                
            # --- 检查是否包含异常值 ---
            if np.min(band2) > ANOMALY_THRESHOLD:
                # 如果最小值都大于 -1e30，说明这张图没有那个异常值，跳过
                continue
                
            count += 1
            print(f"发现异常值: {img_file}")
            
            # --- 制作掩膜 ---
            # 1. 异常值掩膜 (Mask): True where value is bad
            anomaly_mask = band2 < ANOMALY_THRESHOLD

            
            # 2. 有效值背景 (Background): 用于显示地形上下文
            valid_data = band2.copy()
            # 将异常值临时替换为 NaN 或者 有效值的最小值，以免破坏归一化
            valid_min = np.min(band2[~anomaly_mask]) if np.any(~anomaly_mask) else 0
            valid_max = np.max(band2[~anomaly_mask]) if np.any(~anomaly_mask) else 1
            valid_data[anomaly_mask] = valid_min 
            
            # 归一化背景以便显示
            norm_bg = (valid_data - valid_min) / (valid_max - valid_min + 1e-6)
            norm_bg = np.clip(norm_bg * 255, 0, 255).astype(np.uint8)
            norm_bg = cv2.cvtColor(norm_bg, cv2.COLOR_GRAY2RGB) # 转为RGB以便叠加红色

            # --- 绘图 ---
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            
            # 图1: 地形背景 (忽略异常值后)
            axes[0].imshow(norm_bg)
            axes[0].set_title(f"Terrain Context\n(Anomaly replaced by min: {valid_min:.1f})")
            axes[0].axis('off')
            
            # 图2: 异常值位置 (二值图)
            axes[1].imshow(anomaly_mask, cmap='gray')
            axes[1].set_title("Anomaly Mask\n(White = -3.4e38 / NoData)")
            axes[1].axis('off')
            
            # 图3: 红色叠加警告
            # 创建一个纯红色的层
            overlay = norm_bg.copy()
            overlay[anomaly_mask] = [255, 0, 0] # 将异常区域涂红
            
            axes[2].imshow(overlay)
            axes[2].set_title(f"Red Overlay\n(Red = Bad Areas), Total Bad Pixels: {np.sum(anomaly_mask)}")
            axes[2].axis('off')
            
            plt.suptitle(f"File: {img_file}", fontsize=14)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f"anomaly_check_{img_file}.jpg"))
            plt.close()
            
        except Exception as e:
            print(f"Error: {e}")

    if count == 0:
        print("未在前几张图中发现异常值，或者阈值设置不匹配。")
    else:
        print(f"已生成 {count} 张可视化对比图，请检查输出文件夹。")

def fix_layer_by_copying(layer_2d, anomaly_mask_2d):
    """
    对单层 2D 数据执行相邻值复制修复
    """
    # 必须转换为 float 才能赋值 np.nan
    # 如果是 uint8 的 mask，这里也会变成 float
    working_layer = layer_2d.astype(np.float32)
    
    # 将异常位置设为 NaN
    working_layer[anomaly_mask_2d] = np.nan
    
    df = pd.DataFrame(working_layer)
    
    # 1. 左右填充 (处理纵向坏线) - 优先用左边的值填充
    df = df.ffill(axis=1).bfill(axis=1)
    # 2. 上下填充 (处理横向坏线) - 优先用上面的值填充
    df = df.ffill(axis=0).bfill(axis=0)
    
    return df.values

def get_mask_path(img_path, subset):
    """
    根据图像路径查找对应的 Mask 文件
    """
    # 假设文件名完全一致
    mask_name = img_path.name
    # 构建路径: root/subset/masks/filename.tif
    return img_path.parent.parent / MASK_DIR_NAME / mask_name

def process_dataset(SRC_ROOT, DST_ROOT, SUBSETS=SUBSETS):
    # 创建输出根目录
    Path(DST_ROOT).mkdir(parents=True, exist_ok=True)
    
    total_imgs = 0
    total_masks = 0
    
    for subset in SUBSETS:
        # 输入路径
        src_img_dir = Path(SRC_ROOT) / subset / IMG_DIR_NAME
        
        # 输出路径
        dst_img_dir = Path(DST_ROOT) / subset / IMG_DIR_NAME
        dst_mask_dir = Path(DST_ROOT) / subset / MASK_DIR_NAME
        
        # 创建子集目录
        dst_img_dir.mkdir(parents=True, exist_ok=True)
        if subset != 'test':
            dst_mask_dir.mkdir(parents=True, exist_ok=True)
        
        # 获取文件列表
        if not src_img_dir.exists():
            print(f"跳过 {subset} (目录不存在)")
            continue
            
        files = list(src_img_dir.glob("*.tif")) + list(src_img_dir.glob("*.tiff"))
        print(f"正在处理 {subset} 集，共 {len(files)} 个文件...")
        
        for img_path in tqdm(files):
            # 1. === 读取和处理图像 ===
            # tifffile 读取的数据形状可能是 (C, H, W) 或 (H, W, C)
            img_data = tifffile.imread(str(img_path))
            original_dtype = img_data.dtype
            
            # 简单的维度判断：通常波段数较少 (例如 < 20)，而宽高很大
            # 我们假设数据是 (Bands, Height, Width)
            # 如果是 (H, W, C)，需要根据你的数据情况调整索引方式
            if img_data.ndim == 3 and img_data.shape[0] < img_data.shape[2]:
                # 认为是 (C, H, W) 格式
                band_idx = TARGET_BAND_IDX
                is_channel_first = True
                target_band = img_data[band_idx, :, :]
            elif img_data.ndim == 3:
                # 认为是 (H, W, C) 格式
                band_idx = TARGET_BAND_IDX
                is_channel_first = False
                target_band = img_data[:, :, band_idx]
            else:
                # 可能是单波段 (H, W)，如果不涉及多波段逻辑可跳过
                continue

            # 生成异常 Mask
            anomaly_mask = target_band < ANOMALY_THRESHOLD
            has_anomaly = np.any(anomaly_mask)
            
            if has_anomaly:
                # 修复图像的该波段
                fixed_band = fix_layer_by_copying(target_band, anomaly_mask)
                
                # 写回原数组
                if is_channel_first:
                    img_data[band_idx, :, :] = fixed_band.astype(original_dtype)
                else:
                    img_data[:, :, band_idx] = fixed_band.astype(original_dtype)
            
            # 保存修复后的图像
            # photometric='minisblack' 是常见的灰度/多波段模式
            tifffile.imwrite(
                str(dst_img_dir / img_path.name), 
                img_data,
                photometric='minisblack'
            )
            total_imgs += 1

            # 2. === 处理 Mask (train/val) ===
            if subset == 'test':
                continue
                
            mask_path = get_mask_path(img_path, subset)
            
            if mask_path.exists():
                mask_data = tifffile.imread(str(mask_path))
                mask_dtype = mask_data.dtype
                
                # 如果图像有异常，Mask 必须在相同位置进行“复制”
                if has_anomaly:
                    # 判断 Mask 维度
                    if mask_data.ndim == 2:
                        # 单通道 Mask (H, W)
                        fixed_mask = fix_layer_by_copying(mask_data, anomaly_mask)
                        mask_data = fixed_mask.astype(mask_dtype)
                        
                    elif mask_data.ndim == 3:
                        # 多通道 Mask (C, H, W) - 虽然少见，但支持一下
                        # 假设 Channel First
                        for c in range(mask_data.shape[0]):
                            fixed_layer = fix_layer_by_copying(mask_data[c], anomaly_mask)
                            mask_data[c] = fixed_layer.astype(mask_dtype)
                
                # 保存 Mask (压缩以节省空间，Mask 通常有很多重复值)
                tifffile.imwrite(
                    str(dst_mask_dir / mask_path.name), 
                    mask_data,
                    photometric='minisblack',
                    compression='zlib'
                )
                total_masks += 1

    print(f"\n处理完成！")
    print(f"图像保存至: {DST_ROOT}")
    print(f"共处理图像: {total_imgs}, Mask: {total_masks}")


def process_dataset_test(src_img_dir, dst_img_dir):
    # 创建输出根目录
    Path(DST_ROOT).mkdir(parents=True, exist_ok=True)
    
    total_imgs = 0
    total_masks = 0
    
    src_img_dir = Path(src_img_dir)
    dst_img_dir = Path(dst_img_dir)
    dst_img_dir.mkdir(parents=True, exist_ok=True)
            
    files = list(src_img_dir.glob("*.tif")) + list(src_img_dir.glob("*.tiff"))
    print(f"正在处理 {src_img_dir.name} 集，共 {len(files)} 个文件...")
    
    for img_path in tqdm(files):
        # 1. === 读取和处理图像 ===
        # tifffile 读取的数据形状可能是 (C, H, W) 或 (H, W, C)
        img_data = tifffile.imread(str(img_path))
        original_dtype = img_data.dtype
        
        # 简单的维度判断：通常波段数较少 (例如 < 20)，而宽高很大
        # 我们假设数据是 (Bands, Height, Width)
        # 如果是 (H, W, C)，需要根据你的数据情况调整索引方式
        if img_data.ndim == 3 and img_data.shape[0] < img_data.shape[2]:
            # 认为是 (C, H, W) 格式
            band_idx = TARGET_BAND_IDX
            is_channel_first = True
            target_band = img_data[band_idx, :, :]
        elif img_data.ndim == 3:
            # 认为是 (H, W, C) 格式
            band_idx = TARGET_BAND_IDX
            is_channel_first = False
            target_band = img_data[:, :, band_idx]
        else:
            # 可能是单波段 (H, W)，如果不涉及多波段逻辑可跳过
            continue

        # 生成异常 Mask
        anomaly_mask = target_band < ANOMALY_THRESHOLD
        has_anomaly = np.any(anomaly_mask)
        
        if has_anomaly:
            # 修复图像的该波段
            fixed_band = fix_layer_by_copying(target_band, anomaly_mask)
            
            # 写回原数组
            if is_channel_first:
                img_data[band_idx, :, :] = fixed_band.astype(original_dtype)
            else:
                img_data[:, :, band_idx] = fixed_band.astype(original_dtype)
        
        # 保存修复后的图像
        # photometric='minisblack' 是常见的灰度/多波段模式
        tifffile.imwrite(
            str(dst_img_dir / img_path.name), 
            img_data,
            photometric='minisblack'
        )
        total_imgs += 1


    print(f"\n处理完成！")
    print(f"图像保存至: {dst_img_dir}")
    print(f"共处理图像: {total_imgs}, Mask: {total_masks}")

import os
import numpy as np
import tifffile as tiff
from tqdm import tqdm

def sta_ms(folder_list):
    """
    统计一个或多个文件夹下的 TIFF 图像均值和标准差。
    支持自动过滤无效值 (NaN, -Inf, NoData)。
    
    Args:
        folder_list (list or str): 文件夹路径列表，例如 ['train/imgs', 'val/imgs']
                                   也可以是单个字符串路径。
    """
    # 1. 如果输入是单个字符串，转为列表
    if isinstance(folder_list, str):
        folder_list = [folder_list]
    
    # 2. 收集所有图片路径
    all_img_paths = []
    print("Scanning folders...")
    for folder in folder_list:
        if not os.path.exists(folder):
            print(f"Warning: Folder not found: {folder}")
            continue
            
        files = [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith(('.tif', '.tiff'))]
        all_img_paths.extend(files)
        print(f"  - Found {len(files)} images in: {folder}")
    
    if not all_img_paths:
        print("Error: No images found in any provided folders.")
        return

    print(f"Total images to process: {len(all_img_paths)}")

    # 3. 初始化 (读取第一张图来确定通道数)
    try:
        first_img = tiff.imread(all_img_paths[0])
    except Exception as e:
        print(f"Error reading first image: {e}")
        return

    # 维度判断逻辑
    if first_img.ndim == 3 and first_img.shape[0] < first_img.shape[1] and first_img.shape[0] < first_img.shape[2]:
        channels = first_img.shape[0]
        is_channel_first = True
    elif first_img.ndim == 3:
        channels = first_img.shape[2]
        is_channel_first = False
    else:
        channels = 1
        is_channel_first = False

    print(f"Detected {channels} channels. Starting calculation...")

    # 初始化累加器 (Float64 防止溢出)
    channel_sum = np.zeros(channels, dtype=np.float64)
    channel_sq_sum = np.zeros(channels, dtype=np.float64)
    channel_pixel_count = np.zeros(channels, dtype=np.float64)

    # 4. 主循环：遍历所有图片
    for img_path in tqdm(all_img_paths, unit="img"):
        try:
            img = tiff.imread(img_path).astype(np.float32)
            
            # --- 数据清洗 (关键步骤) ---
            # 1. 处理 Inf
            img[np.isinf(img)] = np.nan
            # 2. 处理极小值 NoData (如 -3.4e38, -9999 等)
            img[img < -1e6] = np.nan
            
            # 3. (可选) 如果明确知道 Band 2 (Slope) 不应小于 0，可以加这行：
            # if channels >= 2:
            #     # 假设 Band 2 是索引 1
            #     if is_channel_first: img[1][img[1] < 0] = np.nan
            #     else: img[:,:,1][img[:,:,1] < 0] = np.nan

            # 调整维度 -> (Pixels, Channels)
            if is_channel_first:
                # (C, H, W) -> (C, H*W) -> (H*W, C)
                img_flat = img.reshape(channels, -1).T
            else:
                if channels > 1:
                    img_flat = img.reshape(-1, channels)
                else:
                    img_flat = img.reshape(-1, 1)

            # --- 累加 ---
            # 找出有效值 (非 NaN)
            valid_mask = ~np.isnan(img_flat)
            
            # 将 NaN 替换为 0 以便求和
            img_safe = np.nan_to_num(img_flat)
            
            # 累加 Sum
            channel_sum += np.sum(img_safe, axis=0)
            # 累加 Square Sum (注意：0的平方是0，不影响)
            channel_sq_sum += np.sum(img_safe ** 2, axis=0)
            # 累加 有效像素计数
            channel_pixel_count += np.sum(valid_mask, axis=0)

        except Exception as e:
            print(f"Error processing {img_path}: {e}")
            continue

    # 5. 计算结果
    # 防止除以 0
    channel_pixel_count[channel_pixel_count == 0] = 1.0 
    
    mean = channel_sum / channel_pixel_count
    std = np.sqrt((channel_sq_sum / channel_pixel_count) - (mean ** 2))

    print("\n" + "="*50)
    print(f"Statistics over {len(folder_list)} folders")
    print("="*50)
    print(f"Total Valid Pixels Processed (Band 1): {int(channel_pixel_count[0])}")
    print("-" * 20)
    
    print("Copy-Paste Format:")
    print(f"mean = {mean.tolist()}")
    print(f"std  = {std.tolist()}")
    
    print("-" * 20)
    print("Detailed:")
    for i in range(channels):
        print(f"Band {i+1}: Mean={mean[i]:.4f}, Std={std[i]:.4f}")


def analyze_band_importance(img_dir, mask_dir):
    img_files = [f for f in os.listdir(img_dir) if f.endswith('.tif')]
    
    # 定义波段名称 (根据你的数据集描述)
    band_names = [
        "Band 1: Thermal Inertia",
        "Band 2: Slope",
        "Band 3: DEM",
        "Band 4: Grayscale",
        "Band 5: Red",
        "Band 6: Green",
        "Band 7: Blue"
    ]
    
    results = {name: {'edge_score': [], 'correlation': [], 'mi': []} for name in band_names}
    
    print(f"正在分析 {len(img_files)} 张图像...")
    
    for img_file in tqdm(img_files[:100]): # 为了演示速度，这里只取前100张，实际使用建议跑全量
        img_path = os.path.join(img_dir, img_file)
        mask_path = os.path.join(mask_dir, img_file)
        
        if not os.path.exists(mask_path): continue
        
        # 读取数据
        try:
            img_data = tiff.imread(img_path)
            if img_data.shape[0] == 7: img_data = np.transpose(img_data, (1, 2, 0)) # (H,W,C)
            
            mask_data = tiff.imread(mask_path)
            if mask_data.ndim == 3: mask_data = mask_data[:,:,0] # 确保是2D
            mask_data = (mask_data > 0).astype(np.uint8) # 二值化 0, 1
            
            # 如果mask是空的（没有滑坡），跳过边缘分析
            has_landslide = np.sum(mask_data) > 0
            mask_edges = get_mask_edges(mask_data) if has_landslide else None
            
            mask_flat = mask_data.flatten()
            
            for i, name in enumerate(band_names):
                band = img_data[:, :, i]
                band_norm = normalize(band) # 归一化用于边缘检测
                band_flat = band.flatten()
                
                # Metric 1: 边缘重合度 (Edge Overlap)
                # 只有当图片里有滑坡时才计算
                if has_landslide:
                    edge_score = calculate_edge_overlap(band_norm, mask_edges)
                    results[name]['edge_score'].append(edge_score)
                
                # Metric 2: 相关性 (Correlation)
                # 计算像素值与标签(0/1)的相关系数
                corr = np.corrcoef(band_flat, mask_flat)[0, 1]
                if not np.isnan(corr):
                    results[name]['correlation'].append(abs(corr)) # 取绝对值，关注相关程度
                
                # Metric 3: 互信息 (Mutual Information) - 简化版（基于直方图太慢，这里用score）
                # 为了速度，对数据进行分箱 (Binning)
                # mi = mutual_info_score(mask_flat, np.digitize(band_flat, bins=10))
                # results[name]['mi'].append(mi)

        except Exception as e:
            print(f"Error processing {img_file}: {e}")
            continue

    # --- 汇总结果并可视化 ---
    summary = []
    for name in band_names:
        avg_edge = np.mean(results[name]['edge_score']) if results[name]['edge_score'] else 0
        avg_corr = np.mean(results[name]['correlation']) if results[name]['correlation'] else 0
        summary.append({
            "Band": name,
            "Edge Alignment Score": avg_edge,
            "Pixel Correlation": avg_corr
        })
    
    df = pd.DataFrame(summary)
    
    # 绘图
    plt.figure(figsize=(12, 6))
    
    # 图1: 边缘重合度
    plt.subplot(1, 2, 1)
    sns.barplot(data=df, y="Band", x="Edge Alignment Score", palette="viridis")
    plt.title("Method 1: Edge Alignment (Boundary Match)")
    plt.xlabel("Average Overlap Score")
    
    # 图2: 像素相关性
    plt.subplot(1, 2, 2)
    sns.barplot(data=df, y="Band", x="Pixel Correlation", palette="magma")
    plt.title("Method 2: Pixel Intensity Correlation")
    plt.xlabel("Absolute Correlation Coefficient")
    
    plt.tight_layout()
    plt.show()
    
    print("\n分析结果数值:")
    print(df)
    
    # 找出最佳波段
    best_edge = df.loc[df['Edge Alignment Score'].idxmax()]
    print(f"\n结论:\n边界最清晰的波段是: {best_edge['Band']}")



if __name__ == '__main__':
    pass
    TRAIN_SRC = '/scrinvme/huilin/bdd/cp_data/mars_seg/Mars_LSc_2025_dataset_1st_phase'
    TRAIN_REPAIR = '/scrinvme/huilin/bdd/cp_data/mars_seg/Mars_LSc_2025_dataset_1st_phase_updateB2'
    DATA_ROOT = TRAIN_REPAIR


    # process_dataset(TRAIN_SRC, TRAIN_REPAIR)


    # sta_ms(os.path.join(DATA_ROOT, "train/images"))
    # sta_ms(os.path.join(DATA_ROOT, "test/images"))

    # analyze_band_importance(os.path.join(DATA_ROOT, "train/images"), os.path.join(DATA_ROOT, "train/masks"))
