import os
import numpy as np
import tifffile as tiff
import matplotlib.pyplot as plt
import cv2

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
def get_mask_edges(mask):
    """提取Mask的轮廓边缘"""
    # 膨胀一下mask然后减去原mask得到边界，或者直接用Canny
    # 这里用Canny提取二值图的边缘
    edges = cv2.Canny(mask.astype(np.uint8) * 255, 100, 200)
    return edges > 0
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


if __name__ == "__main__":
    # 配置您的路径
    # IMG_DIR = r"/scrinvme/huilin/bdd/cp_data/mars_seg/Mars_LSc_2025_dataset_1st_phase/train/images"
    # OUTPUT_DIR = r"vis_result_band2"
    
    # visualize_band2_and_edges(IMG_DIR, OUTPUT_DIR)

    dataset_dir = r"/scrinvme/huilin/bdd/cp_data/mars_seg/Mars_LSc_2025_dataset_1st_phase/train/images"
    output_vis = r"vis_anomaly_check"
    
    visualize_anomaly_location(dataset_dir, output_vis)

    