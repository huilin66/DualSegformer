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



def add_label(img, text, color=(255, 255, 255)):
    """在图片上方添加文字标签的辅助函数"""
    h, w = img.shape[:2]
    # 创建一个黑色背景条用于写字
    label_bg = np.zeros((30, w, 3), dtype=np.uint8)
    cv2.putText(label_bg, text, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 
                0.6, color, 1, cv2.LINE_AA)
    # 将文字条拼接到原图上方
    return cv2.vconcat([label_bg, img])

def to_bgr(img_gray):
    """将灰度图/二值图转换为 BGR 三通道图，方便拼接和显示"""
    if img_gray.ndim == 2:
        return cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)
    return img_gray


def normalize_to_uint8(data):
    """
    将任意范围的数值 (float/int) 归一化到 0-255 并转为 uint8，
    用于可视化 DEM、Slope、Thermal 等非图像数据。
    """
    data = data.astype(np.float32)
    min_val = np.min(data)
    max_val = np.max(data)
    
    # 避免除以0
    if max_val - min_val < 1e-5:
        return np.zeros_like(data, dtype=np.uint8)
        
    norm_data = (data - min_val) / (max_val - min_val) * 255.0
    return norm_data.astype(np.uint8)

def add_label_header(img, text, bg_color=(0, 0, 0), txt_color=(255, 255, 255)):
    """
    在图像上方添加一个文字标题栏（不遮挡原图内容）
    """
    h, w, c = img.shape
    header_h = 30 # 标题栏高度
    
    # 创建背景
    header = np.zeros((header_h, w, 3), dtype=np.uint8)
    header[:] = bg_color
    
    # 写入文字
    font_scale = 0.5
    thickness = 1
    (text_w, text_h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
    
    # 居中显示
    text_x = (w - text_w) // 2
    text_y = (header_h + text_h) // 2
    
    cv2.putText(header, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 
                font_scale, txt_color, thickness, cv2.LINE_AA)
    
    return cv2.vconcat([header, img])

def to_bgr(img):
    """确保图像是3通道BGR"""
    if len(img.shape) == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return img

def generate_comprehensive_vis(img_dir, vis_dir, gt_dir, pred_dir, output_dir):
    """
    生成 3x4 = 12 张图的综合可视化
    Row 1: B1(Thermal), B2(Slope), B3(DEM), B4(Gray)
    Row 2: B5(R), B6(G), B7(B), RGB_Composite
    Row 3: GT, Pred, Error_Map, Overlay_Vis
    """
    os.makedirs(output_dir, exist_ok=True)

    # 1. 获取文件列表 (以 GT 为锚点)
    gt_files = [f for f in os.listdir(gt_dir) if f.endswith(('.png', '.tif', '.tiff'))]
    
    if not gt_files:
        print("GT 文件夹为空")
        return

    # 文件名映射函数
    def get_file_map(directory):
        mapping = {}
        for f in os.listdir(directory):
            name = os.path.splitext(f)[0]
            mapping[name] = os.path.join(directory, f)
        return mapping

    img_map = get_file_map(img_dir)
    pred_map = get_file_map(pred_dir)

    for gt_file in tqdm(gt_files, desc="Generating VIS", unit="img"):
        base_name = os.path.splitext(gt_file)[0]
        
        if base_name not in img_map or base_name not in pred_map:
            continue
            
        gt_path = os.path.join(gt_dir, gt_file)
        img_path = img_map[base_name]
        pred_path = pred_map[base_name]
        vis_path = os.path.join(vis_dir, base_name + ".jpg")
        save_path = os.path.join(output_dir, base_name + "_12grid.jpg")

        # ===========================
        # 1. 读取数据
        # ===========================
        # 读取 7波段 TIFF (假设形状是 H,W,C 或 C,H,W)
        try:
            multi_band_img = tiff.imread(img_path)
        except Exception as e:
            print(f"读取TIFF失败 {base_name}: {e}")
            continue

        # 确保形状为 (H, W, C)
        if multi_band_img.ndim == 3 and multi_band_img.shape[0] == 7:
            # 如果是 (7, 128, 128) -> 转为 (128, 128, 7)
            multi_band_img = np.transpose(multi_band_img, (1, 2, 0))
        
        # 读取 Masks
        gt_img = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
        pred_img = cv2.imread(pred_path, cv2.IMREAD_GRAYSCALE)
        vis_polyon_img = cv2.imread(vis_path, cv2.IMREAD_COLOR)

        # 确保尺寸一致
        h, w = gt_img.shape
        gt_bin = gt_img # 二值化
        pred_bin = pred_img

        # ===========================
        # 2. 准备各个可视化组件
        # ===========================
        
        vis_images = [] # 存储12张图

        # --- Row 1: 物理/单通道波段 ---
        # Band 1: Thermal, Band 2: Slope, Band 3: DEM, Band 4: Gray
        titles_row1 = ["B1: Thermal", "B2: Slope", "B3: DEM", "B4: Gray(CTX)"]
        for i in range(4):
            band_data = multi_band_img[:, :, i]
            norm_band = normalize_to_uint8(band_data)
            # 应用伪彩色让DEM/Slope更好看 (可选，这里用灰度保持纯粹)
            vis_img = to_bgr(norm_band)
            vis_images.append(add_label_header(vis_img, titles_row1[i]))

        # --- Row 2: 色彩分量 & RGB合成 ---
        # Band 5: R, Band 6: G, Band 7: B
        # 注意：MMLS中 B5=R, B6=G, B7=B
        b_r = normalize_to_uint8(multi_band_img[:, :, 4])
        b_g = normalize_to_uint8(multi_band_img[:, :, 5])
        b_b = normalize_to_uint8(multi_band_img[:, :, 6])

        titles_row2 = ["B5: Red", "B6: Green", "B7: Blue", "RGB Composite"]
        
        # 单通道展示
        vis_images.append(add_label_header(to_bgr(b_r), titles_row2[0]))
        vis_images.append(add_label_header(to_bgr(b_g), titles_row2[1]))
        vis_images.append(add_label_header(to_bgr(b_b), titles_row2[2]))

        # 合成 RGB (OpenCV 使用 BGR 顺序)
        rgb_composite = cv2.merge([b_b, b_g, b_r])
        vis_images.append(add_label_header(rgb_composite, titles_row2[3]))

        # --- Row 3: 结果分析 ---
        
        # 1. GT Mask
        vis_gt = to_bgr(gt_bin * 255)
        vis_images.append(add_label_header(vis_gt, "GT Mask"))

        # 2. Pred Mask
        vis_pred = to_bgr(pred_bin * 255)
        vis_images.append(add_label_header(vis_pred, "Pred Mask"))

        # 3. Error Map (TP=白, FP=红, FN=蓝)
        status_map = 2 * gt_bin + pred_bin
        error_vis = np.zeros((h, w, 3), dtype=np.uint8)
        error_vis[status_map == 1] = [0, 0, 255]    # FP: Red
        error_vis[status_map == 2] = [255, 0, 0]    # FN: Blue
        error_vis[status_map == 3] = [255, 255, 255]# TP: White
        vis_images.append(add_label_header(error_vis, "Error Map"))

        vis_images.append(add_label_header(vis_polyon_img, "Vis"))

        # ===========================
        # 3. 拼合 3x4 网格
        # ===========================
        # Row 1
        row1 = cv2.hconcat(vis_images[0:4])
        # Row 2
        row2 = cv2.hconcat(vis_images[4:8])
        # Row 3
        row3 = cv2.hconcat(vis_images[8:12])

        final_grid = cv2.vconcat([row1, row2, row3])

        # 保存
        cv2.imwrite(save_path, final_grid)


if __name__ == "__main__":
    dataset_root = r"/scrinvme/huilin/bdd/cp_data/mars_seg/Mars_LSc_2025_dataset_1st_phase_updateB2"
    img_dir = os.path.join(dataset_root, "train/images")
    mask_dir = os.path.join(dataset_root, "train/masks")
    
    analyze_band_importance(img_dir, mask_dir)

    dataset_root = r"/scrinvme/huilin/bdd/cp_data/mars_seg/Mars_LSc_2025_dataset_1st_phase_updateB2"
    img_dir = os.path.join(dataset_root, "val/images")
    mask_dir = os.path.join(dataset_root, "val/masks")
    
    analyze_band_importance(img_dir, mask_dir)

    # IMG_DIR = r'/scrinvme/huilin/bdd/cp_data/mars_seg/Mars_LSc_2025_dataset_1st_phase/val/images'
    # VIS_DIR = r'/scrinvme/huilin/bdd/cp_data/mars_seg/Mars_LSc_2025_dataset_1st_phase/yolo_data/val/vis_result'
    # GT_DIR = r'/scrinvme/huilin/bdd/cp_data/mars_seg/Mars_LSc_2025_dataset_1st_phase/val/masks'
    # PRED_DIR = r'val_ana/mask'
    # OUTPUT_VIS_DIR = r'val_ana/compare'

    # if os.path.exists(GT_DIR) and os.path.exists(PRED_DIR):
    #     generate_comprehensive_vis(IMG_DIR, VIS_DIR, GT_DIR, PRED_DIR, OUTPUT_VIS_DIR)
    # else:
    #     print("路径配置错误，请检查 GT_DIR 和 PRED_DIR")
    # 修改为你的路径
