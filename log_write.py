import re
import pandas as pd
import os

def parse_training_log_to_csv(log_path, output_csv, default_model_name="m3lsnet"):
    """
    读取训练log，提取文件夹名、最优mIoU及其对应的F1, FG, BG指标，并写入CSV。
    """
    with open(log_path, 'r', encoding='utf-8') as f:
        content = f.read()

    model_match = re.search(r"Model:\s+([\w\d_]+)", content)
    if model_match:
        model_name = model_match.group(1)
    else:
        model_name = default_model_name

    # 1. 提取文件夹名 (Folder Name)
    # 匹配 "Outputs will be saved to: outputs/..."
    folder_match = re.search(r"Outputs will be saved to: outputs[/\\]([\w\d_]+)", content)
    folder_name = folder_match.group(1) if folder_match else "Unknown"

    # 2. 提取所有指标 (Metrics)
    # 替换换行符以便正则匹配跨行数据
    normalized_content = content.replace('\n', ' ')
    
    # 正则匹配模式：依次寻找 mIoU, F1, IoU (FG), IoU (BG)
    # 格式示例: mIoU: 0.8170 | F1: 0.8608 | IoU (FG): 0.7556 | IoU (BG): 0.8784
    pattern = r"mIoU:\s*([\d\.]+).*?F1:\s*([\d\.]+).*?IoU \(FG\):\s*([\d\.]+).*?IoU \(BG\):\s*([\d\.]+)"
    matches = re.findall(pattern, normalized_content)
    
    # 转换为DataFrame
    data = []
    for m in matches:
        data.append({
            'mIoU': float(m[0]),
            'F1': float(m[1]),
            'IoU (FG)': float(m[2]),
            'IoU (BG)': float(m[3])
        })
    
    if not data:
        print(f"No metrics found in {log_path}")
        data.append({
            'mIoU': 0,
            'F1': 0,
            'IoU (FG)': 0,
            'IoU (BG)': 0
        })

    df_metrics = pd.DataFrame(data)

    # 3. 找到最优 mIoU 所在的行
    best_row = df_metrics.loc[df_metrics['mIoU'].idxmax()]

    # 4. 构建结果数据
    result = {
        'ID': '',
        'Folder Name': folder_name,
        'Submission Name': '',
        'Best mIoU': best_row['mIoU'],
        'F1': best_row['F1'],
        'IoU (FG)': best_row['IoU (FG)'],
        'IoU (BG)': best_row['IoU (BG)'],
        'online score': '',  # 空列，用于手动填写
        'Model': model_name,
        'Mark': '',
        'Take strategy': '',
    }

    # 5. 写入或追加到 CSV
    df_result = pd.DataFrame([result])
    
    # 如果文件不存在则写入头部，如果存在则追加（根据需求调整）
    if not os.path.exists(output_csv):
        df_result.to_csv(output_csv, index=False)
    else:
        df_result.to_csv(output_csv, mode='a', header=False, index=False)
    
    print(f"Successfully processed {log_path} -> Best mIoU: {best_row['mIoU']}")

def batch_write(logs_dir, output_csv, task_list=None):
    """
    批量处理log_dir目录下的所有log文件，将结果写入output_csv。
    """
    if task_list is None:
        task_list = os.listdir(logs_dir)
    task_list.sort()
    for task_name in task_list:
        log_dir = os.path.join(logs_dir, task_name, 'logs')
        log_path = os.path.join(log_dir, f'train_log_{task_name}.txt')
        parse_training_log_to_csv(log_path, output_csv)


if __name__ == '__main__':
    log_dir = '/localnvme/project/M3LSNet/outputs'
    output_csv = 'training_metrics.csv'
    task_list = [
        # '20260227_102956',
        # '20260227_124411',
        # '20260227_145525',
        # '20260227_171848',

        # '20260227_112615',
        # '20260227_234152',
        # '20260228_000443',
        # '20260228_110445',
        # '20260228_114219',

        # '20260228_182250',
        # '20260228_183649',
        # '20260228_185116',
        # '20260228_190556',
        # '20260228_192119',

        # '20260228_193552',
        # '20260228_195121',

        # '20260301_220437',
        # '20260301_220615',
        # '20260301_221725',
        # '20260301_221923',
        # '20260301_223101',
        # '20260301_223250',
        # '20260301_224425',
        # '20260301_224618',
        # '20260301_225824',
        # '20260301_230020',
        # '20260301_231148',
        # '20260301_231357',

        # '20260302_121456',
        # '20260302_122655',
        # '20260302_123856',
        # '20260302_125058',
        # '20260302_130304',
        # '20260302_131628',
        # '20260302_133033',

        # '20260302_150733',
        # '20260302_152732',
        '20260302_160613',
        '20260302_161817',
        '20260302_160619',
        '20260302_161951',
    ]
    batch_write(
        log_dir, 
        output_csv, 
        task_list
        )