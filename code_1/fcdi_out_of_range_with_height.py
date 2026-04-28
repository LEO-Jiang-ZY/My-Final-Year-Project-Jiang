import os
import pandas as pd
import numpy as np
import logging
from datetime import datetime
import argparse

def setup_logging(script_dir):
    log_dir = os.path.join(script_dir, "log_analysis")
    os.makedirs(log_dir, exist_ok=True)
    run_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"fcdi_out_of_range_height_run_{run_time}.md")
    
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter('%(message)s'))
    logging.getLogger().addHandler(file_handler)
    
    logging.info(f"# FCDI Out-of-Range + Height Mapping Run Log - {run_time}\n")
    return log_file

def parse_thresholds(thresh_str, default_pos, default_neg):
    """解析 --thresholds 参数，返回 {label: (pos, neg)} 字典"""
    if not thresh_str:
        return {}
    thresh_dict = {}
    for item in thresh_str.split(','):
        try:
            label, pos, neg = [x.strip() for x in item.split(':')]
            thresh_dict[label] = (float(pos), float(neg))
        except:
            logging.warning(f"阈值格式错误: {item}，将使用默认值")
    return thresh_dict

def main(extreme_pos, extreme_neg, target_labels, thresholds_str):
    input_dir = r"E:\A_exploring_clouds\data\test_3_days\original_data_clear_0_5_phases_0_8"
    height_file = r"E:\A_exploring_clouds\data\test_3_days\channel_pair_height.csv"
    output_dir = r"E:\A_exploring_clouds\data\test_3_days\original_data_out_of_range"
    os.makedirs(output_dir, exist_ok=True)
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    setup_logging(script_dir)
    
    # 解析每个标签的独立阈值
    custom_thresholds = parse_thresholds(thresholds_str, extreme_pos, extreme_neg)
    
    logging.info(f"全局默认阈值: >{extreme_pos} 或 <{extreme_neg}")
    if custom_thresholds:
        logging.info(f"自定义阈值设置: {custom_thresholds}")
    
    # 读取高度映射表
    height_df = pd.read_csv(height_file, header=None,
                           names=['pair_id', 'channel1', 'channel2', 'height1', 'height2', 'avg_height'])
    height_map = height_df.set_index('pair_id').to_dict('index')
    logging.info(f"成功加载高度映射表，共 {len(height_map)} 个通道对")
    
    for csv_file in [f for f in os.listdir(input_dir) if f.lower().endswith('.csv')]:
        file_path = os.path.join(input_dir, csv_file)
        base_name = os.path.splitext(csv_file)[0]
        
        logging.info(f"\n=== 开始处理文件: {csv_file} ===")
        
        try:
            df = pd.read_csv(file_path, encoding='utf-8')
        except Exception as e:
            logging.error(f"读取失败 {csv_file}: {str(e)}")
            continue
        
        if 'label_name' not in df.columns:
            logging.warning(f"文件缺少 label_name 列，跳过")
            continue
        
        channel_cols = [col for col in df.columns if col.startswith('channel_')]
        labels_to_process = target_labels if target_labels else df['label_name'].unique()
        
        for label in labels_to_process:
            # 为当前标签选择阈值
            pos, neg = custom_thresholds.get(label, (extreme_pos, extreme_neg))
            logging.info(f"  标签 '{label}' 使用阈值 → 正:{pos}  负:{neg}")
            
            filtered_df = df[df['label_name'] == label].copy()
            if len(filtered_df) == 0:
                continue
            
            label_folder = os.path.join(output_dir, label)
            os.makedirs(label_folder, exist_ok=True)
            
            file_records = []
            
            for col in channel_cols:
                pair_id = int(col.split('_')[1])
                if pair_id not in height_map:
                    continue
                
                mask = (filtered_df[col] > pos) | (filtered_df[col] < neg)
                if not mask.any():
                    continue
                
                extreme_rows = filtered_df[mask].copy()
                extreme_rows['Dataset'] = base_name
                extreme_rows['Label'] = label
                extreme_rows['Channel_Pair_ID'] = pair_id
                extreme_rows['FCDI_Value'] = extreme_rows[col]
                
                h = height_map[pair_id]
                extreme_rows['Channel1'] = h['channel1']
                extreme_rows['Channel2'] = h['channel2']
                extreme_rows['Height_Channel1_km'] = h['height1']
                extreme_rows['Height_Channel2_km'] = h['height2']
                extreme_rows['Avg_Height_km'] = h['avg_height']
                extreme_rows['Threshold_Pos'] = pos
                extreme_rows['Threshold_Neg'] = neg
                
                keep_cols = ['Dataset', 'Label', 'Channel_Pair_ID', 'FCDI_Value',
                             'Channel1', 'Channel2', 'Height_Channel1_km',
                             'Height_Channel2_km', 'Avg_Height_km',
                             'Threshold_Pos', 'Threshold_Neg',
                             'alon1', 'alat1', 'surface_type']
                
                extreme_rows = extreme_rows[keep_cols].rename(columns={
                    'alon1': 'Lon', 'alat1': 'Lat', 'surface_type': 'Surface_Type'
                })
                
                file_records.append(extreme_rows)
            
            if file_records:
                file_df = pd.concat(file_records, ignore_index=True)
                
                detail_path = os.path.join(label_folder, f"{base_name}_out_of_range_detail.csv")
                summary_path = os.path.join(label_folder, f"{base_name}_out_of_range_summary.csv")
                
                file_df.to_csv(detail_path, index=False, encoding='utf-8')
                
                summary = file_df.groupby(['Label', 'Channel_Pair_ID']).agg({
                    'FCDI_Value': ['count', 'mean'],
                    'Avg_Height_km': ['mean', 'min', 'max']
                }).round(4)
                summary.columns = ['Count', 'Mean_FCDI', 'Mean_Height_km', 'Min_Height_km', 'Max_Height_km']
                summary = summary.reset_index()
                summary.to_csv(summary_path, index=False, encoding='utf-8')
                
                logging.info(f"  → 文件 {base_name} 的结果已保存至 {label}/")
    
    # ==================== 生成带占比的频次表（每个标签独立） ====================
    # 此处使用之前收集的逻辑（简化版，直接从已保存的文件读取也可，这里直接重建）
    logging.info("\n正在生成各标签的通道对频次表（含占比）...")
    for label in os.listdir(output_dir):
        label_path = os.path.join(output_dir, label)
        if not os.path.isdir(label_path):
            continue
        
        all_detail_files = [f for f in os.listdir(label_path) if f.endswith('_detail.csv')]
        if not all_detail_files:
            continue
        
        label_dfs = []
        for f in all_detail_files:
            label_dfs.append(pd.read_csv(os.path.join(label_path, f)))
        
        label_df = pd.concat(label_dfs, ignore_index=True)
        
        pair_freq = label_df['Channel_Pair_ID'].value_counts().reset_index(name='Total_Count')
        pair_freq = pair_freq.sort_values('Total_Count', ascending=False)
        
        total = pair_freq['Total_Count'].sum()
        pair_freq['Percentage (%)'] = (pair_freq['Total_Count'] / total * 100).round(4)
        
        height_summary = label_df.groupby('Channel_Pair_ID')['Avg_Height_km'].mean().round(2).reset_index()
        pair_freq = pair_freq.merge(height_summary, on='Channel_Pair_ID')
        
        pair_info = []
        for pid in pair_freq['Channel_Pair_ID']:
            if pid in height_map:
                h = height_map[pid]
                pair_info.append({'Channel_Pair_ID': pid, 'Channel1': h['channel1'], 'Channel2': h['channel2']})
        pair_info_df = pd.DataFrame(pair_info)
        pair_freq = pair_freq.merge(pair_info_df, on='Channel_Pair_ID')
        
        pair_freq = pair_freq[['Channel_Pair_ID', 'Channel1', 'Channel2', 'Total_Count', 'Percentage (%)', 'Avg_Height_km']]
        
        freq_path = os.path.join(label_path, "test_3_days_channel_pair_frequency.csv")
        pair_freq.to_csv(freq_path, index=False, encoding='utf-8')
        
        logging.info(f"✅ 标签 '{label}' 的频次表（含占比）已生成：{freq_path}")
    
    logging.info("\n# 所有任务已完成！不同标签使用独立阈值，结果按标签分文件夹保存。")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FCDI Out-of-Range Analysis - 支持每个标签不同阈值 + 按标签分文件夹")
    parser.add_argument("--extreme_pos", type=float, default=4.0)
    parser.add_argument("--extreme_neg", type=float, default=-4.0)
    parser.add_argument("--labels", type=str, default="ice", help="要统计的标签，用逗号分隔")
    parser.add_argument("--thresholds", type=str, default="ice:0:0", 
                        help='每个标签独立阈值，例如: "ice:6.0:-6.0,clear:3.5:-3.5,water:4.5:-4.5"')
    args = parser.parse_args()
    
    target_labels = [lbl.strip() for lbl in args.labels.split(',')] if args.labels else []
    main(args.extreme_pos, args.extreme_neg, target_labels, args.thresholds)