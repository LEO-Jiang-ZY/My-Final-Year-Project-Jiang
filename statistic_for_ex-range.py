import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import logging
from datetime import datetime
import argparse

def setup_logging(script_dir):
    log_dir = os.path.join(script_dir, "log_analysis")
    os.makedirs(log_dir, exist_ok=True)
    run_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"extreme_analysis_run_{run_time}.md")
    
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter('%(message)s'))
    logging.getLogger().addHandler(file_handler)
    
    logging.info(f"# 极端值分析程序运行日志 - {run_time}\n")
    return log_file

def get_matched_csv_files(labels_dir, threshold_clear, threshold_phases):
    thresh_clear_str = str(threshold_clear).replace('.', '_')
    thresh_phases_str = str(threshold_phases).replace('.', '_')
    
    all_csv_files = [f for f in os.listdir(labels_dir) if f.endswith('.csv') and f'clear_{thresh_clear_str}_phases_{thresh_phases_str}' in f]
    
    if not all_csv_files:
        raise FileNotFoundError(f"未找到匹配 threshold_clear={threshold_clear}, threshold_phases={threshold_phases} 的CSV文件")
    
    file_info = []
    for csv_file in all_csv_files:
        # 从文件名提取 date_ad：假设文件名格式为 {date_ad}_labels_clear_xxx_phases_xxx.csv
        parts = csv_file.split('_labels_')
        if len(parts) == 2:
            date_ad = parts[0]
            file_path = os.path.join(labels_dir, csv_file)
            file_info.append((date_ad, file_path))
            logging.info(f"✅ 发现匹配文件：{csv_file} (date_ad: {date_ad})")
    
    return file_info

def load_single_df(file_path):
    df = pd.read_csv(file_path, encoding='utf-8')
    logging.info(f"✅ 加载单个标签数据：{file_path}（{len(df)}行）")
    return df

def analyze_extremes_per_date(df, target_label, extreme_pos, extreme_neg, channel_cols, date_ad):
    filtered_df = df[df['label_name'] == target_label]
    if len(filtered_df) == 0:
        logging.warning(f"日期 {date_ad} 没有找到标签 '{target_label}' 的数据，跳过分析")
        return [], np.zeros(len(channel_cols)), pd.DataFrame()
    
    extreme_records = []  # (date_ad, surface_type, alon1, alat1, channel, value)
    channel_extreme_counts = np.zeros(len(channel_cols))
    extreme_points = []  # 收集有极端值的整行索引
    
    for idx, row in filtered_df.iterrows():
        has_extreme = False
        for ch_idx, channel in enumerate(channel_cols):
            value = row[channel]
            if value > extreme_pos or value < extreme_neg:
                extreme_records.append((
                    date_ad,
                    row['surface_type'],
                    row['alon1'],
                    row['alat1'],
                    channel,
                    value
                ))
                channel_extreme_counts[ch_idx] += 1
                has_extreme = True
        if has_extreme:
            extreme_points.append(idx)
    
    extreme_points_df = filtered_df.loc[extreme_points].copy() if extreme_points else pd.DataFrame()
    
    total_extremes = len(extreme_records)
    logging.info(f"✅ 日期 {date_ad} 针对 '{target_label}' 标签，发现{total_extremes}个极端值（>{extreme_pos} 或 <{extreme_neg}）")
    logging.info(f"✅ 日期 {date_ad} 极端通道计数总计：{channel_extreme_counts.sum()}")
    
    return extreme_records, channel_extreme_counts, extreme_points_df

def save_extreme_records(extreme_records, save_dir, target_label, threshold_clear, threshold_phases, extreme_pos, extreme_neg, date_ad):
    os.makedirs(save_dir, exist_ok=True)
    
    thresh_clear_str = str(threshold_clear).replace('.', '_')
    thresh_phases_str = str(threshold_phases).replace('.', '_')
    range1 = int(extreme_pos) if extreme_pos.is_integer() else extreme_pos
    range2 = int(extreme_neg) if extreme_neg.is_integer() else extreme_neg
    
    extreme_txt_path = os.path.join(save_dir, f"extreme_fcdi_{target_label}_{thresh_clear_str}phases{thresh_phases_str}_range_{range1}_{range2}_{date_ad}.txt")
    
    with open(extreme_txt_path, "w", encoding="utf-8") as f:
        if not extreme_records:
            f.write(f"No extreme FCDI values (>{extreme_pos} or <{extreme_neg}) found for {date_ad}.\n")
        else:
            f.write("date_ad,surface_type,alon1,alat1,channel,value\n")
            for rec in extreme_records:
                f.write(f"{rec[0]},{rec[1]},{rec[2]:.2f},{rec[3]:.2f},{rec[4]},{rec[5]:.2f}\n")
    logging.info(f"📝 已保存日期 {date_ad} 极端值TXT：{extreme_txt_path}")

def save_extreme_points_df(extreme_points_df, save_dir, target_label, threshold_clear, threshold_phases, extreme_pos, extreme_neg, date_ad):
    if extreme_points_df.empty:
        logging.info(f"日期 {date_ad} 没有极端点数据，跳过CSV保存")
        return
    
    os.makedirs(save_dir, exist_ok=True)
    
    thresh_clear_str = str(threshold_clear).replace('.', '_')
    thresh_phases_str = str(threshold_phases).replace('.', '_')
    range1 = int(extreme_pos) if extreme_pos.is_integer() else extreme_pos
    range2 = int(extreme_neg) if extreme_neg.is_integer() else extreme_neg
    
    extreme_csv_path = os.path.join(save_dir, f"extreme_points_{date_ad}_{target_label}_{thresh_clear_str}phases{thresh_phases_str}_range_{range1}_{range2}.csv")
    extreme_points_df.to_csv(extreme_csv_path, index=False, encoding='utf-8')
    logging.info(f"💾 已保存日期 {date_ad} 极端点完整数据CSV：{extreme_csv_path}")

def plot_extreme_distribution(channel_cols, channel_extreme_counts, save_dir, target_label, threshold_clear, threshold_phases, extreme_pos, extreme_neg, date_ad):
    if np.sum(channel_extreme_counts) == 0:
        logging.warning(f"日期 {date_ad} 没有极端值，跳过绘图")
        return
    
    plt.figure(figsize=(20, 8))
    plt.bar(channel_cols, channel_extreme_counts, color='salmon', edgecolor='black')
    plt.xlabel('Channel Pairs', fontsize=12)
    plt.ylabel('Count of Extreme Values', fontsize=12)
    plt.title(f'Extreme FCDI Values Distribution for Label: {target_label} on {date_ad} (>{extreme_pos} or <{extreme_neg})', fontsize=14)
    plt.xticks(rotation=90, ha='right', fontsize=8)
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    
    thresh_clear_str = str(threshold_clear).replace('.', '_')
    thresh_phases_str = str(threshold_phases).replace('.', '_')
    range1 = int(extreme_pos) if extreme_pos.is_integer() else extreme_pos
    range2 = int(extreme_neg) if extreme_neg.is_integer() else extreme_neg
    
    save_path = os.path.join(save_dir, f"extreme_distribution_{target_label}_{thresh_clear_str}phases{thresh_phases_str}_range_{range1}_{range2}_{date_ad}.png")
    plt.savefig(save_path, dpi=300)
    plt.close()
    logging.info(f"📊 已生成日期 {date_ad} 极端值分布图：{save_path}")

def main(threshold_phases, extreme_pos, extreme_neg):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    setup_logging(script_dir)
    
    labels_dir = os.path.join(script_dir, "labels")
    save_dir = r"/data/lgh/jzy/exploring_clouds/result/statistic_FCDI_value_all_channel/histogram_single_label/extreme_analysis"
    
    target_label = "clear" ##
    channel_cols = [f'channel_{i}' for i in range(1, 231)]
    
    threshold_clears = np.arange(0.5, 1.0, 0.05)
    
    for threshold_clear in threshold_clears:
        logging.info(f"\n## 处理 threshold_clear = {threshold_clear:.2f} (threshold_phases = {threshold_phases:.2f}, extreme_range = >{extreme_pos} or <{extreme_neg})\n")
        
        try:
            file_info_list = get_matched_csv_files(labels_dir, threshold_clear, threshold_phases)
            for date_ad, file_path in file_info_list:
                logging.info(f"\n### 处理日期：{date_ad}\n")
                df = load_single_df(file_path)
                extreme_records, channel_extreme_counts, extreme_points_df = analyze_extremes_per_date(df, target_label, extreme_pos, extreme_neg, channel_cols, date_ad)
                save_extreme_records(extreme_records, save_dir, target_label, threshold_clear, threshold_phases, extreme_pos, extreme_neg, date_ad)
                save_extreme_points_df(extreme_points_df, save_dir, target_label, threshold_clear, threshold_phases, extreme_pos, extreme_neg, date_ad)
                plot_extreme_distribution(channel_cols, channel_extreme_counts, save_dir, target_label, threshold_clear, threshold_phases, extreme_pos, extreme_neg, date_ad)
        except Exception as e:
            logging.error(f"处理 threshold_clear={threshold_clear:.2f} 时出错：{str(e)}")
    
    logging.info("\n# 程序运行完成")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FCDI极端值分析与绘图程序（针对clear标签，扩大极端范围，按日期分别处理）")
    parser.add_argument("--thresh_phases", type=float, default=0.8, help="phases标签的阈值 (默认0.8)")
    parser.add_argument("--extreme_pos", type=float, default=4.0, help="正极端阈值 (默认4.0)")
    parser.add_argument("--extreme_neg", type=float, default=-4.0, help="负极端阈值 (默认-4.0)")
    args = parser.parse_args()
    
    main(args.thresh_phases, args.extreme_pos, args.extreme_neg)