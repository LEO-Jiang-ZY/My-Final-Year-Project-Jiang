import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import logging
from datetime import datetime

def setup_logging(script_dir):
    log_dir = os.path.join(script_dir, "log_analysis")
    os.makedirs(log_dir, exist_ok=True)
    run_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"fcdi_distribution_stats_run_{run_time}.md")
    
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter('%(message)s'))
    logging.getLogger().addHandler(file_handler)
    
    logging.info(f"# FCDI Distribution Statistics Run Log - {run_time}\n")
    logging.info("Purpose: Calculate complete statistical distribution of FCDI values for different labels and thresholds\n")
    return log_file

def write_stats_txt(stats, path):
    """将单个标签的统计摘要写入 TXT 文件"""
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"FCDI Statistics Summary\n")
        f.write(f"Dataset : {stats['Dataset']}\n")
        f.write(f"Label   : {stats['Label']}\n")
        f.write(f"{'='*50}\n\n")
        f.write(f"Sample Count : {stats['Sample Count']}\n")
        f.write(f"Mean         : {stats['Mean']}\n")
        f.write(f"Median       : {stats['Median']}\n")
        f.write(f"Mode         : {stats['Mode'] if not np.isnan(stats['Mode']) else 'N/A'}\n")
        f.write(f"Range        : {stats['Range']}\n")
        f.write(f"Variance     : {stats['Variance']}\n")
        f.write(f"Std Dev      : {stats['Std Dev']}\n")
        f.write(f"Min          : {stats['Min']}\n")
        f.write(f"Max          : {stats['Max']}\n")

def write_frequency_txt(freq_df, path):
    """将频率分布写入 TXT 文件"""
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"FCDI Frequency Distribution\n")
        f.write(f"{'='*60}\n")
        f.write(f"Bin Lower     Bin Upper       Count     Frequency     Frequency(%)\n")
        f.write(f"{'-'*60}\n")
        for _, row in freq_df.iterrows():
            f.write(f"{row['Bin Lower']:>10.4f}  {row['Bin Upper']:>10.4f}  {int(row['Count']):>8}  {row['Frequency']:>10.6f}  {row['Frequency (%)']:>12.4f}\n")

def main():
    # ====================== Path Configuration ======================
    input_dir = r"E:\A_exploring_clouds\data\test_3_days\original_data"
    stat_dir = r"E:\A_exploring_clouds\data\test_3_days\original_data_stats"
    
    os.makedirs(stat_dir, exist_ok=True)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    setup_logging(script_dir)
    
    logging.info(f"Input directory: {input_dir}")
    logging.info(f"Statistics output directory: {stat_dir}\n")
    
    # Collect all stats and all frequency data for the final summary files
    all_stats_list = []
    all_freq_list = []          # NEW: 用于汇总所有频率分布
    
    # Get all CSV files
    csv_files = [f for f in os.listdir(input_dir) if f.lower().endswith('.csv')]
    logging.info(f"Found {len(csv_files)} CSV files, starting processing...\n")
    
    for csv_file in csv_files:
        file_path = os.path.join(input_dir, csv_file)
        base_name = os.path.splitext(csv_file)[0]
        
        logging.info(f"Processing file: {csv_file}")
        
        try:
            df = pd.read_csv(file_path, encoding='utf-8')
        except Exception as e:
            logging.error(f"Failed to read file {csv_file}: {str(e)}")
            continue
        
        if 'label_name' not in df.columns:
            logging.warning(f"File {csv_file} has no 'label_name' column, skipping")
            continue
        
        channel_cols = [col for col in df.columns if col.startswith('channel_')]
        if not channel_cols:
            logging.warning(f"File {csv_file} has no channel_ columns, skipping")
            continue
        
        unique_labels = sorted(df['label_name'].unique())
        logging.info(f"  → This file contains {len(unique_labels)} labels: {unique_labels}")
        
        for label in unique_labels:
            filtered_df = df[df['label_name'] == label]
            if len(filtered_df) == 0:
                continue
            
            fcdi_values = filtered_df[channel_cols].values.flatten()
            fcdi_series = pd.Series(fcdi_values)
            
            # Calculate statistics
            stats = {
                'Dataset': base_name,
                'Label': label,
                'Sample Count': len(fcdi_values),
                'Mean': round(fcdi_series.mean(), 4),
                'Median': round(fcdi_series.median(), 4),
                'Range': round(fcdi_series.max() - fcdi_series.min(), 4),
                'Variance': round(fcdi_series.var(), 4),
                'Std Dev': round(fcdi_series.std(), 4),
                'Min': round(fcdi_series.min(), 4),
                'Max': round(fcdi_series.max(), 4),
            }
            rounded = np.round(fcdi_values, decimals=2)
            mode_series = pd.Series(rounded).mode()
            stats['Mode'] = round(mode_series.iloc[0], 4) if not mode_series.empty else np.nan
            
            all_stats_list.append(stats)
            
            # ==================== Frequency distribution (for both per-label TXT and global CSV) ====================
            hist, bin_edges = np.histogram(fcdi_values, bins=30)
            freq_df = pd.DataFrame({
                'Bin Lower': bin_edges[:-1].round(4),
                'Bin Upper': bin_edges[1:].round(4),
                'Count': hist,
                'Frequency': (hist / len(fcdi_values)).round(6),
                'Frequency (%)': ((hist / len(fcdi_values)) * 100).round(4)
            })
            
            # Add Dataset and Label for the global summary CSV
            freq_df['Dataset'] = base_name
            freq_df['Label'] = label
            all_freq_list.append(freq_df.copy())   # NEW: 收集到全局列表
            
            # ==================== Save per-label TXT files ====================
            base_path = os.path.join(stat_dir, f"{base_name}_{label}")
            
            stats_path = base_path + "_stats_summary.txt"
            write_stats_txt(stats, stats_path)
            
            freq_path = base_path + "_frequency_distribution.txt"
            write_frequency_txt(freq_df, freq_path)   # 保持原来的 per-label TXT
            
            # ==================== Plot Histogram (PNG unchanged) ====================
            plt.figure(figsize=(12, 7))
            plt.hist(fcdi_values, bins=30, color='skyblue', edgecolor='black', alpha=0.75)
            plt.axvline(stats['Mean'], color='red', linestyle='--', linewidth=2, label=f'Mean = {stats["Mean"]}')
            plt.axvline(stats['Median'], color='green', linestyle='--', linewidth=2, label=f'Median = {stats["Median"]}')
            if not np.isnan(stats['Mode']):
                plt.axvline(stats['Mode'], color='orange', linestyle='--', linewidth=2, label=f'Mode ≈ {stats["Mode"]}')
            plt.title(f'{base_name}\nLabel: {label} - FCDI Value Distribution (Samples: {stats["Sample Count"]})', fontsize=14)
            plt.xlabel('FCDI Value')
            plt.ylabel('Frequency')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            
            plot_path = base_path + "_histogram.png"
            plt.savefig(plot_path, dpi=300)
            plt.close()
            
            logging.info(f"  ✅ Label '{label}' completed → TXT summary + frequency + histogram saved")
        
        logging.info(f"File {csv_file} processing finished\n")
    
    # ==================== Create the two final aggregated summary files ====================
    
    # 1. All statistics summary (TXT)
    all_stats_summary_path = os.path.join(stat_dir, "test_3_days_stats_all_summary.txt")
    with open(all_stats_summary_path, "w", encoding="utf-8") as f:
        f.write("FCDI STATISTICS SUMMARY - ALL DATASETS AND LABELS\n")
        f.write("=" * 80 + "\n")
        f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"{'Dataset':<35} {'Label':<15} {'Samples':>8} {'Mean':>10} {'Median':>10} {'Mode':>10} {'Std Dev':>10} {'Range':>10}\n")
        f.write("-" * 120 + "\n")
        
        for s in all_stats_list:
            mode_val = s['Mode'] if not np.isnan(s['Mode']) else "N/A"
            f.write(f"{s['Dataset']:<35} {s['Label']:<15} {s['Sample Count']:>8} "
                    f"{s['Mean']:>10.4f} {s['Median']:>10.4f} {mode_val:>10} "
                    f"{s['Std Dev']:>10.4f} {s['Range']:>10.4f}\n")
        
        f.write("\n" + "=" * 80 + "\n")
        f.write("End of summary. Detailed per-label TXT files are in the same folder.\n")
    
    # 2. All frequency distributions combined (CSV) - NEW FEATURE
    if all_freq_list:
        combined_freq = pd.concat(all_freq_list, ignore_index=True)
        # Reorder columns for better readability
        combined_freq = combined_freq[['Dataset', 'Label', 'Bin Lower', 'Bin Upper', 'Count', 'Frequency', 'Frequency (%)']]
        all_freq_summary_path = os.path.join(stat_dir, "test_3_days_frequency_all_summary.csv")
        combined_freq.to_csv(all_freq_summary_path, index=False, encoding='utf-8')
    else:
        all_freq_summary_path = "None (no data)"
    
    # ==================== Final logging ====================
    logging.info("# All processing completed!")
    logging.info(f"   Per-label TXT files and histograms saved in: {stat_dir}")
    logging.info(f"   COMPLETE STATS SUMMARY TXT : {all_stats_summary_path}")
    logging.info(f"   COMPLETE FREQUENCY SUMMARY CSV : {all_freq_summary_path}")
    logging.info("   You can now easily compare all labels in the two summary files and set custom extreme thresholds.")

if __name__ == "__main__":
    main()