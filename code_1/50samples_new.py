import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shutil  # 新增：用于复制文件

def load_channel_heights(height_file_path):
    if not os.path.exists(height_file_path):
        raise FileNotFoundError(f"通道高度文件不存在：{height_file_path}")
    
    try:
        height_df = pd.read_csv(
            height_file_path,
            sep=r"\s+",
            header=None,
            names=['channel_pair', 'height'],
            dtype={'channel_pair': str, 'height': float}
        )
    except Exception as e:
        raise ValueError(f"读取通道高度文件失败：{str(e)}")
    
    if height_df.shape[1] != 2:
        raise ValueError(f"通道高度文件需2列，实际{height_df.shape[1]}列")
    if height_df['height'].isna().any():
        raise ValueError("通道高度文件存在空值")
    
    try:
        channel_pairs = height_df['channel_pair'].astype(int).tolist()
    except ValueError:
        raise ValueError("通道对编号必须为整数")
    
    print(f"✅ 加载通道对-高度数据（{len(height_df)}条），通道对编号：{channel_pairs[:5]}...")
    return height_df, channel_pairs

def load_fcdi_data(fcdi_csv_path, channel_pairs):
    if not os.path.exists(fcdi_csv_path):
        raise FileNotFoundError(f"FCDI数据文件不存在：{fcdi_csv_path}")
    
    try:
        fcdi_all = pd.read_csv(
            fcdi_csv_path,
            encoding='utf-8',
            header=None,
            sep=','
        )
    except Exception as e:
        raise ValueError(f"读取FCDI数据失败：{str(e)}")
    
    # 前11列：alon1, alat1, cris_fov, cris_for, all_viirs_num_matched, clear, lw, slw, mp, ice, notsure
    required_cols = ['alon1', 'alat1', 'cris_fov', 'cris_for', 'all_viirs_num_matched', 'clear', 'lw', 'slw', 'mp', 'ice', 'notsure']
    total_columns = fcdi_all.shape[1]
    if total_columns != 242:  # 11 + 1空 + 230通道
        raise ValueError(f"FCDI数据列数不匹配，实际{total_columns}列，预期242列")
    
    # 加载所有列，不筛选FCDI列（保持原FCDI不变）
    fcdi_df = fcdi_all.copy()
    
    # 为所有列分配列名：前11列固定，第12列(索引11)为'empty'，第13列起(索引12)为channel_1到channel_230
    channel_cols_all = [f'channel_{i}' for i in range(1, 231)]  # 230个通道
    fcdi_df.columns = required_cols + ['empty'] + channel_cols_all
    
    # 只为可视化和NPZ定义匹配的通道对列（不影响CSV保存）
    channel_cols = [f'channel_{x}' for x in channel_pairs if f'channel_{x}' in fcdi_df.columns]
    if len(channel_cols) != len(channel_pairs):
        raise ValueError("部分通道对未在CSV中找到")
    
    print(f"✅ 加载FCDI数据（共{len(fcdi_df)}个点），总列数{total_columns}（包含空列和所有230个通道对），用于可视化的匹配通道对{len(channel_cols)}个")
    return fcdi_df, channel_cols

def generate_labels(fcdi_df, threshold):
    labels = []
    label_names = {0: 'clear', 1: 'lw', 2: 'slw', 3: 'mp', 4: 'ice', 5: 'notsure'}
    for _, row in fcdi_df.iterrows():
        total = row['all_viirs_num_matched']
        if total == 0:
            labels.append(5)
            continue
        if row['clear'] / total >= threshold:
            labels.append(0)
            continue
        flag = -1
        phases = ['lw', 'slw', 'mp', 'ice']
        for i, phase in enumerate(phases):
            if row[phase] / total >= threshold:
                labels.append(i + 1)
                flag = 1
                break
        if flag == -1:
            labels.append(5)
    fcdi_df['label'] = labels
    fcdi_df['label_name'] = fcdi_df['label'].map(label_names)
    return fcdi_df

def plot_single_point(plot_df, alon, alat, label, label_name, save_dir, threshold=None):
    os.makedirs(save_dir, exist_ok=True)
    
    alon_str = f"{alon:.4f}".replace('.', '_')
    alat_str = f"{alat:.4f}".replace('.', '_')
    thresh_suffix = f"_thresh_{str(threshold).replace('.', '_')}" if threshold is not None else ""
    save_path = os.path.join(save_dir, f"FCDI_{alon_str}_{alat_str}_{label}_{label_name}{thresh_suffix}.png")
    
    min_fcdi = plot_df['fcdi_value'].min()
    max_fcdi = plot_df['fcdi_value'].max()
    print(f"当前点FCDI值范围：[{min_fcdi:.3f}, {max_fcdi:.3f}]")
    
    plt.figure(figsize=(10, 8))
    plt.scatter(
        x=plot_df['fcdi_value'],
        y=plot_df['height'],
        c='darkred',
        s=50,
        alpha=0.8,
        edgecolors='black',
        linewidth=0.5
    )
    
    plt.xlabel('FCDI值', fontsize=12, fontweight='bold')
    plt.ylabel('通道对高度', fontsize=12, fontweight='bold')
    plt.title(f'FCDI-高度散点图\n({alon:.4f}°E, {alat:.4f}°N, Label: {label_name})', fontsize=14, fontweight='bold')
    plt.xlim(-5, 5)
    plt.ylim(0, 15)
    plt.grid(axis='both', alpha=0.3, linestyle='--')
    plt.tight_layout()
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"📊 已生成：{os.path.basename(save_path)}")
    return save_path  # 返回路径，便于后续复制

def main():
    height_file_path = r"D:\exploring_clouds\channel_heights.txt"
    fcdi_csv_path = r"D:\exploring_clouds\data\test20251019.csv"
    save_single_dir = r"E:\exploring_clouds\figure"
    
    height_df, channel_pairs = load_channel_heights(height_file_path)
    fcdi_df, channel_cols = load_fcdi_data(fcdi_csv_path, channel_pairs)
    
    # 选定50个点并保存
    sampled_df = fcdi_df.sample(n=50, random_state=42).reset_index(drop=True)
    base_name = os.path.basename(fcdi_csv_path).replace('.csv', '')
    samples_csv_path = os.path.join(os.path.dirname(fcdi_csv_path), f"{base_name}_50samples.csv")
    sampled_df.to_csv(samples_csv_path, index=False)
    print(f"✅ 保存50个采样点至：{samples_csv_path}")
    
    thresholds = [0.5, 0.65, 0.75, 0.85, 0.95]
    label_dict = {}  # 存储每个点的所有阈值标签 {idx: {thresh: (label, label_name)}}
    plot_paths = {}  # 存储每个点的所有阈值散点图路径 {idx: {thresh: path}}
    
    for threshold in thresholds:
        thresh_str = str(threshold).replace('.', '_')
        thresh_dir = os.path.join(save_single_dir, f"threshold_{thresh_str}")
        os.makedirs(thresh_dir, exist_ok=True)
        
        # 生成标签
        fcdi_sampled = generate_labels(sampled_df.copy(), threshold)
        
        # 保存标签信息到txt
        label_info_path = os.path.join(thresh_dir, f"labels_threshold_{thresh_str}.txt")
        with open(label_info_path, "w", encoding="utf-8") as f:
            f.write("行号\t经度(alon1)\t纬度(alat1)\t标签\t标签名\n")
            for idx, row in fcdi_sampled.iterrows():
                f.write(f"{idx}\t{row['alon1']:.4f}\t{row['alat1']:.4f}\t{row['label']}\t{row['label_name']}\n")
                # 收集标签
                if idx not in label_dict:
                    label_dict[idx] = {}
                label_dict[idx][threshold] = (row['label'], row['label_name'])
        print(f"📝 标签信息保存至：{label_info_path}")
        
        # 生成每个点的散点图
        print(f"\n开始为阈值 {threshold} 生成散点图...")
        for idx, row in fcdi_sampled.iterrows():
            alon = row['alon1']
            alat = row['alat1']
            label = row['label']
            label_name = row['label_name']
            fcdi_values = row[channel_cols].values
            plot_df = height_df.copy()
            plot_df['fcdi_value'] = fcdi_values
            plot_path = plot_single_point(plot_df, alon, alat, label, label_name, thresh_dir, threshold=threshold)
            # 收集路径
            if idx not in plot_paths:
                plot_paths[idx] = {}
            plot_paths[idx][threshold] = plot_path
        print(f"所有散点图 for 阈值 {threshold} 已保存至：{thresh_dir}")
    
    # 新功能：找出标签变化的点
    change_dir = os.path.join(save_single_dir, "sample_change")
    os.makedirs(change_dir, exist_ok=True)
    
    change_info_path = os.path.join(change_dir, "changed_labels.txt")
    with open(change_info_path, "w", encoding="utf-8") as f:
        f.write("行号\t经度(alon1)\t纬度(alat1)\t阈值\t标签\t标签名\n")
        changed_points = []
        for idx in label_dict:
            labels_set = set(label_dict[idx][thresh][0] for thresh in thresholds)
            if len(labels_set) > 1:  # 标签有变化
                changed_points.append(idx)
                alon = sampled_df.loc[idx, 'alon1']
                alat = sampled_df.loc[idx, 'alat1']
                for thresh in sorted(label_dict[idx]):
                    label, label_name = label_dict[idx][thresh]
                    f.write(f"{idx}\t{alon:.4f}\t{alat:.4f}\t{thresh}\t{label}\t{label_name}\n")
                    
                    # 复制对应的散点图到change_dir
                    src_path = plot_paths[idx][thresh]
                    dest_path = os.path.join(change_dir, os.path.basename(src_path))
                    shutil.copy(src_path, dest_path)
                    print(f"📊 复制变化点散点图：{os.path.basename(dest_path)}")
    
    print(f"✅ 标签变化点信息保存至：{change_info_path}")
    print(f"✅ 变化点数量：{len(changed_points)}，散点图已复制至：{change_dir}")

if __name__ == "__main__":
    plt.rcParams["font.family"] = ["SimHei", "Microsoft YaHei"]
    plt.rcParams['axes.unicode_minus'] = False
    main()