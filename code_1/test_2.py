import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

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
    if fcdi_all.shape[1] < 11 + 230:
        raise ValueError(f"FCDI数据列数不足")
    
    # 假设通道对从1到230，对应列索引10 + x (index 11 for 1, 12 for 2, etc.)
    column_offset = 10  # 列索引 = channel_pair + 10
    channel_col_indices = [x + column_offset for x in channel_pairs]
    
    # 检查索引
    max_idx = max(channel_col_indices)
    if max_idx >= fcdi_all.shape[1]:
        raise ValueError(f"最大列索引 {max_idx} 超出文件列数 {fcdi_all.shape[1]}")
    
    all_indices = list(range(11)) + channel_col_indices  # 前11列 + 选中的通道对列
    fcdi_df = fcdi_all.iloc[:, all_indices].copy()
    
    channel_cols = [f'channel_{x}' for x in channel_pairs]
    fcdi_df.columns = required_cols + channel_cols
    
    print(f"✅ 加载FCDI数据（共{len(fcdi_df)}个点），匹配{len(channel_pairs)}个通道对")
    return fcdi_df, channel_cols

def generate_labels(fcdi_df):
    labels = []
    label_names = {0: 'clear', 1: 'lw', 2: 'slw', 3: 'mp', 4: 'ice', 5: 'notsure'}
    for _, row in fcdi_df.iterrows():
        total = row['all_viirs_num_matched']
        if total == 0:
            labels.append(5)
            continue
        if row['clear'] / total >= 0.5:
            labels.append(0)
            continue
        flag = -1
        phases = ['lw', 'slw', 'mp', 'ice']
        for i, phase in enumerate(phases):
            if row[phase] / total >= 0.8:
                labels.append(i + 1)
                flag = 1
                break
        if flag == -1:
            labels.append(5)
    fcdi_df['label'] = labels
    fcdi_df['label_name'] = fcdi_df['label'].map(label_names)
    return fcdi_df

def save_new_csv(fcdi_df, fcdi_csv_path):
    base_name = os.path.basename(fcdi_csv_path).replace('.csv', '')
    new_csv_path = os.path.join(os.path.dirname(fcdi_csv_path), f"{base_name}_match_label.csv")
    fcdi_df.to_csv(new_csv_path, index=False)
    print(f"✅ 新CSV保存至：{new_csv_path}")
    return new_csv_path

def save_npz(fcdi_df, height_df, channel_cols, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    fcdi_values = fcdi_df[channel_cols].values  # (rows, channels)
    heights = height_df['height'].values  # (channels,)
    x = fcdi_values.flatten()
    y = np.tile(heights, len(fcdi_df))
    labels = np.repeat(fcdi_df['label'].values, len(heights))
    npz_path = os.path.join(save_dir, "fcdi_height.npz")
    np.savez(npz_path, fcdi=x, height=y, labels=labels)
    print(f"✅ NPZ保存至：{npz_path}")

def plot_single_point(plot_df, alon, alat, label, label_name, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    
    alon_str = f"{alon:.4f}".replace('.', '_')
    alat_str = f"{alat:.4f}".replace('.', '_')
    save_path = os.path.join(save_dir, f"FCDI_{alon_str}_{alat_str}_{label}_{label_name}.png")
    
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
    plt.xlim(-15, 15)
    plt.ylim(0, 15)
    plt.grid(axis='both', alpha=0.3, linestyle='--')
    plt.tight_layout()
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"📊 已生成：{os.path.basename(save_path)}")

def plot_all_points(fcdi_df, height_df, channel_cols, save_path):
    plt.figure(figsize=(14, 10))
    colors_list = ['#f5eec8', '#0260a0', '#3ca4e5', '#38bdea', '#93d2f5', '#d6daf7']
    for label in range(6):
        subset = fcdi_df[fcdi_df['label'] == label]
        if len(subset) == 0:
            continue
        fcdi_values = subset[channel_cols].values.flatten()
        heights = np.tile(height_df['height'].values, len(subset))
        plt.scatter(
            x=fcdi_values,
            y=heights,
            c=colors_list[label],
            alpha=0.4,
            s=30,
            label=f'Label {label}'
        )
    plt.xlabel('FCDI值', fontsize=14)
    plt.ylabel('通道对高度', fontsize=14)
    plt.title('FCDI-高度散点图（按标签着色）', fontsize=16)
    plt.xlim(-15, 15)
    plt.ylim(0, 15)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"📊 所有点散点图保存至：{save_path}")

def main():
    height_file_path = r"D:\exploring_clouds\channel_heights.txt"
    fcdi_csv_path = r"D:\exploring_clouds\data\test20251019.csv"
    save_single_dir = r"D:\exploring_clouds\figure"
    save_all_path = r"D:\exploring_clouds\fcdi_height_scatter_all.png"
    save_dir = r"D:\exploring_clouds"
    
    height_df, channel_pairs = load_channel_heights(height_file_path)
    fcdi_df, channel_cols = load_fcdi_data(fcdi_csv_path, channel_pairs)
    fcdi_df = generate_labels(fcdi_df)
    save_new_csv(fcdi_df, fcdi_csv_path)
    save_npz(fcdi_df, height_df, channel_cols, save_dir)
    
    # 随机抽样1000行
    total_rows = len(fcdi_df)
    sample_size = min(1000, total_rows)
    fcdi_sampled = fcdi_df.sample(n=sample_size, random_state=42).reset_index(drop=True)
    print(f"\n🎯 已随机抽取 {sample_size} 条数据（共 {total_rows} 条）用于单个散点图绘图")
    
    # 导出抽样信息
    os.makedirs(save_single_dir, exist_ok=True)
    sample_info_path = os.path.join(save_single_dir, "sample_info.txt")
    with open(sample_info_path, "w", encoding="utf-8") as f:
        f.write("行号\t经度(alon1)\t纬度(alat1)\t标签\t标签名\n")
        for idx, row in fcdi_sampled.iterrows():
            f.write(f"{idx}\t{row['alon1']:.4f}\t{row['alat1']:.4f}\t{row['label']}\t{row['label_name']}\n")
    print(f"📝 抽样信息已保存至：{sample_info_path}")
    
    # 生成每个采样点的单个散点图
    print(f"\n开始生成每个采样点的散点图（共{sample_size}个）...")
    for _, row in fcdi_sampled.iterrows():
        alon = row['alon1']
        alat = row['alat1']
        label = row['label']
        label_name = row['label_name']
        fcdi_values = row[channel_cols].values
        plot_df = height_df.copy()
        plot_df['fcdi_value'] = fcdi_values
        plot_single_point(plot_df, alon, alat, label, label_name, save_single_dir)
    print(f"\n所有单个散点图生成完成！保存路径：{save_single_dir}")
    
    # 可选：生成所有点的综合散点图
    plot_all_points(fcdi_df, height_df, channel_cols, save_all_path)

if __name__ == "__main__":
    plt.rcParams["font.family"] = ["SimHei", "Microsoft YaHei"]
    plt.rcParams['axes.unicode_minus'] = False
    main()