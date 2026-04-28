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
    column_offset = 10  # 修正偏移量为10
    channel_col_indices = [x + column_offset for x in channel_pairs]
    
    # 检查索引
    for idx in channel_col_indices:
        if idx < 11 or idx >= fcdi_all.shape[1]:
            raise ValueError(f"无效列索引：{idx}")
    
    all_indices = list(range(11)) + channel_col_indices  # 前11列 + 选中的通道对列
    fcdi_df = fcdi_all.iloc[:, all_indices].copy()
    
    channel_cols = [f'channel_{x}' for x in channel_pairs]
    fcdi_df.columns = required_cols + channel_cols
    
    print(f"✅ 加载FCDI数据（共{len(fcdi_df)}个点），匹配{len(channel_pairs)}个通道对")
    return fcdi_df, channel_cols

def generate_labels(fcdi_df):
    labels = []
    for _, row in fcdi_df.iterrows():
        total = row['all_viirs_num_matched']
        if total == 0:
            labels.append(5)  # 或处理为notsure
            continue
        if row['clear'] / total >= 0.95:
            labels.append(0)
            continue
        flag = -1
        phases = ['lw', 'slw', 'mp', 'ice']
        for i, phase in enumerate(phases):
            if row[phase] / total >= 0.95:
                labels.append(i + 1)
                flag = 1
                break
        if flag == -1:
            labels.append(5)
    fcdi_df['label'] = labels
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
    # 对于所有点，保存 FCDI 和 height 的 mesh 或 flatten
    # 假设保存 flatten 的 x,y
    x = fcdi_values.flatten()
    y = np.tile(heights, len(fcdi_df))
    labels = np.repeat(fcdi_df['label'].values, len(heights))
    npz_path = os.path.join(save_dir, "fcdi_height.npz")
    np.savez(npz_path, fcdi=x, height=y, labels=labels)
    print(f"✅ NPZ保存至：{npz_path}")

def plot_scatter(fcdi_df, height_df, channel_cols, save_path):
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
    print(f"📊 散点图保存至：{save_path}")

def main():
    height_file_path = r"D:\exploring_clouds\channel_heights.txt"
    fcdi_csv_path = r"D:\exploring_clouds\data\test20251019.csv"
    save_dir = r"D:\exploring_clouds"
    save_plot_path = os.path.join(save_dir, "fcdi_height_scatter.png")
    
    height_df, channel_pairs = load_channel_heights(height_file_path)
    fcdi_df, channel_cols = load_fcdi_data(fcdi_csv_path, channel_pairs)
    fcdi_df = generate_labels(fcdi_df)
    save_new_csv(fcdi_df, fcdi_csv_path)
    save_npz(fcdi_df, height_df, channel_cols, save_dir)
    plot_scatter(fcdi_df, height_df, channel_cols, save_plot_path)

if __name__ == "__main__":
    plt.rcParams["font.family"] = ["SimHei", "Microsoft YaHei"]
    plt.rcParams['axes.unicode_minus'] = False
    main()