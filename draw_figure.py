import os
import pandas as pd
import matplotlib.pyplot as plt


def load_channel_heights(height_file_path):
    if not os.path.exists(height_file_path):
        raise FileNotFoundError(f"通道高度文件不存在：{height_file_path}")
    
    try:
        height_df = pd.read_csv(
            height_file_path,
            sep=r"\s+",  # 通道对-高度文件通常是空格分隔，保持不变
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
    if len(height_df) != 109:
        raise ValueError(f"通道高度数据应为109条，实际{len(height_df)}条")
    
    try:
        channel_pairs = height_df['channel_pair'].astype(int).tolist()
    except ValueError:
        raise ValueError("通道对编号必须为整数（如2,10,127...）")
    
    print(f"✅ 加载通道对-高度数据（109条），通道对编号：{channel_pairs[:5]}...")
    return height_df, channel_pairs


def load_fcdi_data(fcdi_csv_path, channel_pairs, column_offset):
    if not os.path.exists(fcdi_csv_path):
        raise FileNotFoundError(f"FCDI数据文件不存在：{fcdi_csv_path}")
    
    try:
        # 核心修改：将sep从r"\s+"改为','（CSV文件标准分隔符）
        fcdi_all = pd.read_csv(
            fcdi_csv_path,
            encoding='utf-8',
            header=None,  # 第一行是数据，无表头
            sep=','  # 关键修正：CSV用逗号分隔，而非空格
        )
    except Exception as e:
        raise ValueError(f"读取FCDI数据失败：{str(e)}（若仍报错，需检查文件实际分隔符）")
    
    # 计算列索引（映射关系已正确，无需修改）
    try:
        channel_col_indices = [x + column_offset for x in channel_pairs]
        print("\n===== 通道对-列索引映射关系 =====\n")
        for pair, idx in zip(channel_pairs[:10], channel_col_indices[:10]):  # 只打印前10个，避免输出过长
            print(f"通道对 {pair} → 列索引 {idx}")
        if len(channel_pairs) > 10:
            print("...（其余99个通道对映射略）")
        print("=================================\n")
        
        # 检查索引有效性
        for idx in channel_col_indices:
            if not isinstance(idx, int) or idx < 0:
                raise ValueError(f"计算出无效列索引：{idx}（请检查offset是否正确）")
        
        # 检查列数是否足够
        max_col_index = max(channel_col_indices)
        required_min_columns = max_col_index + 1
        if fcdi_all.shape[1] < required_min_columns:
            raise ValueError(
                f"FCDI数据列数不足，至少需要{required_min_columns}列（当前{ fcdi_all.shape[1] }列）\n"
                f"提示：若实际文件列数足够，可能是分隔符仍不匹配（如制表符需改为sep='\\t'）"
            )
    except Exception as e:
        raise ValueError(f"通道对与CSV列映射失败：{str(e)}")
    
    # 筛选列并设置列名（逻辑不变）
    required_col_count = 8  # 前8列：alon1, alat1, clear等必要信息
    all_indices = list(range(required_col_count)) + channel_col_indices
    fcdi_df = fcdi_all.iloc[:, all_indices].copy()
    
    required_cols = ['alon1', 'alat1', 'clear', 'lw', 'slw', 'mp', 'ice', 'notsure']
    channel_cols = [f'channel_{x}' for x in channel_pairs]
    fcdi_df.columns = required_cols + channel_cols
    
    print(f"✅ 加载FCDI数据（共{len(fcdi_df)}个经纬度点），已匹配109个通道对列")
    return fcdi_df, channel_cols


# 绘图函数保持不变
def plot_single_point(plot_df, alon, alat, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    
    alon_str = f"{alon:.2f}".replace('.', '_')
    alat_str = f"{alat:.2f}".replace('.', '_')
    save_path = os.path.join(save_dir, f"FCDI_{alon_str}E_{alat_str}N.png")
    
    # 1. 先计算当前点FCDI值的最大/最小值（用于设置x轴范围）
    min_fcdi = plot_df['fcdi_value'].min()
    max_fcdi = plot_df['fcdi_value'].max()
    # 打印范围，确认是否包含负值（可选，用于验证）
    print(f"当前点FCDI值范围：[{min_fcdi:.3f}, {max_fcdi:.3f}]")
    
    # 2. 绘制散点图（原有代码不变）
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
    
    # 3. 图表配置：添加手动设置x轴范围（核心修改）
    plt.xlabel('FCDI值', fontsize=12, fontweight='bold')
    plt.ylabel('通道对高度', fontsize=12, fontweight='bold')
    plt.title(f'FCDI-高度散点图\n({alon:.2f}°E, {alat:.2f}°N)', fontsize=14, fontweight='bold')
    # 手动设置x轴范围：下限=min_fcdi-0.5（比最小值小0.5，留余量），上限=max_fcdi+0.5
    plt.xlim(-15, 15)  # 固定范围为[-20, 20]，根据实际数据调整
    plt.ylim(0,15)
    plt.grid(axis='both', alpha=0.3, linestyle='--')
    plt.tight_layout()
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"📊 已生成：{os.path.basename(save_path)}")


def plot_all_points(fcdi_df, height_df, channel_cols, save_path):
    plt.figure(figsize=(14, 10))
    for idx, row in fcdi_df.iterrows():
        alon = row['alon1']
        alat = row['alat1']
        fcdi_values = row[channel_cols].values
        plot_df = height_df.copy()
        plot_df['fcdi_value'] = fcdi_values
        plt.scatter(
            x=plot_df['fcdi_value'],
            y=plot_df['height'],
            alpha=0.4,
            s=30,
            label=f'({alon:.2f}, {alat:.2f})' if idx < 10 else ""
        )
    plt.xlabel('FCDI值', fontsize=14, fontweight='bold')
    plt.ylabel('通道对高度', fontsize=14, fontweight='bold')
    plt.title(f'所有经纬度点的FCDI-高度分布（共{len(fcdi_df)}个点）', fontsize=16, fontweight='bold')
    plt.grid(axis='both', alpha=0.3, linestyle='--')
    if len(fcdi_df) > 10:
        plt.legend(title='前10个经纬度点', title_fontsize=10, fontsize=8, loc='upper right')
    else:
        plt.legend(fontsize=10, loc='upper right')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"📊 所有点的散点图已保存至：{save_path}")


def main():
    # 参数配置（column_offset已验证正确，无需修改）
    height_file_path = r"D:\exploring_clouds\channel_heights.txt"
    fcdi_csv_path = r"D:\exploring_clouds\data\test202510019.csv"
    save_single_dir = r"D:\exploring_clouds\figure"
    save_all_path = r"D:\exploring_clouds\figure_all.png"
    plot_mode = "single"  # 按需选择"single"或"all"
    column_offset = -2  # 已验证可让所有索引非负，保持不变
    
    try:
        # 加载数据并绘图
        height_df, channel_pairs = load_channel_heights(height_file_path)
        fcdi_df, channel_cols = load_fcdi_data(fcdi_csv_path, channel_pairs, column_offset)
        
        if plot_mode == "single":
            print(f"\n🚀 开始生成每个点的散点图（共{len(fcdi_df)}个）...")
            for idx, row in fcdi_df.iterrows():
                alon = row['alon1']
                alat = row['alat1']
                fcdi_values = row[channel_cols].values  # 109个匹配的FCDI值
                plot_df = height_df.copy()
                plot_df['fcdi_value'] = fcdi_values  # 长度匹配，无报错
                plot_single_point(plot_df, alon, alat, save_single_dir)
            print(f"\n✅ 所有单图生成完成！保存路径：{save_single_dir}")
        
        elif plot_mode == "all":
            print(f"\n🚀 开始生成所有点的散点图...")
            plot_all_points(fcdi_df, height_df, channel_cols, save_all_path)
        
        else:
            raise ValueError("plot_mode只能是'single'或'all'")

    except Exception as e:
        print(f"\n❌ 运行失败：{str(e)}")


if __name__ == "__main__":
    # 中文显示配置
    plt.rcParams["font.family"] = ["SimHei", "Microsoft YaHei", "Heiti TC"]
    plt.rcParams['axes.unicode_minus'] = False
    main()