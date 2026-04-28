#海陆分开分析 总+海+陆
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm  # 导入进度条库

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


def load_pure_cloud_data(folder_path, surface_type="All"):
    """
    加载纯净云数据。
    surface_type: "All"(所有), "land"(仅陆地), "ocean"(仅海洋)
    """
    all_data = []
    all_labels = []

    # 筛选符合海陆条件的文件
    file_list = [f for f in os.listdir(folder_path) if f.endswith('.npy')]
    if surface_type == "land":
        file_list = [f for f in file_list if "land" in f.lower()]
    elif surface_type == "ocean":
        file_list = [f for f in file_list if "ocean" in f.lower()]

    if len(file_list) == 0:
        print(f"⚠️ 未找到符合 {surface_type} 条件的数据文件！")
        return None, None

    # 使用 tqdm 显示读取进度条
    print(f"\n📂 正在读取 [{surface_type}] 场景下的数据文件...")
    for file_name in tqdm(file_list, desc=f"Loading {surface_type} data", unit="file"):
        file_path = os.path.join(folder_path, file_name)
        data = np.load(file_path)

        valid_mask = data[:, 4] > 0
        data = data[valid_mask]
        totals = data[:, 4]

        labels = np.full(len(data), 5)

        for i in range(4):
            cloud_ratio = data[:, 6 + i] / totals
            condition = (labels == 5) & (cloud_ratio >= 0.8)
            labels[condition] = i + 1

        fcdi_data = data[:, 11:237]  # 1-226通道对
        all_data.append(fcdi_data)
        all_labels.append(labels)

    final_fcdi = np.vstack(all_data)
    final_labels = np.concatenate(all_labels)

    cloud_mask = (final_labels >= 1) & (final_labels <= 4)
    X_pure = final_fcdi[cloud_mask]
    y_pure = final_labels[cloud_mask]

    print(f"✅ [{surface_type}] 数据加载完毕！共提取纯净云样本 {len(X_pure)} 个。")
    return X_pure, y_pure


def analyze_and_export(X, y, height_csv_path, surface_type="All"):
    """执行 OvR 稳定性筛选并导出表格和图表"""
    df_height = pd.read_csv(height_csv_path, encoding='gbk')
    heights = df_height['平均高度'].values[:226]
    cloud_names = {1: '水云(LW)', 2: '过冷水云(SLW)', 3: '混合云(MP)', 4: '冰云(Ice)'}

    results_df = pd.DataFrame({
        '通道编号': np.arange(1, 227),
        '平均高度(km)': heights
    })

    scaler = MinMaxScaler()
    top10_dict = {}
    random_seeds = [42, 123, 888, 2026, 9999]

    print(f"\n🔥 开始计算 [{surface_type}] 场景下各云相态的专属敏感度...")

    # 遍历四种云相态，加入进度条
    for target_class in tqdm([1, 2, 3, 4], desc=f"Analyzing clouds ({surface_type})", unit="class"):
        c_name = cloud_names[target_class]

        # 1. MAD
        class_X = X[y == target_class]
        mad_scores = np.mean(np.abs(class_X), axis=0) if len(class_X) > 0 else np.zeros(X.shape[1])

        # 2. 互信息 MI
        y_binary = (y == target_class).astype(int)
        mi_scores = mutual_info_classif(X, y_binary, random_state=42)

        # 3. 随机森林 (5次稳定性筛选)
        rf_scores_all_runs = []
        for seed in random_seeds:
            rf = RandomForestClassifier(n_estimators=100, random_state=seed, n_jobs=-1)
            rf.fit(X, y_binary)
            rf_scores_all_runs.append(rf.feature_importances_)
        rf_scores_stable = np.mean(rf_scores_all_runs, axis=0)

        # 4. 融合得分
        mad_norm = scaler.fit_transform(mad_scores.reshape(-1, 1)).flatten()
        mi_norm = scaler.fit_transform(mi_scores.reshape(-1, 1)).flatten()
        rf_norm = scaler.fit_transform(rf_scores_stable.reshape(-1, 1)).flatten()

        comp_scores = (mad_norm + mi_norm + rf_norm) / 3.0
        results_df[f'{c_name}_综合得分'] = comp_scores

        # 提取Top10
        temp_df = pd.DataFrame({'通道编号': np.arange(1, 227), '高度': heights, '得分': comp_scores})
        top10_dict[c_name] = temp_df.sort_values(by='得分', ascending=False).head(10)

    # 保存表格
    csv_filename = f'{surface_type}_FCDI各云相敏感度排名表.csv'
    results_df.to_csv(csv_filename, index=False, encoding='utf-8-sig')

    # 打印该场景的结论
    print("\n" + "=" * 65)
    print(f"🏆 【{surface_type} 场景】每种云相态专属的最敏感 Top 10 通道")
    print("=" * 65)
    for target_class in [1, 2, 3, 4]:
        c_name = cloud_names[target_class]
        print(f"\n【{c_name}】Top 10:")
        for rank, (_, row) in enumerate(top10_dict[c_name].iterrows(), 1):
            print(
                f"  Top {rank:2d}: 通道对 [{int(row['通道编号']):3d}] | 高度: {row['高度']:5.2f} km | 得分: {row['得分']:.4f}")

    # ================= 绘制对比折线图 =================
    plt.figure(figsize=(15, 6))
    colors = ['blue', 'cyan', 'green', 'red']
    for target_class, color in zip([1, 2, 3, 4], colors):
        c_name = cloud_names[target_class]
        plt.plot(range(1, 227), results_df[f'{c_name}_综合得分'], label=c_name, color=color, linewidth=1.5)

    plt.title(f'[{surface_type} 场景] 四种云相态专属敏感度得分曲线 (多指标融合)')
    plt.xlabel('通道对编号 (1 - 226)')
    plt.ylabel('综合敏感度得分 (0-1)')
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    img_name = f'{surface_type}_Sensitivity_Lineplot.png'
    plt.savefig(img_name, dpi=300)
    plt.close()  # 关闭当前画布，防止与下一个场景重叠

# def analyze_and_export(X, y, height_csv_path, surface_type="All"):
#     """执行 OvR 稳定性筛选并导出表格和图表 (含极速降采样优化)"""
#     df_height = pd.read_csv(height_csv_path, encoding='gbk')
#     heights = df_height['平均高度'].values[:226]
#     cloud_names = {1: '水云(LW)', 2: '过冷水云(SLW)', 3: '混合云(MP)', 4: '冰云(Ice)'}
#
#     results_df = pd.DataFrame({
#         '通道编号': np.arange(1, 227),
#         '平均高度(km)': heights
#     })
#
#     scaler = MinMaxScaler()
#     top10_dict = {}
#     random_seeds = [42, 123, 888, 2026, 9999]
#
#     # === 极速加速核心：设置最大采样数 ===
#     # 特征评估抽取 3 万条数据足够代表全局规律，极大提升运行速度
#     MAX_SAMPLES = 30000
#
#     print(f"\n🔥 开始计算 [{surface_type}] 场景下各云相态的专属敏感度...")
#
#     for target_class in tqdm([1, 2, 3, 4], desc=f"Analyzing {surface_type} clouds", unit="class"):
#         c_name = cloud_names[target_class]
#
#         # 1. MAD (物理指标，计算极快，使用全部数据)
#         class_X = X[y == target_class]
#         mad_scores = np.mean(np.abs(class_X), axis=0) if len(class_X) > 0 else np.zeros(X.shape[1])
#
#         # 2. 准备 AI 模型的 OvR 标签 (目标云=1, 其他云=0)
#         y_binary = (y == target_class).astype(int)
#
#         # === 降采样逻辑：如果数据量大于 3万，则随机抽取 3万条 ===
#         if len(X) > MAX_SAMPLES:
#             np.random.seed(42)  # 固定随机种子，保证每次运行结果一致
#             sampled_indices = np.random.choice(len(X), MAX_SAMPLES, replace=False)
#             X_ai = X[sampled_indices]
#             y_binary_ai = y_binary[sampled_indices]
#         else:
#             X_ai = X
#             y_binary_ai = y_binary
#
#         # tqdm.write 可以防止打印文字打断进度条的显示
#         tqdm.write(f"   -> 正在使用 {len(X_ai)} 条样本计算 [{c_name}] 的互信息 (MI)...")
#         mi_scores = mutual_info_classif(X_ai, y_binary_ai, random_state=42)
#
#         tqdm.write(f"   -> 正在计算 [{c_name}] 的随机森林稳定性重要性 (5次)...")
#         rf_scores_all_runs = []
#         for seed in random_seeds:
#             rf = RandomForestClassifier(n_estimators=100, random_state=seed, n_jobs=-1)
#             rf.fit(X_ai, y_binary_ai)
#             rf_scores_all_runs.append(rf.feature_importances_)
#         rf_scores_stable = np.mean(rf_scores_all_runs, axis=0)
#
#         # 3. 归一化与加权融合
#         mad_norm = scaler.fit_transform(mad_scores.reshape(-1, 1)).flatten()
#         mi_norm = scaler.fit_transform(mi_scores.reshape(-1, 1)).flatten()
#         rf_norm = scaler.fit_transform(rf_scores_stable.reshape(-1, 1)).flatten()
#
#         comp_scores = (mad_norm + mi_norm + rf_norm) / 3.0
#         results_df[f'{c_name}_综合得分'] = comp_scores
#
#         # 提取Top10
#         temp_df = pd.DataFrame({'通道编号': np.arange(1, 227), '高度': heights, '得分': comp_scores})
#         top10_dict[c_name] = temp_df.sort_values(by='得分', ascending=False).head(10)
#
#     # 保存表格
#     csv_filename = f'{surface_type}_FCDI各云相敏感度排名表.csv'
#     results_df.to_csv(csv_filename, index=False, encoding='utf-8-sig')
#
#     # 打印该场景的结论
#     print("\n" + "=" * 65)
#     print(f"🏆 【{surface_type} 场景】每种云相态专属的最敏感 Top 10 通道")
#     print("=" * 65)
#     for target_class in [1, 2, 3, 4]:
#         c_name = cloud_names[target_class]
#         print(f"\n【{c_name}】Top 10:")
#         for rank, (_, row) in enumerate(top10_dict[c_name].iterrows(), 1):
#             print(
#                 f"  Top {rank:2d}: 通道对 [{int(row['通道编号']):3d}] | 高度: {row['高度']:5.2f} km | 得分: {row['得分']:.4f}")
#
#     # ================= 绘制对比折线图 =================
#     plt.figure(figsize=(15, 6))
#     colors = ['blue', 'cyan', 'green', 'red']
#     for target_class, color in zip([1, 2, 3, 4], colors):
#         c_name = cloud_names[target_class]
#         plt.plot(range(1, 227), results_df[f'{c_name}_综合得分'], label=c_name, color=color, linewidth=1.5)
#
#     plt.title(f'[{surface_type} 场景] 四种云相态专属敏感度得分曲线 (多指标融合)')
#     plt.xlabel('通道对编号 (1 - 226)')
#     plt.ylabel('综合敏感度得分 (0-1)')
#     plt.legend()
#     plt.grid(True, linestyle=':', alpha=0.6)
#     plt.tight_layout()
#     img_name = f'{surface_type}_Sensitivity_Lineplot.png'
#     plt.savefig(img_name, dpi=300)
#     plt.close()  # 关闭当前画布


if __name__ == '__main__':
    # 你的文件路径
    folder_path = r"E:\A_exploring_clouds\Raw_data_230"
    height_csv_path = r"E:\A_exploring_clouds\data\channel_pair_average_height.csv"

    # 三重循环：分别分析 综合(All)、陆地(land)、海洋(ocean)
    for surface in ["All", "land", "ocean"]:
        print("\n" + "#" * 70)
        print(f"🚀🚀🚀 正在启动大类分析流程 ---> 目标场景：【{surface}】 🚀🚀🚀")
        print("#" * 70)

        X_cloud, y_cloud = load_pure_cloud_data(folder_path, surface_type=surface)

        if X_cloud is not None and len(X_cloud) > 0:
            analyze_and_export(X_cloud, y_cloud, height_csv_path, surface_type=surface)
        else:
            print(f"❌ 跳过 {surface} 场景的分析，因为没有提取到足够的数据。")

    print("\n🎉🎉🎉 全部分析流程已完美结束！请查看生成的 CSV 和 PNG 文件。")