import os
import cv2
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from sklearn.manifold import TSNE
# 替换 basemap 为 cartopy
from cartopy import crs as ccrs
from cartopy.feature import COASTLINE
import numpy as np
from PIL import Image
import pandas as pd
from tqdm import tqdm
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score
from matplotlib.patches import FancyBboxPatch
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import torch
import seaborn as sns
import cartopy.mpl.ticker as cticker  # 用于设置经纬度刻度

OMP_NUM_THREADS = 2
plt.rcParams['font.sans-serif'] = ['SimHei']  # 显示中文
plt.rcParams['font.serif'] = ['Times New Roman']  # 英文字体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

def draw_label(date, ad):
    path = r"D:\exploring_clouds\Raw_data_230"
    data_ocean = np.load(os.path.join(path, f"{date}_{ad}_ocean_FOV5.npy"))  # 例如 20200525_as_ocean_FOV5.npy
    data_land = np.load(os.path.join(path, f"{date}_{ad}_land_FOV5.npy"))    # 例如 20200525_as_land_FOV5.npy
    data = np.vstack([data_land, data_ocean])
    
    # 标签生成逻辑（与原代码一致）
    label = []
    for line in data:
        if line[5] / line[4] >= 0.95: #0.5-0.9
            label.append(0)
            continue
        flag = -1
        for i in range(4):
            if line[6 + i] / line[4] >= 0.95: #0.8-0.9
                label.append(i + 1)
                flag = 1
                break
        if flag == -1:
            label.append(5)
    
    label_text_path = os.path.join(save_dir, f"label_text_{date}_{ad}.txt")  # 文件名含date和ad，避免重复
    
    # 2. 保存label数组到文本文件（每行一个标签，整数格式）
    np.savetxt(
        fname=label_text_path,  # 保存路径
        X=label,                # 要保存的数组
        fmt="%d",               # 格式：整数（因为label是0-5的整数）
        delimiter="\n"          # 分隔符：换行（每行一个标签，便于阅读）
    )
    print(f"标签文本文件已保存至：{label_text_path}")  # 打印保存路径，方便确认


    # 绘图部分（用 cartopy 替代 basemap）
    plt.figure(figsize=(12, 10))
    # 设置投影为等角圆柱投影（与 basemap 默认投影一致）
    ax = plt.axes(projection=ccrs.PlateCarree())
    
    # 绘制海岸线（替代 basemap.drawcoastlines()）
    ax.add_feature(COASTLINE, linewidth=0.5)
    
    # 散点图（与原代码一致）
    colors_list = ['#f5eec8', '#0260a0', '#3ca4e5', '#38bdea', '#93d2f5', '#d6daf7']
    point_colors = [colors_list[val] for val in label]
    plt.scatter(data[:, 0], data[:, 1], c=point_colors, s=2, transform=ccrs.PlateCarree())
    
    # 设置经纬度范围（与原代码一致）
    ax.set_xlim(-180, 180)
    ax.set_ylim(-60, 60)
    
    # 设置经纬度刻度（替代原 plt.xticks/plt.yticks）
    ax.set_xticks([-180, -120, -60, 0, 60, 120, 180], crs=ccrs.PlateCarree())
    ax.set_yticks([-60, -40, -20, 0, 20, 40, 60], crs=ccrs.PlateCarree())
    
    # 设置刻度标签（与原代码一致）
    ax.xaxis.set_major_formatter(cticker.LongitudeFormatter())  # 自动处理经度标签（如120W）
    ax.yaxis.set_major_formatter(cticker.LatitudeFormatter())   # 自动处理纬度标签（如60N）
    ax.tick_params(axis='both', labelsize=18)  # 设置刻度字体大小
    
    plt.tight_layout()
    plt.savefig(r'D:\exploring_clouds\data\label.png', bbox_inches='tight', dpi=300)
    plt.show()

if __name__ == '__main__':
    save_dir = r"D:\exploring_clouds\data"
    date = '20200525'
    ad = 'as'
    draw_label(date, ad)