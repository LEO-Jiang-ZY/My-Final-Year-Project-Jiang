import os
import cv2
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from sklearn.manifold import TSNE
from mpl_toolkits.basemap import Basemap
import numpy as np
from PIL import Image
import pandas as pd
from tqdm import tqdm
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.metrics import calinski_harabasz_score,davies_bouldin_score
from matplotlib.patches import FancyBboxPatch
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import torch
import seaborn as sns
OMP_NUM_THREADS=2
plt.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体（这里是黑体，用于显示中文）
plt.rcParams['font.serif'] = ['Times New Roman']  # 英文字体为新罗马字体
plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像时负号'-'显示为方块的问题

def draw_label(date,ad):
    path=r"D:\exploring_clouds\Raw_data_230"
    data_ocean=np.load((os.path.join(path,date+'_'+ad+'_ocean_FOV5.npy')))
    data_land=np.load((os.path.join(path,date+'_'+ad+'_land_FOV5.npy')))
    data=np.vstack([data_land,data_ocean])
    label=[]
    for line in data:#0晴空，
        if line[5]/line[4]>=0.5:
            label.append(0)
            continue
        flag=-1
        for i in range(4):
            if line[6+i]/line[4]>=0.8:
                label.append(i+1)
                flag=1
                break
        if flag==-1:
            label.append(5)
    plt.figure(figsize=(12, 10))
    colors = ['#f5eec8',
              '#0260a0',
              '#3ca4e5',
              '#38bdea',
              '#93d2f5',
              '#d6daf7']
    point_colors = [colors[val] for val in label]
    plt.scatter(data[:,0],data[:,1],c=point_colors,s=2)

    map1 = Basemap()
    map1.drawcoastlines()
    plt.xlim((-180, 180))
    plt.ylim((-60, 60))
    y_ticks = ['60S', '40S', '20S', '0', '20N', '40N', '60N']
    x_ticks = ['180', '120W', '60W', '0', '60E', '120E', '180']
    yt = [-60, -40, -20, 0, 20, 40, 60]
    xt = [-180, -120, -60, 0, 60, 120, 180]
    plt.xticks(xt, x_ticks, rotation=0, fontsize=18)
    plt.yticks(yt, y_ticks, rotation=0, fontsize=18)
    plt.tight_layout()
    plt.savefig(r'D:\exploring_clouds\data\label.png',bbox_inches='tight',dpi=300)
    plt.show()



if __name__=='__main__':
    date='20210601'
    ad='A'
    draw_label(date,ad)
    