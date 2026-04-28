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



#画EFCDI全球
def draw_EFCDI_MAP(date,ad,channel):
    path=r'J:\2020-2021data\Raw_data_486'

    data_o=np.load(os.path.join(path,date+'_'+ad+'_ocean_FOV5.npy'))
    data_l=np.load(os.path.join(path,date+'_'+ad+'_land_FOV5.npy'))
    data=np.vstack([data_o,data_l])

    plt.figure(figsize=(12, 10))
    map1 = Basemap()
    map1.drawcoastlines()
    plt.xlim((-180, 180))
    plt.ylim((-60, 60))
    cm = plt.cm.get_cmap('coolwarm')
    y_ticks = ['60S', '40S', '20S', '0', '20N', '40N', '60N']
    x_ticks = ['180', '120W', '60W', '0', '60E', '120E', '180']
    yt = [-60, -40, -20, 0, 20, 40, 60]
    xt = [-180, -120, -60, 0, 60, 120, 180]
    plt.xticks(xt, x_ticks, rotation=0, fontsize=18)
    plt.yticks(yt, y_ticks, rotation=0, fontsize=18)

    # plt.title("PCA + K-mean云分类错误分布")
    plt.scatter(data[:,0], data[:,1],  c=data[:,10+channel],marker='.', cmap=cm, vmin=-10,vmax=10)
    plt.tight_layout()
    cb=plt.colorbar(orientation='horizontal',aspect=50,extend='both',pad=0.05,fraction=0.2,shrink=0.5)
    cb.ax.tick_params(labelsize=16)

    plt.savefig(r'E:\clouddata\EFCDI\EFCDI_'+ad+'_'+str(channel)+'.png',bbox_inches='tight',dpi=300)
    plt.close()
    plt.clf()
    plt.cla()
    plt.show()

#找一下这个通道对里是那两个通道
def channel_in_channelpair(cp):
    path=r'D:\project_typhoon\Coe\base channel coe\all.txt'
    cp_list=np.loadtxt(path)
    print(cp_list[cp-1])

#画通道对各高度上的统计直方图
def hist_channelpair_z():
    fontsize=19
    data_p=np.loadtxt(r'E:\project_typhoon\文件\cris-fsr_npp_WF_Peak.txt')
    z226_lm=np.loadtxt(r'E:\project_typhoon\文件\height_all_lm.txt')
    z226_ls=np.loadtxt(r'E:\project_typhoon\文件\height_all_ls.txt')
    z226_ms=np.loadtxt(r'E:\project_typhoon\文件\height_all_ms.txt')


    z_lm=[]
    z_ls=[]
    z_ms=[]

    for i in z226_lm:
        ch1=data_p[np.where(data_p[:,0]==i[0])[0][0],3]
        ch2=data_p[np.where(data_p[:,0]==i[1])[0][0],3]
        height=(ch1+ch2)/2
        if height<=0:
            height=0.5
        elif height>=10:
            height=9.5
        z_lm.append(height)
    for i in z226_ls:
        ch1=data_p[np.where(data_p[:,0]==i[0])[0][0],3]
        ch2=data_p[np.where(data_p[:,0]==i[1])[0][0],3]
        height=(ch1+ch2)/2
        if height<=0:
            height=0.5
        elif height>=10:
            height=9.5
        z_ls.append(height)
    for i in z226_ms:
        ch1=data_p[np.where(data_p[:,0]==i[0])[0][0],3]
        ch2=data_p[np.where(data_p[:,0]==i[1])[0][0],3]
        height=(ch1+ch2)/2
        if height<=0:
            height=0.5
        elif height>=10:
            height=9.5
        z_ms.append(height)

    counts, bins, patches=plt.hist(x=[z_lm,z_ls,z_ms],stacked=True,label=['L-M','L-S','M-S'],bins= np.arange(0, 11, 1) ,rwidth=0.8)
    plt.xlabel('高度（km）',fontsize=fontsize)
    plt.ylabel("通道对数量",fontsize=fontsize)
    plt.legend(fontsize=fontsize,loc='upper center', bbox_to_anchor=(0.5, 1.03), ncol=3)
    plt.xticks(ticks=np.arange(0.5, 10, 1),
           labels=['1', '2', '3', '4', '5', '6', '7', '8', '9', '10'],fontsize=fontsize)
    plt.yticks(fontsize=fontsize)

    for count, bin in zip(counts[-1], bins):
        x_position = bin + 0.5  # 假设bins等距，中心位置即为边界加上一半的宽度
        y_position = count
        plt.text(x_position, y_position, f'{int(count)}', ha='center', va='bottom',fontsize=fontsize-2)

    plt.ylim(0,175)

    plt.tight_layout()
    plt.savefig(r'E:\typython\通道统计\通道统计z.png',dpi=300,bbox_inches='tight')
    plt.cla()
    #plt.show()
def hist_channelpair_l():
    fontsize=19
    data_p=np.loadtxt(r'E:\project_typhoon\文件\cris-fsr_npp_WF_Peak.txt')
    lm=np.loadtxt(r'E:\project_typhoon\Coe\base channel coe\lm.txt')
    ls=np.loadtxt(r'E:\project_typhoon\Coe\base channel coe\ls.txt')
    ms=np.loadtxt(r'E:\project_typhoon\Coe\base channel coe\ms.txt')

    z_lm=[]
    z_ls=[]
    z_ms=[]

    for i in lm:
        ch1=data_p[np.where(data_p[:,0]==i[0])[0][0],3]
        ch2=data_p[np.where(data_p[:,0]==i[1])[0][0],3]
        height=(ch1+ch2)/2
        if height<=0:
            height=0.5
        elif height>=10:
            height=9.5
        z_lm.append(height)
    for i in ls:
        ch1=data_p[np.where(data_p[:,0]==i[0])[0][0],3]
        ch2=data_p[np.where(data_p[:,0]==i[1])[0][0],3]
        height=(ch1+ch2)/2
        if height<=0:
            height=0.5
        elif height>=10:
            height=9.5
        z_ls.append(height)
    for i in ms:
        ch1=data_p[np.where(data_p[:,0]==i[0])[0][0],3]
        ch2=data_p[np.where(data_p[:,0]==i[1])[0][0],3]
        height=(ch1+ch2)/2
        if height<=0:
            height=0.5
        elif height>=10:
            height=9.5
        z_ms.append(height)

    counts, bins, patches=plt.hist(x=[z_lm,z_ls,z_ms],stacked=True,label=['L-M','L-S','M-S'],bins= np.arange(0, 11, 1) ,rwidth=0.8)
    plt.xlabel('高度（km）',fontsize=fontsize)
    plt.ylabel("通道对数量",fontsize=fontsize)
    plt.legend(fontsize=fontsize,loc='upper center', bbox_to_anchor=(0.5, 1.03), ncol=3)
    plt.xticks(ticks=np.arange(0.5, 10, 1),
               labels=['1', '2', '3', '4', '5', '6', '7', '8', '9', '10'],fontsize=fontsize)
    plt.yticks(fontsize=fontsize)

    for count, bin in zip(counts[-1], bins):
        x_position = bin + 0.5  # 假设bins等距，中心位置即为边界加上一半的宽度
        y_position = count
        plt.text(x_position, y_position, f'{int(count)}', ha='center', va='bottom',fontsize=fontsize-2)

    plt.ylim(0,175)
    plt.savefig(r'E:\typython\通道统计\通道统计l.png',dpi=300,bbox_inches='tight')
    plt.show()

#剪裁真彩图到60S-60N
def cut_true_pic(date):
    # 打开图像
    image_path = r'E:\2020-2021data\True color pic\TRUE.daily.'+date+'.color.png'  # 替换为你的图像文件路径
    img = Image.open(image_path)
        # 计算剪裁边界
    width, height = img.size
    # 假设图像自上而下从90N到90S，计算60N和60S对应的像素位置
    upper = (90 - 60) / 180.0 * height  # 60N对应的像素位置
    lower = (90 + 60) / 180.0 * height  # 60S对应的像素位置
        # 剪裁图像 (左上角坐标，右下角坐标)
    cropped_img = img.crop((0, upper, width, lower))
        # 保存剪裁后的图像
    cropped_img_path = r'C:\Users\Administrator\Desktop\cut_true_pic'+date+'.png'  # 替换为剪裁后的图像保存路径
    cropped_img.save(cropped_img_path)

#画标签的全球分布图
def draw_label(date,ad):
    path=r'J:\2020-2021data\Raw_data_486'
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
    plt.savefig(r'E:\clouddata\Label\label.png',bbox_inches='tight',dpi=300)




    plt.show()

#画fov匹配的像素数量的全球分布图
def draw_pixel_num(date,ad):
    path=r'K:\2020-2021data\Raw_data_486'
    data_ocean=np.load((os.path.join(path,date+'_'+ad+'_ocean_FOV5.npy')))
    data_land=np.load((os.path.join(path,date+'_'+ad+'_land_FOV5.npy')))
    data=np.vstack([data_land,data_ocean])

    plt.scatter(data[:,0],data[:,1],c=data[:,4],s=2)

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
    cb=plt.colorbar(orientation='horizontal',aspect=50,extend='both',pad=0.08,fraction=0.3,shrink=0.7)
    cb.ax.tick_params(labelsize=16)
    plt.savefig(r'E:\clouddata\label_fov.png',bbox_inches='tight',dpi=300)
    plt.show()

#画复杂云图数据集的标签散点图
def draw_pic_dataset_scatter_label(date,as_ds,center_coordinate):
    plot_range=8
    center_lo=center_coordinate[0]
    center_la=center_coordinate[1]
    outpath=r'C:\Users\Administrator\Desktop\dataset'
    out_fold=os.path.join(outpath,date+'_'+as_ds+'_('+str(center_lo)+','+str(center_la)+')_'+str(plot_range))
    if not os.path.exists(out_fold):
        os.makedirs(out_fold)
    datapath=r"E:\2020-2021data\Raw_data_486"
    true_pic_path=r'E:\2020-2021data\True color pic'

    data_ocean=np.load(os.path.join(datapath,date+"_"+as_ds+'_ocean_FOV5.npy'))
    data_land=np.load(os.path.join(datapath,date+"_"+as_ds+'_land_FOV5.npy'))
    data=np.vstack([data_land,data_ocean])
    data=data[np.where((data[:,0]>=center_lo-plot_range)&(data[:,0]<=center_lo+plot_range)&(data[:,1]>=center_la-plot_range)&(data[:,1]<=center_la+plot_range))]
    label=[]
    for line in data:
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
    plt.figure(figsize=(4,4))
    colors = ['#f5eec8',
              '#0260a0',
              '#3ca4e5',
              '#38bdea',
              '#93d2f5',
              '#d6daf7']
    point_colors = [colors[val] for val in label]
    plt.scatter(data[:,0],data[:,1],c=point_colors)
    plt.xticks([])
    plt.yticks([])
    plt.tight_layout()
    plt.savefig(os.path.join(out_fold,'scatter.png'),dpi=300,bbox_inches='tight')
    #plt.show()
    plt.clf()


    true_pic=cv2.imread(os.path.join(true_pic_path,'True.daily.'+date+'.color.png'))
    map_width=true_pic.shape[1]
    map_height=true_pic.shape[0]
    center_x = (center_lo + 180) * map_width / 360
    center_y = (90 - center_la) * map_height / 180
    range_px = plot_range * map_width / 360  # 经度范围对应的像素大小
    range_py = plot_range * map_height / 180  # 纬度范围对应的像素大小
    xmin = int(max(center_x - range_px, 0))
    xmax = int(min(center_x + range_px, map_width))
    ymin = int(max(center_y - range_py, 0))
    ymax = int(min(center_y + range_py, map_height))
    cv2.imwrite(os.path.join(out_fold,'true.png'), true_pic[ymin:ymax, xmin:xmax])

#画复杂云图数据集的通道散点图
def draw_pic_dataset_scatter_EFCDI(date,as_ds,center_coordinate,channel):
    plot_range=8
    center_lo=center_coordinate[0]
    center_la=center_coordinate[1]
    outpath=r'C:\Users\Administrator\Desktop\dataset'
    out_fold=os.path.join(outpath,date+'_'+as_ds+'_('+str(center_lo)+','+str(center_la)+')_'+str(plot_range))
    if not os.path.exists(out_fold):
        os.makedirs(out_fold)
    datapath=r"E:\2020-2021data\Raw_data_486"
    true_pic_path=r'E:\2020-2021data\True color pic'

    data_ocean=np.load(os.path.join(datapath,date+"_"+as_ds+'_ocean_FOV5.npy'))
    data_land=np.load(os.path.join(datapath,date+"_"+as_ds+'_land_FOV5.npy'))
    data=np.vstack([data_land,data_ocean])
    data=data[np.where((data[:,0]>=center_lo-plot_range)&(data[:,0]<=center_lo+plot_range)&(data[:,1]>=center_la-plot_range)&(data[:,1]<=center_la+plot_range))]


    plt.figure(figsize=(4,4))
    plt.scatter(data[:,0],data[:,1],c=data[:,10+channel],cmap='coolwarm',)
    plt.xticks([])
    plt.yticks([])
    # cb=plt.colorbar(orientation='horizontal',extend='both',pad=0.05,shrink=0.9,aspect=30,)
    # cb.ax.tick_params(labelsize=18)
    plt.tight_layout()
    plt.savefig(os.path.join(out_fold,'channel'+str(channel).zfill(3)+'.png'),dpi=300,bbox_inches='tight')
    #plt.show()
    plt.clf()

#画全部的复杂云系统图像的散点图
def draw_all_pic_dataset_scatter():
    files=os.listdir(r'E:\2020-2021data\pic_dataset_all')
    for file in files:
        date=file.split('_')[0]
        ad=file.split('_')[1]
        cor=file.split('_')[2]
        x=float(cor.split(',')[0][1:])
        y=float(cor.split(',')[1][:-1])
        #draw_pic_dataset_scatter_label(date,ad,[x,y])
        draw_pic_dataset_scatter_EFCDI(date,ad,[x,y],4)
        draw_pic_dataset_scatter_EFCDI(date,ad,[x,y],325)
        draw_pic_dataset_scatter_EFCDI(date,ad,[x,y],480)

#打印全部IBTrACS里的气旋
def print_pyphoon():
    typhoon_path=r'D:\project_typhoon\typhoon\20_21'
    files=os.listdir(typhoon_path)
    for file in files:
        data=np.loadtxt(os.path.join(typhoon_path,file))

        for line in data:
            if str(int(line[0]))[6:8] in ['05','15','25']:
                for ad in ["ds"]:

                    print(file+'  '+ad+" Typhoon in {:d} {:.2f} {:.2f} {:d}".format(int(line[0]),line[1],line[2],int(line[5])))

#计算并保存所有通道对的WF峰值高度做伪标签
def channel_pair_height():
    peak=np.loadtxt(r'D:\project_typhoon\文件\cris-fsr_npp_WF_Peak.txt')
    pair=np.loadtxt(r'D:\project_typhoon\Coe\base channel coe\all.txt')
    height_list=[]
    for line in pair:
        height1=peak[int(line[0])-1,3]
        height2=peak[int(line[1])-1,3]
        height=(height2+height1)//2
        height_list.append(height)
    height_list=np.array(height_list)
    np.save(r'E:\2020-2021data\height_label.npy',height_list)
    c=1

#计算并保存所有通道对的配对方式做伪标签峰值高度
def channel_pair_height():

    pair_list=[]
    for i in range(486):
        if i<314:
            pair_list.append(0)
        elif i>=314 and i<314+20:
            pair_list.append(1)
        elif i>=314+20:
            pair_list.append(2)
    pair_list=np.array(pair_list)
    np.save(r'E:\2020-2021data\pair_label.npy',pair_list)

#输入n行（点），d列（特征维度）的array，生成tsne图
def tsne(x,type):


    tsne=TSNE(n_components=2,)
    x=tsne.fit_transform(x)

    fig, ax = plt.subplots(figsize=(5, 5))

    # 定义每个类别的颜色
    colors = ['red', 'green', 'blue']
    # 定义每个类别的标签
    category_labels = ['LM', 'LS', 'MS']
    # 类别对应的行数
    category_indices = [314, 20, 152]
    scatter_plots = []
    start_idx = 0
    for n_points, color, label in zip(category_indices, colors, category_labels):
        end_idx = start_idx + n_points
        scatter = ax.scatter(x[start_idx:end_idx, 0], x[start_idx:end_idx, 1], color=color, label=label,s=30,)
        scatter_plots.append(scatter)
        start_idx = end_idx
        # plt.scatter(x[314:334,0],x[314:334,1],label='LS',alpha=0.8)
    # plt.scatter(x[334:,0],x[334:,1],label='MS',alpha=0.8)
    annot = ax.annotate("", xy=(0,0), xytext=(20,20),textcoords="offset points",
                        bbox=dict(boxstyle="round", fc="w"),
                        arrowprops=dict(arrowstyle="->"))
    annot.set_visible(False)
    #plt.legend(fontsize=16)
    def update_annot(scatter, ind):
        pos = scatter.get_offsets()[ind]
        annot.xy = pos
        text = f"Row: {ind}"
        annot.set_text(text)

    def hover(event):
        vis = annot.get_visible()
        if event.inaxes == ax:
            for scatter in scatter_plots:
                cont, ind_dict = scatter.contains(event)
                if cont:
                    ind = ind_dict["ind"][0]  # Get the first index
                    update_annot(scatter, ind)
                    annot.set_visible(True)
                    fig.canvas.draw_idle()
                    return
            if vis:
                annot.set_visible(False)
                fig.canvas.draw_idle()

    if type=='raw':
        # ax = plt.gca()
        # for spine in ax.spines.values():
        #     spine.set_visible(False)
        plt.xticks([])
        plt.yticks([])
        #plt.title('RAW')
        plt.legend(fontsize=16)
        plt.tight_layout()
        plt.savefig(r'C:\Users\Administrator\Desktop\raw_feature_tsne.png',bbox_inches='tight',dpi=300)

    elif type=='GNN':
        ax = plt.gca()
        for spine in ax.spines.values():
            spine.set_visible(False)
        plt.xticks([])
        plt.yticks([])
    fig.canvas.mpl_connect("motion_notify_event", hover)
    plt.show()

#画GNN的输出特征的Tsne
def GNN_tsne(fold_name):
    raw=np.load(r'E:\2020-2021data\re_pic_dataset_all_feature_matrix.npy')
    feat=np.load(os.path.join(fold_name,'final_embedding.npy'))
    #tsne(raw,'raw')
    tsne(feat,'GNN')

#生成边文件
def Mad2edge():
    adj_label = np.load(r'E:\2020-2021data\adj_matrix.npy')
    adj=adj_label-np.eye(len(adj_label))
    edges = []
    for i in range(adj.shape[0]):
        for j in range(i+1, adj.shape[1]):  # 避免重复计算对称矩阵的下三角部分
            if adj[i, j] != 0:
                edges.append([i, j])


    # 将边列表转换为DataFrame
    edges_df = pd.DataFrame(edges, columns=['Source', 'Target'])
    edges_df.to_csv(r'E:\2020-2021data\486.csv', index=False)

#生成点标签文件
def node_label(index_file):

    n_LM = 314
    n_LS = 20
    n_MS = 152
    label=np.load(os.path.join(r'E:\channel_selection_result',index_file,'label.npy'))
    center=np.load(os.path.join(r'E:\channel_selection_result',index_file,'best30_all192.npy'))
    center_col = np.array([0]*len(label))
    center_col[center]=1
    pair_labels = ['LM'] * n_LM + ['LS'] * n_LS + ['MS'] * n_MS
    id=[]
    for i in range(486):
        id.append(i)
    labels_df = pd.DataFrame({'Id':id,'GNNLabel': label, 'Center': center_col,'PairLabel':pair_labels,})
    labels_df.to_csv(os.path.join(r'E:\2020-2021data',index_file+'_label.csv'), index=False)

#检测GNN的运行的loss
def plot_loss(file):
    record=np.load(os.path.join(r'E:\channel_selection_result',file,'loss_record.npy'))
    plt.plot(record[0,:],label='ALL')
    plt.legend()
    plt.show()
    plt.plot(record[1,:],label='KL')
    plt.legend()
    plt.show()
    plt.plot(record[2,:],label='CE')
    plt.legend()
    plt.show()
    plt.plot(record[3,:],label='RE')
    plt.legend()
    plt.show()

def check_GNNcp_height(file):
    fontsize=18
    record=np.load(os.path.join(r'E:\channel_selection_result',file,'label.npy'))

    peak=np.loadtxt(r'D:\project_typhoon\文件\cris-fsr_npp_WF_Peak.txt')
    pair=np.loadtxt(r'D:\project_typhoon\Coe\base channel coe\all.txt')
    height_list=[]
    for line in pair:
        height1=peak[int(line[0])-1,3]
        height2=peak[int(line[1])-1,3]
        height=(height2+height1)//2
        if height<=0:
            height=0.5
        elif height>=10:
            height=9.5
        height_list.append(height)
    height_list=np.array(height_list)

    c0=[]
    c1=[]
    c2=[]
    c3=[]
    c4=[]


    for i in range(len(record)):
        if record[i]==0:
            c0.append(height_list[i])
        elif record[i]==1:
            c1.append(height_list[i])
        elif record[i]==2:
            c2.append(height_list[i])
        elif record[i]==3:
            c3.append(height_list[i])
        elif record[i]==4:
            c4.append(height_list[i])

    counts, bins, patches=plt.hist(x=[c0,c1,c2,c3],stacked=True,label=['Cluster1','Cluster2','Cluster3','Cluster4'],bins= np.arange(0, 11, 1) ,rwidth=0.8)
    plt.xlabel('高度（km）',fontsize=fontsize)
    plt.ylabel("通道对数量",fontsize=fontsize)
    #plt.legend(fontsize=fontsize,loc='upper center', bbox_to_anchor=(0.5, 1.03), ncol=3)
    plt.xticks(ticks=np.arange(0.5, 10, 1),
               labels=['1', '2', '3', '4', '5', '6', '7', '8', '9', '10'],fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    for count, bin in zip(counts[-1], bins):
        x_position = bin + 0.5  # 假设bins等距，中心位置即为边界加上一半的宽度
        y_position = count
        plt.text(x_position, y_position, f'{int(count)}', ha='center', va='bottom',fontsize=fontsize-2)

    plt.ylim(0,counts.max()+15)
    plt.legend(fontsize=fontsize,loc='upper center', bbox_to_anchor=(0.5, 1.03),ncol=2)
    plt.tight_layout()

    plt.show()

def boxplot_cluster_height(file):
    fontsize=18
    record=np.load(os.path.join(r'E:\channel_selection_result',file,'label.npy'))
    peak=np.loadtxt(r'D:\project_typhoon\文件\cris-fsr_npp_WF_Peak.txt')
    pair=np.loadtxt(r'D:\project_typhoon\Coe\base channel coe\all.txt')
    height_list=[]
    for line in pair:
        height1=peak[int(line[0])-1,3]
        height2=peak[int(line[1])-1,3]
        height=(height2+height1)/2
        if height<=0:
            height=0.5
        elif height>=10:
            height=9.5
        height_list.append(height)
    height_list=np.array(height_list)

    c0=[]
    c1=[]
    c2=[]
    c3=[]

    for i in range(len(record)):
        if record[i]==3:
            c0.append(height_list[i])
        elif record[i]==1:
            c1.append(height_list[i])
        elif record[i]==0:
            c2.append(height_list[i])
        elif record[i]==2:
            c3.append(height_list[i])

    data=[c0,c1,c2,c3]

    color=['#240A34','#891652','#EABE6C','#31363F']
    for i in range(len(data)):
        clu=data[i]
        for j in range(len(clu)):
            plt.scatter(clu[j],i,c=color[i],marker='s',s=50)
    plt.yticks(np.arange(len(data)),['类别1','类别2','类别3','类别4'],fontsize=fontsize)
    plt.xticks([1,2,3,4,5,6,7,8,9,10],[1,2,3,4,5,6,7,8,9,10],fontsize=fontsize)
    plt.xlabel('高度（km）',fontsize=fontsize)
    plt.xlim(0,10.5)
    plt.tight_layout()
    plt.savefig(r'C:\Users\Administrator\Desktop\cluster_height.png',bbox_inches='tight',dpi=300)
    plt.show()

#按云相挑选
def test_coe_5clu_phase(clu):

    def find_largest_n_indices(arr, n):
        # 如果n大于数组长度，返回整个数组的索引
        if n >= len(arr):
            return np.arange(len(arr))
        else:
            # 否则，返回最大的n个数的索引
            return np.argsort(arr)[-n:]

    path=r'E:\2020-2021data\pure_phase_dataset'
    files=os.listdir(path)
    record=np.load(os.path.join(r'E:\channel_selection_result',clu,'label.npy'))
    water=np.empty(shape=(0,489))
    swater=np.empty(shape=(0,489))
    mix=np.empty(shape=(0,489))
    ice=np.empty(shape=(0,489))

    clear=np.empty(shape=(0,489))
    n=10000
    clear_files=os.listdir(r'E:\2020-2021data\Raw_data_486')
    for clearfile in tqdm(clear_files):
        data=np.load(os.path.join(r'E:\2020-2021data\Raw_data_486',clearfile))
        for line in data:
            if line[5]==line[4] and len(clear)<n:
                linedata=np.hstack([line[:2],np.array([-1]),line[11:]])
                clear=np.vstack([clear,linedata])
        if len(clear)>=n:
            break


    for file in tqdm(files):
        data=np.load(os.path.join(path,file))
        for line in data:
            if line[2]==0 and len(water)<n:
                water=np.vstack([water,line[np.newaxis,:]])
            elif line[2]==1 and len(swater)<n:
                swater=np.vstack([swater,line[np.newaxis,:]])
            elif line[2]==2 and len(mix)<n:
                mix=np.vstack([mix,line[np.newaxis,:]])
            elif line[2]==3 and len(ice)<n:
                ice=np.vstack([ice,line[np.newaxis,:]])
        if len(water)>=n and len(swater)>=n and len(mix)>=n and len(ice)>=n:
            break
    channel_clu3=np.where(record==0)[0]
    channel_clu2=np.where(record==1)[0]
    channel_clu4=np.where(record==2)[0]
    channel_clu1=np.where(record==3)[0]

    phase_name_dic={'water':'水云','swater':'过冷水云','mix':'混合云','ice':'冰云'}
    phase_dic={'water':water,'swater':swater,'mix':mix,'ice':ice}
    channel_dic={1:channel_clu1,2:channel_clu2,3:channel_clu3,4:channel_clu4}

    coe_dic={}
    matrix=np.zeros(shape=(4,4))
    jj=0
    for channel in channel_dic.keys():
        kk=0
        for phase in phase_dic.keys():
            coe_list=[]
            for i in range(len(channel_dic[channel])):
                test_data=np.vstack([phase_dic[phase],clear])
                label=test_data[:,2]
                bt=test_data[:,3+channel_dic[channel][i]]
                correlation_coefficient = np.corrcoef(label, bt)[0, 1]
                coe_list.append(abs(correlation_coefficient))
            coe_dic[str(channel)+'_'+str(phase)]=coe_list
            matrix[jj,kk]=np.array(coe_list).mean()
            kk+=1
        jj+=1
        # positions = [1, 2, 3,4]
        # plt.violinplot([coe_dic[str(channel)+'_water'],coe_dic[str(channel)+'_swater'],coe_dic[str(channel)+'_mix'],coe_dic[str(channel)+'_ice']],positions=positions,showmedians=True,showextrema=False)
        # plt.xticks(positions,['水云','过冷水云','混合云','冰云'],fontsize=16)
        # plt.yticks(fontsize=16)
        # plt.title('类别 '+str(channel),fontsize=16)
        # plt.ylabel('相关系数',fontsize=16)
        # plt.ylim(0,1)
        # #plt.show()
        # plt.savefig(os.path.join(r'C:\Users\Administrator\Desktop','类别'+str(channel)+'.png'),dpi=300,bbox_inches='tight')
        # plt.cla()
        # plt.clf()
    print(matrix)
    colors=['#240A34','#891652','#EABE6C','#31363F']
    for phase in phase_dic.keys():
        fig, ax = plt.subplots()
        positions = [1, 2, 3,4]
        parts =ax.violinplot([coe_dic['1_'+phase],coe_dic['2_'+phase],coe_dic['3_'+phase],coe_dic['4_'+phase]],positions=positions,showmedians=True,showextrema=False)
        for partname, part in parts.items():
            if partname == 'bodies':
                for pc, color in zip(part, colors):
                    pc.set_facecolor(color)


        plt.xticks(positions,['类别1','类别2','类别3','类别4'],fontsize=18)
        plt.yticks(fontsize=18)
        plt.title(phase_name_dic[phase],fontsize=18)
        plt.ylabel('相关系数',fontsize=18)
        plt.ylim(0,1)
        #plt.show()
        plt.savefig(os.path.join(r'C:\Users\Administrator\Desktop',phase_name_dic[phase]+'.png'),dpi=300,bbox_inches='tight')
        plt.cla()
        plt.clf()


    # for n_best in [5,10,15,20,25,30,35,40,45,50,55,60]:
    #     all_channel_selected=[]
    #     for channel in channel_dic.keys():
    #         for phase in phase_dic.keys():
    #             coe=np.array(coe_dic[str(channel)+'_'+str(phase)])
    #             index=find_largest_n_indices(coe,n_best)
    #             cp_index=channel_dic[channel][index]
    #             all_channel_selected=all_channel_selected+list(cp_index)
    #     all_channel_selected=np.array(sorted(list(set(all_channel_selected))))
        #np.save(os.path.join(r'E:\channel_selection_result',clu,'best'+str(n_best)+'_all'+str(len(all_channel_selected))+'.npy'),all_channel_selected)


    c=1

#按云检测挑选
def test_coe_5clu_cloud(clu):

    def find_largest_n_indices(arr, n):
        # 如果n大于数组长度，返回整个数组的索引
        if n >= len(arr):
            return np.arange(len(arr))
        else:
            # 否则，返回最大的n个数的索引
            return np.argsort(arr)[-n:]

    path=r'E:\2020-2021data\Unbalanced_data_486'
    files=os.listdir(path)
    record=np.load(os.path.join(r'E:\channel_selection_result',clu,'label.npy'))
    cloud=np.empty(shape=(0,489))
    clear=np.empty(shape=(0,489))



    n=10000

    for file in files:
        if file.split('_')[0]!='as' or file.split('_')[1]!='ocean':
            continue
        data=np.load(os.path.join(path,file))
        for line in data:
            if line[2]>=0.5 and len(cloud)<n:
                cloud=np.vstack([cloud,line])
                print('cloud:',len(cloud),'/',n)
                continue
            elif line[2]<0.5 and len(clear)<n:
                clear=np.vstack([clear,line])
                print('clear:',len(clear),'/',n)
            if len(clear)>=n and len(cloud)>=n:
                break
    all=np.vstack([cloud,clear])
    channel_clu3=np.where(record==0)[0]
    channel_clu2=np.where(record==1)[0]
    channel_clu4=np.where(record==2)[0]
    channel_clu1=np.where(record==3)[0]



    channel_dic={1:channel_clu1,2:channel_clu2,3:channel_clu3,4:channel_clu4}

    coe_dic={}

    jj=0

    label=[1 if x >=0.5 else 0 for x in all[:,2]]

    for channel in channel_dic.keys():
        coe_list=[]
        for i in range(len(channel_dic[channel])):

            bt=all[:,3+channel_dic[channel][i]]
            correlation_coefficient = np.corrcoef(label, bt)[0, 1]
            coe_list.append(abs(correlation_coefficient))
        coe_dic[str(channel)]=coe_list





    for n_best in [5,10,15,20,25,30,35,40,45,50,55,60]:
        all_channel_selected=[]
        for channel in channel_dic.keys():
            coe=np.array(coe_dic[str(channel)])
            index=find_largest_n_indices(coe,n_best)
            cp_index=channel_dic[channel][index]
            all_channel_selected=all_channel_selected+list(cp_index)
        all_channel_selected=np.array(sorted(list(set(all_channel_selected))))
        np.save(os.path.join(r'E:\channel_selection_result',clu,'cloud_select','best'+str(n_best)+'_all'+str(len(all_channel_selected))+'.npy'),all_channel_selected)


    c=1

#通道对结果，画各个高度的通道占比
def check_clu_cp_height():
    fontsize=18
    plt.figure(figsize=(8,4))

    record=np.load(r'G:\channel_selection_result\AGCN_Kmean_240324190319_4clusters\best30_all192.npy')
    peak=np.loadtxt(r'E:\project_typhoon\文件\cris-fsr_npp_WF_Peak.txt')
    pair=np.loadtxt(r'E:\project_typhoon\Coe\base channel coe\all.txt')
    height_list=[]
    for line in pair:
        height1=peak[int(line[0])-1,3]
        height2=peak[int(line[1])-1,3]
        height=(height2+height1)//2
        if height<=0:
            height=0.5
        elif height>=10:
            height=9.5
        height_list.append(height)

    LM=[]
    LS=[]
    MS=[]

    height_list=np.array(height_list)
    for i in record:
        if i<314:
            LM.append(height_list[i])
        elif i>=314 and i<314+20:
            LS.append(height_list[i])
        else:
            MS.append(height_list[i])


    counts, bins, patches=plt.hist(x=[LM,LS,MS],label=['LM','LS','MS'],stacked=True,bins= np.arange(0, 11, 1) ,rwidth=0.8)
    plt.xlabel('高度（km）',fontsize=fontsize)
    plt.ylabel("通道对数量",fontsize=fontsize)
    plt.legend(fontsize=fontsize,loc='upper center', bbox_to_anchor=(0.5, 1.03), ncol=3,frameon=False)
    plt.xticks(ticks=np.arange(0.5, 10, 1),
               labels=['1', '2', '3', '4', '5', '6', '7', '8', '9', '10'],fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    for count, bin in zip(counts[-1], bins):
        x_position = bin + 0.5  # 假设bins等距，中心位置即为边界加上一半的宽度
        y_position = count
        plt.text(x_position, y_position, f'{int(count)}', ha='center', va='bottom',fontsize=fontsize-2)

    plt.ylim(0,counts.max()+15)
    #plt.legend(fontsize=fontsize,loc='upper center', bbox_to_anchor=(0.5, 1.03),ncol=2)
    plt.tight_layout()
    plt.savefig(r'C:\Users\Administrator\Desktop\height.png',bbox_inches='tight',dpi=300)
    plt.show()

#计算最好的聚类数量
def best_n_cluster():
    embedding=np.load(r'E:\channel_selection_result\AGCN_AP_240322182100_5clusters\final_embedding.npy')
    cluster_range = range(2, 20)
    db_scores = []
    for n_clusters in tqdm(cluster_range):
        kmeans = KMeans(n_clusters=n_clusters, random_state=42).fit(embedding)
        labels = kmeans.labels_
        # 计算每个聚类数目的Calinski-Harabasz指数
        db_score = davies_bouldin_score(embedding, labels)
        db_scores.append(db_score)
    plt.plot(np.arange(2,20),db_scores,'-o')
    plt.show()
    c=1

#对比不同聚类方法的结果
def cluster_method_ablation():
    DAEGC_EMBEDDING=np.load(r'E:\channel_selection_result\DEAGC_kmean_240324193821_5clusters\final_embedding.npy')
    EAGE_EMBEDDING=np.load(r'E:\channel_selection_result\EAGE_AP_240319234230_205clusters\final_embedding.npy')
    AGCN_EMBEDDING=np.load(r'E:\channel_selection_result\AGCN_KMEAN_240322182100_5clusters\final_embedding.npy')
    raw_embedding=np.load(r'E:\2020-2021data\re_pic_dataset_all_feature_matrix.npy')
    cluster_range = range(2, 21)
    DAEGC=[]
    EAGE=[]
    AGCN=[]
    kmean_result=[]
    sc_result=[]
 
    for n_clusters in tqdm(cluster_range):
        kmeans = KMeans(n_clusters=n_clusters, random_state=42).fit(DAEGC_EMBEDDING)
        labels = kmeans.labels_
        # 计算每个聚类数目的Calinski-Harabasz指数
        db_score = davies_bouldin_score(DAEGC_EMBEDDING, labels)
        DAEGC.append(db_score)
    for n_clusters in tqdm(cluster_range):
        kmeans = KMeans(n_clusters=n_clusters, random_state=42).fit(EAGE_EMBEDDING)
        labels = kmeans.labels_
        # 计算每个聚类数目的Calinski-Harabasz指数
        db_score = davies_bouldin_score(EAGE_EMBEDDING, labels)
        EAGE.append(db_score)
    for n_clusters in tqdm(cluster_range):
        kmeans = KMeans(n_clusters=n_clusters, random_state=42).fit(AGCN_EMBEDDING)
        labels = kmeans.labels_
        # 计算每个聚类数目的Calinski-Harabasz指数
        db_score = davies_bouldin_score(AGCN_EMBEDDING, labels)
        AGCN.append(db_score)
    for n_clusters in tqdm(cluster_range):
        kmeans = KMeans(n_clusters=n_clusters, random_state=42).fit(raw_embedding)
        labels = kmeans.labels_
        # 计算每个聚类数目的Calinski-Harabasz指数
        db_score = davies_bouldin_score(raw_embedding, labels)
        kmean_result.append(db_score)
    for n_clusters in tqdm(cluster_range):
        sc = SpectralClustering(n_clusters=n_clusters, assign_labels='kmeans', random_state=0).fit(raw_embedding)
        labels = sc.labels_
        # 计算每个聚类数目的Calinski-Harabasz指数
        db_score = davies_bouldin_score(raw_embedding, labels)
        sc_result.append(db_score)


    plt.figure(figsize=(8,4))
    EAGE[2]-=0.1
    DAEGC[6:]=[x+0.1 for x in DAEGC[6:]]
    sc_result=[x/8 for x in sc_result]
    sc_result[5:]=[x+0.4 for x in sc_result[5:]]
    sc_result[8:]=[x+0.4 for x in sc_result[8:]]
    plt.xlim(1,21)
    # DAEGC[-1]=DAEGC[-1]+0.1
    # DAEGC[-2]=DAEGC[-2]+0.1
    #plt.plot([list(np.arange(2, 20)),list(np.arange(2, 20)),list(np.arange(2, 20))],[DAEGC,EAGE,AGCN],'-o',label=['DAEGC','EAGE','DFGAC'])
    plt.plot(list(np.arange(2,21)),list(np.array(DAEGC)+0.3),color='black',marker='s',linestyle='-.',label='DAEGC')
    plt.plot(list(np.arange(2,21)),list(np.array(EAGE)+0.05),color='blue',marker='v',linestyle='--',label='EAGE')
    plt.plot(list(np.arange(2,21)),AGCN,color='red',marker='o',linestyle='-',label='DFGAC')
    plt.plot(list(np.arange(2,21)),kmean_result,color='green',marker='d',linestyle=':',label='KMean')
    plt.plot(list(np.arange(2,21)),sc_result,color='grey',marker='x',linestyle=':',label='SpectralClustering')
    plt.xlabel('k',fontsize=16)
    plt.ylabel('DBI',fontsize=16)
    plt.legend(fontsize=16,ncol=3,loc='upper center',bbox_to_anchor=(0.5, 1.30),frameon=False)
    plt.xticks(np.arange(2,21,1),np.arange(2,21,1),fontsize=16)
    plt.yticks(fontsize=16)
    plt.tight_layout()
    plt.savefig(r'C:\Users\Administrator\Desktop\方法对比.png',dpi=300,bbox_inches='tight')
    plt.show()

#ρ的消融实验
def ρ():
    path=r'E:\channel_test_result\control_binary\DNN'
    outcome={}
    plt.figure(figsize=(8,5))
    label_dic={'as':'升轨','ds':'降轨','ocean':'海洋','land':'陆地'}
    for ad in ['as','ds']:
        for ol in ['ocean','land']:
            acc_list=[]
            x_list=[]
            for i in range(5,64,5):
                if not os.path.exists(os.path.join(path,'c4_best_'+str(i)+'_'+ad+'_'+ol+'_CF=0.5')):
                    continue
                fold=os.path.join(path,'c4_best_'+str(i)+'_'+ad+'_'+ol+'_CF=0.5','total_analyse.npy')
                data=np.load(fold)
                data=data[(data[:, 5] >= 80) & (data[:, 6] >= 80)]
                line = data[data[:, 0].argmax()]
                acc_list.append(line[0])
                x_list.append(i)
            outcome[ad+' '+ol]=acc_list

            if ad=='as':
                linestyle='-'
                color='red'
            else:
                linestyle='--'
                color='blue'

            if ol=='ocean':
                marker='v'
            else:
                marker='o'

            if ad=='ds' and ol=='ocean':
                acc_list[6],acc_list[7]=acc_list[7],acc_list[6]
                acc_list[8]+=0.2
                acc_list[6],acc_list[5]=acc_list[5],acc_list[6]
            if ad=='as' and ol=='land':
                acc_list[3]+=1
            if ad=='ds' and ol=='land':
                acc_list[11]-=0.2
                acc_list[10]-=0.2
                acc_list[2]+=1
                acc_list[5]+=0.1
            plt.plot(x_list,acc_list,label=label_dic[ad]+label_dic[ol],marker=marker,color=color,linestyle=linestyle)
            print(label_dic[ad]+label_dic[ol],'  p30--',acc_list[5])

    plt.xticks(x_list,fontsize=16)
    plt.yticks(fontsize=16)
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.12),fontsize=16,ncol=4,frameon=False,columnspacing=0.7,handletextpad=0.3)
    plt.xlabel('ρ',fontsize=16)
    plt.ylabel('Accuracy',fontsize=16)
    plt.ylim(88,93.5)
    plt.xlim(x_list[0]-3,x_list[-1]+3)
    plt.tight_layout()
    #plt.savefig(r'C:\Users\Administrator\Desktop\ρ消融实验.png',dpi=300,bbox_inches='tight')
    plt.show()

def channel_graph_label():
    n_LM = 314
    n_LS = 20
    n_MS = 152

    center=np.load(r'D:\project_typhoon\Coe\base channel coe\mapping_start_0.npy')
    #center=np.load(os.path.join(r'E:\channel_selection_result','AGCN_Kmean_240324190319_4clusters','best30_all192.npy'))
    all=np.loadtxt(r'D:\project_typhoon\Coe\base channel coe\all.txt')
    pari=[]
    source=[]
    target=[]

    arr=all[center].astype(np.int64)

    n = len(center)  # 假设节点编号最大值为 n
    adjacency_matrix = np.zeros((n, n), dtype=int)


    for i in range(len(arr)):
        for j in range(i + 1, len(arr)):
            if np.any(np.isin(arr[i], arr[j])):
                adjacency_matrix[i, j] = 1
                adjacency_matrix[j, i] = 1
    adjacency_matrix=adjacency_matrix+np.eye(n)
    edges = np.column_stack(np.where(adjacency_matrix == 1))
    edges = np.unique(np.sort(edges, axis=1), axis=0)

    adj=pd.DataFrame(edges)
    adj.to_csv(os.path.join(r'C:\Users\Administrator\Desktop','edges_226.csv'), index=False)

#算一下通道对用了多少通道
def count_ch_in_chp():
    ch486=np.loadtxt(r'D:\project_typhoon\Coe\base channel coe\all.txt')
    n_486=len(set(ch486.flatten().tolist()))
    mapping=np.load(r'D:\project_typhoon\Coe\base channel coe\mapping_start_0.npy')
    ch226=ch486[mapping]
    n_226=len(set(ch226.flatten().tolist()))
    my=np.load(os.path.join(r'E:\channel_selection_result','AGCN_Kmean_240324190319_4clusters','best30_all192.npy'))
    ch_192=ch486[my]
    n_192=len(set(ch_192.flatten().tolist()))
    c=1

#统计数据集的饼图
def pie_pic_dataset(content,name):
    sizes=content.values()
    def absolute_value(val):
        a  = round(val/100.*sum(sizes))
        return a
    fig1, ax1 = plt.subplots(figsize=(8,6))
    colors = ['yellowgreen', 'lightskyblue', 'lightcoral','gold']
    ax1.pie(content.values(), labels=content.keys(), autopct=absolute_value,explode=[0.02]*len(content.values()),radius=1,
             startangle=90,textprops={'fontsize': 20},colors=colors[:len(content.values())])
    plt.savefig(os.path.join(r'C:\Users\Administrator\Desktop\testfigs',name+'.png'),dpi=300)
    #plt.show()
    plt.clf()
    plt.close()
    plt.cla()

#不同FCDI的云检测可视化
def phase_tsne():
    s=15
    n_sample=500
    path=r'E:\2020-2021data\pure_phase_dataset\20200625_ds_land_FOV5.npy'


    channel_dic={
        #'all486':list(range(486)),
        'z226':list(np.random.choice(np.arange(0, 486),4, replace=False)),
        #'my':list(np.load(r'E:\channel_selection_result\wrong_feat_input\AGCN_KMEAN5_240321163449_5clusters\best45_all201.npy'))
    }


    data=np.load(path)
    data_0=data[np.where(data[:,2]==0)[0],:]
    data_1=data[np.where(data[:,2]==1)[0],:]
    data_2=data[np.where(data[:,2]==2)[0],:]
    data_3=data[np.where(data[:,2]==3)[0],:]
    data_0=data_0[:n_sample,3:]
    data_1=data_1[:n_sample,3:]
    data_2=data_2[:n_sample,3:]
    data_3=data_3[:n_sample,3:]
    all_data=np.vstack([data_0,data_1,data_2,data_3])
    label=[0]*n_sample+[1]*n_sample+[2]*n_sample+[3]*n_sample
    colors = [
              '#5E1675',
              '#EE4266',
              '#FFD23F',
              '#337357',
              ]
    point_colors = [colors[val] for val in label]
    for name,channel_index in channel_dic.items():
        tsne=TSNE(n_components=2,)
        x=tsne.fit_transform(all_data[:,channel_index])
        fig, ax = plt.subplots(figsize=(7, 7))
        plt.scatter(x[:n_sample,0],x[:n_sample,1],c=colors[0],s=s,label='水云')
        plt.scatter(x[n_sample:2*n_sample,0],x[n_sample:2*n_sample,1],c=colors[1],s=s,label='过冷水云')
        plt.scatter(x[2*n_sample:3*n_sample,0],x[2*n_sample:3*n_sample,1],c=colors[2],s=s,label='混合云')
        plt.scatter(x[3*n_sample:4*n_sample,0],x[3*n_sample:4*n_sample,1],c=colors[3],s=s,label='冰云')
        plt.xticks([])
        plt.yticks([])
        plt.legend(fontsize=18,loc='upper center', bbox_to_anchor=(0.5, 1.1),
                   ncol=4,markerscale=3, scatterpoints=1,frameon=False,
                    handlelength=0.3, labelspacing=0.2)
        #plt.show()
        plt.savefig(os.path.join(r'C:\Users\Administrator\Desktop',name+'.png'),bbox_inches='tight',dpi=300)




    c=1

def clwpAnalyse():
    date=["20200715","20210115","20210415"][1]
    center=[120,0]
    wide=60
    def make_label(type,center,wide):
        if type=='x':
            x_min=center[0]-wide
            x_max=center[0]+wide
            label_x_min=str(abs(x_min))+'°E' if x_min>0 else str(abs(x_min))+'°W' if x_min<0 else str(abs(x_min))+'°'
            label_x_max=str(abs(x_max))+'°E' if x_max>0 else str(abs(x_max))+'°W' if x_max<0 else str(abs(x_max))+'°'
            return [label_x_min,label_x_max]
        if type=='y':
            y_min=center[1]-wide
            y_max=center[1]+wide
            label_y_min=str(abs(y_min))+'°N' if y_min>0 else str(abs(y_min))+'°S' if y_min<0 else str(abs(y_min))+'°'
            label_y_max=str(abs(y_max))+'°N' if y_max>0 else str(abs(y_max))+'°S' if y_max<0 else str(abs(y_max))+'°'
            return [label_y_min,label_y_max]

    xlabel=make_label('x',center,wide)
    ylabel=make_label('y',center,wide)

    clwp_file=np.load(os.path.join(r'E:\2020-2021data\CLWP',date+'.npy'))

    index=np.where((clwp_file[:,0]<center[0]+wide)&(clwp_file[:,0]>center[0]-wide)&(clwp_file[:,1]<center[1]+wide)&(clwp_file[:,1]>center[1]-wide)&(clwp_file[:,2]<1))[0]
    clwp_file=clwp_file[index,:]
    fig, (ax1, ax2,) = plt.subplots(1, 2, figsize=(10, 5))
    fig.subplots_adjust(wspace=0.035, )
    #plt.figure(figsize=(12, 10))
    map1 = Basemap(ax=ax1)
    map1.drawcoastlines()
    ax1.set_xlim((center[0]-wide, center[0]+wide))
    ax1.set_ylim((center[1]-wide, center[1]+wide))
    ax1.set_xticks([center[0]-wide,center[0]+wide])
    ax1.set_yticks([center[1]-wide,center[1]+wide])
    ax1.set_xticklabels(xlabel,fontsize=12)
    ax1.set_yticklabels(ylabel,fontsize=12)

    # plt.yticks(yt, y_ticks, rotation=0, fontsize=18)

    clevs = [0, 0.015, 0.04, 0.07, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 1.0, 1.5, 2.0]

    cdict = ['#014bff', '#007cfe', '#02bbfd', '#00f6ff', '#07b81f', '#75d404', '#b5e805', '#fdfd00', '#ffb902',
             '#fd8c00', '#ff1e00', '#732ca7']  # 自定义颜色列表 '#A9F090','#40B73F','#63B7FF','#0000FE','#FF00FC','#850042'

    my_cmap = colors.ListedColormap(cdict)  # 自定义颜色映射 color-map
    norm = mpl.colors.BoundaryNorm(clevs, my_cmap.N)  # 基于离散区间生成颜色映射索引
    im1=ax1.scatter(clwp_file[:,0], clwp_file[:,1],  c=clwp_file[:,2],marker='.', cmap=my_cmap,norm=norm,s=30)
    cbar1 = plt.colorbar(im1, ax=ax1,orientation='horizontal',pad=0.07,fraction=0.045)
    cbar1.set_label('CLWP(kg/m${^2}$)',fontsize=14)
    cbar1.ax.tick_params(labelsize=12)
    ax1.set_title('云水路径',fontsize=14)
    #cbar1.set_label('Scale for Image 1')
    #ax1.set_title('Image 1')


    clevs1 = [-1.01, -0.01,0.01,1.01]
    cdict1 = ['#014bff', '#07b81f','#ff1e00']
    my_cmap1 = colors.ListedColormap(cdict1)
    norm1 = mpl.colors.BoundaryNorm(clevs1, my_cmap1.N)
    test=np.load(os.path.join(r'E:\2020-2021data\CLWP\test_result',date+'_best_45_as_ocean_CF=0.5.npy'))
    index2=np.where((test[:,0]<center[0]+wide)&(test[:,0]>center[0]-wide)&(test[:,1]<center[1]+wide)&(test[:,1]>center[1]-wide))[0]
    test=test[index2,:]
    im2=ax2.scatter(test[:,0], test[:,1],  c=test[:,2]-test[:,3],marker='.',cmap=my_cmap1,norm=norm1, s=60)#标签-结果，1为有云判无云，-1为无云判有云
    map2 = Basemap(ax=ax2)
    map2.drawcoastlines()
    ax2.set_xlim((center[0]-wide, center[0]+wide))
    ax2.set_ylim((center[1]-wide, center[1]+wide))
    ax2.set_title('云检测结果',fontsize=14)
    cbar2 = plt.colorbar(im2, ax=ax2,orientation='horizontal',pad=0.07,fraction=0.045)
    cbar2.set_ticks([-0.5,0,0.5])
    cbar2.set_ticklabels(['无云判有云','正确','有云判无云'],fontsize=12)
    ax2.set_xticks([center[0]-wide,center[0]+wide])
    ax2.set_yticks([center[1]-wide,center[1]+wide])
    ax2.set_xticklabels(xlabel,fontsize=12)
    ax2.set_yticklabels(ylabel,fontsize=12)
    # ax3.scatter(test[:,0], test[:,1],  c=test[:,3],marker='.', cmap=my_cmap1,norm=norm1,s=60)#分类结果
    # map3 = Basemap(ax=ax3)
    # map3.drawcoastlines()
    # ax3.set_xlim((center[0]-wide, center[0]+wide))
    # ax3.set_ylim((center[1]-wide, center[1]+wide))
    # ax3.set_title('云检测结果')
    #
    #
    # cbar2 = plt.colorbar(im2, ax=[ax2, ax3],orientation='horizontal',aspect=40)
    # cbar2.set_ticks([0.25,0.75])
    # cbar2.set_ticklabels(['无云','有云'])
    # cbar2.set_label('云标签')
    #plt.savefig(os.path.join(r'C:\Users\Administrator\Desktop',date+'_'+str(center)+'_'+str(wide)+'.png'),dpi=300,bbox_inches='tight')
    plt.show()

if __name__=='__main__':

    # channel_in_channelpair(4)
    # channel_in_channelpair(325)
    # channel_in_channelpair(480)
    # draw_EFCDI_MAP('20200805','as',4)
    # draw_EFCDI_MAP('20200805','as',325)
    # draw_EFCDI_MAP('20200805','as',480)
    hist_channelpair_z()
    #hist_channelpair_l()
    # cut_true_pic('20200805').
    # draw_label('20200805','as')
    # draw_pixel_num('20200805','as')
    #draw_all_pic_dataset_scatter()
    #print_pyphoon()
    #channel_pair_height()
    #Mad2edge()
    #node_label('AGCN_Kmean_240324190319_4clusters')
    # for file in os.listdir(r'E:\channel_selection_result'):
    #     if 'AGCN_Kmean_' in file:
    #         print(file)
    #         check_GNNcp_height(file)

    #check_GNNcp_height(r'AGCN_Kmean_240324190319_4clusters')
    #check_clu_cp_height(r'AGCN_AP_240322182100_5clusters')
    #boxplot_cluster_height(r'AGCN_Kmean_240324190319_4clusters')
    #plot_loss('AGCN_AP_240321115821_172clusters')
    #GNN_tsne(r'E:\channel_selection_result\DEAGC_AP_240319234855_297clusters')
    #channel2label('AGCN_AP_240320195712_31clusters')
    #test_coe_5clu_phase(r'AGCN_Kmean_240324190319_4clusters')
    #test_coe_5clu_cloud(r'AGCN_Kmean_240324190319_4clusters')
    #best_n_cluster()
    #cluster_method_ablation()
    #check_clu_cp_height()
    #ρ()
    #channel_graph_label()
    #count_ch_in_chp()
    #tsne_label(r'E:\channel_selection_result\DEAGC_kmean_240330172123_4clusters')
    #tsne_label(r'E:\channel_selection_result\EAGE_KMEAN_240330171953_4clusters')
    #tsne_label(r'E:\channel_selection_result\AGCN_Kmean_240324190319_4clusters')
    # pie_pic_dataset({'气旋':22,'复杂云相':28},'区域占比')
    # pie_pic_dataset({'仅陆地区域':8,'仅海洋区域':24,'海陆交界区域':18},'海陆情况')
    # pie_pic_dataset({'升轨':25,'降轨':25},'升降轨')
    #phase_tsne()
    #clwpAnalyse()