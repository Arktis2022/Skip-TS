import torch
from dataset import get_data_transforms
from torchvision.datasets import ImageFolder
import numpy as np
from torch.utils.data import DataLoader
from resnet import resnet18, resnet34, resnet50, wide_resnet50_2
from de_resnet import de_resnet18, de_resnet50, de_wide_resnet50_2
from dataset import MVTecDataset
from torch.nn import functional as F
from sklearn.metrics import roc_auc_score
import cv2
import matplotlib.pyplot as plt
from sklearn.metrics import auc
from skimage import measure
import pandas as pd
from numpy import ndarray
from statistics import mean
from scipy.ndimage import gaussian_filter
from sklearn import manifold
from matplotlib.ticker import NullFormatter
from scipy.spatial.distance import pdist
import matplotlib
import pickle
import os
from skimage.segmentation import mark_boundaries
plt.switch_backend('agg')
# 计算异常分数图
def cal_anomaly_map(fs_list, ft_list, out_size=224, amap_mode='mul'):
    if amap_mode == 'mul':
        anomaly_map = np.ones([out_size, out_size]) # 初始化异常图
    else:
        anomaly_map = np.zeros([out_size, out_size])
    a_map_list = []
    for i in range(len(ft_list)): # 输入的异常图里面有三个输出
        fs = fs_list[i]
        ft = ft_list[i]
        a_map = 1 - F.cosine_similarity(fs, ft) # 先计算其中一个输出的余弦差值
        a_map = torch.unsqueeze(a_map, dim=1) # 异常图升维
        a_map = F.interpolate(a_map, size=out_size, mode='bilinear', align_corners=True) # 异常图进行差值上采样，因为不同层的输出大小不一样，统一上采样到256x256
        a_map = a_map[0, 0, :, :].to('cpu').detach().numpy()
        a_map_list.append(a_map)
        if amap_mode == 'mul':
            anomaly_map *= a_map 
        else:
            anomaly_map += a_map
    return anomaly_map, a_map_list # 这里返回的异常图和异常图list两个，但是list其实没用上，之后会中途输出

def show_cam_on_image(img, anomaly_map):
    cam = np.float32(anomaly_map)/255 + np.float32(img)/255
    cam = cam / np.max(cam)
    return np.uint8(255 * cam)

def min_max_norm(image):
    a_min, a_max = image.min(), image.max()
    return (image-a_min)/(a_max - a_min) # 数据标准化函数

def cvt2heatmap(gray):
    heatmap = cv2.applyColorMap(np.uint8(gray), cv2.COLORMAP_JET)
    return heatmap

# 计算pro的函数，直接抄现成的    
def compute_pro(masks: ndarray, amaps: ndarray, num_th: int = 200) -> None:

    """Compute the area under the curve of per-region overlaping (PRO) and 0 to 0.3 FPR
    Args:
        category (str): Category of product
        masks (ndarray): All binary masks in test. masks.shape -> (num_test_data, h, w)
        amaps (ndarray): All anomaly maps in test. amaps.shape -> (num_test_data, h, w)
        num_th (int, optional): Number of thresholds
    """

    assert isinstance(amaps, ndarray), "type(amaps) must be ndarray"
    assert isinstance(masks, ndarray), "type(masks) must be ndarray"
    assert amaps.ndim == 3, "amaps.ndim must be 3 (num_test_data, h, w)"
    assert masks.ndim == 3, "masks.ndim must be 3 (num_test_data, h, w)"
    
    #print(amaps.shape)
    #print(masks.shape)
    assert amaps.shape == masks.shape, "amaps.shape and masks.shape must be same"
    assert set(masks.flatten()) == {0, 1}, "set(masks.flatten()) must be {0, 1}"
    assert isinstance(num_th, int), "type(num_th) must be int"

    df = pd.DataFrame([], columns=["pro", "fpr", "threshold"])
    binary_amaps = np.zeros_like(amaps, dtype=np.bool)

    min_th = amaps.min()
    max_th = amaps.max()
    delta = (max_th - min_th) / num_th

    for th in np.arange(min_th, max_th, delta):
        binary_amaps[amaps <= th] = 0
        binary_amaps[amaps > th] = 1

        pros = []
        for binary_amap, mask in zip(binary_amaps, masks):
            for region in measure.regionprops(measure.label(mask)):
                axes0_ids = region.coords[:, 0]
                axes1_ids = region.coords[:, 1]
                tp_pixels = binary_amap[axes0_ids, axes1_ids].sum()
                pros.append(tp_pixels / region.area)

        inverse_masks = 1 - masks
        fp_pixels = np.logical_and(inverse_masks, binary_amaps).sum()
        fpr = fp_pixels / inverse_masks.sum()

        df = df.append({"pro": mean(pros), "fpr": fpr, "threshold": th}, ignore_index=True)

    # Normalize FPR from 0 ~ 1 to 0 ~ 0.3
    df = df[df["fpr"] < 0.3]
    df["fpr"] = df["fpr"] / df["fpr"].max()

    pro_auc = auc(df["fpr"], df["pro"])
    return pro_auc
    
# 如果不需要分割
def evaluation_me(encoder,decoder, res,dataloader,device,print_canshu,score_num):
    decoder.eval() 
    encoder.eval()
    gt_list_sp = [] # label的list
    pr_list_sp = [] # 预测值的list

    with torch.no_grad():
        for img, label, _ in dataloader:
            img = img.to(device) 
            inputs = encoder(img)
            outputs = decoder(inputs[3],inputs[0:3],res)
            anomaly_map, _ = cal_anomaly_map(inputs[0:3], outputs, img.shape[-1], amap_mode='a') # 需要的是最终的这个特征图，异常图可以用加法/乘法，显然加法更合理
            
            #anomaly_map, _ = cal_anomaly_map(inputs[0:2], outputs[0:2], img.shape[-1], amap_mode='a') # 需要的是最终的这个特征图，异常图可以用加法/乘法，显然加法更合理 1+2
            #anomaly_map, _ = cal_anomaly_map(inputs[1:3], outputs[1:3], img.shape[-1], amap_mode='a') # 需要的是最终的这个特征图，异常图可以用加法/乘法，显然加法更合理 2+3
            #anomaly_map, _ = cal_anomaly_map([inputs[0],inputs[2]], [outputs[0],outputs[2]], img.shape[-1], amap_mode='a') # 需要的是最终的这个特征图，异常图可以用加法/乘法，显然加法更合理 1+3
            #print(input[0].shape)
            #print(input[1].shape)
            #print(input[2].shape)
            
            
            
            anomaly_map = gaussian_filter(anomaly_map, sigma=4) # 高斯平滑化
            gt_list_sp.append(label.numpy()[0]) # label
            
            pre_map = np.flipud(np.sort(anomaly_map.flatten()))
            pre = 0
            for x in range(score_num):
                pre +=pre_map[x]
            pre = pre/score_num
            pr_list_sp.append(round(pre,3)) # 异常图中的前num个最大值作为最终结果
            
        if print_canshu==1:
          print(gt_list_sp,pr_list_sp)      # 是否打印中间参数
        auroc_sp = round(roc_auc_score(gt_list_sp, pr_list_sp), 3) # round() 方法返回浮点数x的四舍五入值
        #print(gt_list_sp,pr_list_sp)
    return auroc_sp

# 生成热度图
def evaluation_visualization(encoder, decoder, res, dataloader,device, print_canshu,score_num,img_path):
    count = 0
    decoder.eval()
    with torch.no_grad():
        for img, gt, label, _ , ip in dataloader:
            print(ip[0][-20:-4])
            if (label.item() == 0):
                continue
            img = img.to(device)
            inputs = encoder(img)
            outputs = decoder(inputs[3],inputs[0:3],res)  

            anomaly_map, amap_list = cal_anomaly_map([inputs[0:3][-1]], [outputs[-1]], img.shape[-1], amap_mode='a') # 得到异常图
            anomaly_map = gaussian_filter(anomaly_map, sigma=4)    # 高斯滤波
            ano_map = min_max_norm(anomaly_map)                    # 数据的标准化（normalization）是将数据按比例缩放，使之落入一个小的特定区间
            
            
            ano_map = cvt2heatmap(255-ano_map*255)                     # 转化为热度图
            img = cv2.cvtColor(img.permute(0, 2, 3, 1).cpu().numpy()[0] * 255, cv2.COLOR_BGR2RGB)
            
            img = np.uint8(min_max_norm(img)*255)
            ano_map = show_cam_on_image(img, ano_map)              # 热度图和原图片叠加
            
            # 先画热度图
            plt.subplot(1,3,1)
            plt.imshow(ano_map)
            plt.axis('off')

            # 再画ground_truth
            gt = gt.cpu().numpy().astype(int)[0][0]*255
            plt.subplot(1,3,2)
            plt.imshow(gt,cmap='gray')
            plt.axis('off')
                    
            # 最后画原图
            plt.subplot(1,3,3)
            plt.imshow(img)
            plt.axis('off')
            
          
            
            if (os.path.exists(img_path)==0):
                os.mkdir(img_path)
                
            # 保存图片
            plt.savefig(img_path+str(ip[0][-20:-4]).replace('/','_')+'.png')
            
            count += 1
            
# 生成不需要mask的热度图
def evaluation_visualization_no_seg(encoder, decoder, res, dataloader,device, print_canshu,score_num,img_path):
    count = 0
    decoder.eval()
    with torch.no_grad():
        for img, label, _  in dataloader:
            #print(ip[0][-20:-4])
            if (label.item() == 0):
                continue
            img = img.to(device)
            inputs = encoder(img)
            outputs = decoder(inputs[3],inputs[0:3],res)  

            anomaly_map, amap_list = cal_anomaly_map([inputs[0:3][-1]], [outputs[-1]], img.shape[-1], amap_mode='a') # 得到异常图
            anomaly_map = gaussian_filter(anomaly_map, sigma=4)    # 高斯滤波
            ano_map = min_max_norm(anomaly_map)                    # 数据的标准化（normalization）是将数据按比例缩放，使之落入一个小的特定区间
            
            
            ano_map = cvt2heatmap(255-ano_map*255)                     # 转化为热度图
            img = cv2.cvtColor(img.permute(0, 2, 3, 1).cpu().numpy()[0] * 255, cv2.COLOR_BGR2RGB)
            
            img = np.uint8(min_max_norm(img)*255)
            ano_map = show_cam_on_image(img, ano_map)              # 热度图和原图片叠加
            
            # 先画热度图
            plt.subplot(1,2,1)
            plt.imshow(ano_map)
            plt.axis('off')

            # 再画ground_truth
            #gt = gt.cpu().numpy().astype(int)[0][0]*255
            #plt.subplot(1,3,2)
            #plt.imshow(gt,cmap='gray')
            #plt.axis('off')
                    
            # 最后画原图
            plt.subplot(1,2,2)
            plt.imshow(img)
            plt.axis('off')
            
          
            
            if (os.path.exists(img_path)==0):
                os.mkdir(img_path)
                
            # 保存图片
            plt.savefig(img_path+str(count).replace('/','_')+'.png')
            
            count += 1
            
# 计算分割相关参数，时间非常久
def evaluation(encoder,decoder, res,dataloader,device,img_path):
    decoder.eval()
    gt_list_px = []
    pr_list_px = []
    gt_list_sp = []
    pr_list_sp = []
    aupro_list = []
    with torch.no_grad():
        for img, gt, label, _,_ in dataloader:

            img = img.to(device)
            inputs = encoder(img)
            outputs = decoder(inputs[3],inputs[0:3],res) 
            
            # 根据encoder的前三个输出和decoder的三个输出计算异常图谱
            anomaly_map, _ = cal_anomaly_map(inputs[0:3], outputs, img.shape[-1], amap_mode='a')
            # 高斯滤波
            anomaly_map = gaussian_filter(anomaly_map, sigma=4)
            gt[gt > 0.5] = 1
            gt[gt <= 0.5] = 0
            if label.item()!=0:
                aupro_list.append(compute_pro(gt.squeeze(0).cpu().numpy().astype(int),
                                              anomaly_map[np.newaxis,:,:]))
            
            # numpy中的ravel()、flatten()、squeeze()都有将多维数组转换为一维数组的功能                                  
            gt_list_px.extend(gt.cpu().numpy().astype(int).ravel())
            pr_list_px.extend(anomaly_map.ravel())
            
            gt_list_sp.append(np.max(gt.cpu().numpy().astype(int)))
            pr_list_sp.append(np.max(anomaly_map))
        auroc_px = round(roc_auc_score(gt_list_px, pr_list_px), 3)
        auroc_sp = round(roc_auc_score(gt_list_sp, pr_list_sp), 3)
    return auroc_px, auroc_sp, round(np.mean(aupro_list),3)