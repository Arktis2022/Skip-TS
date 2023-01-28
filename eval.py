#-*-coding:gb2312-*-
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
from collections import OrderedDict
import matplotlib.pyplot as plt
plt.rc('font',family='Times New Roman') 
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
from test import cal_anomaly_map
import argparse
from test import min_max_norm,cvt2heatmap,show_cam_on_image,cal_anomaly_map
import sys
import seaborn as sns
from skimage.segmentation import mark_boundaries
plt.switch_backend('agg')

# 如果不需要分割
def evaluation_me(encoder,decoder, res,dataloader,device,print_canshu,score_num):
    decoder.eval() 
    
    gt_list_sp = [] # label的list
    pr_list_sp = [] # 预测值的list
    
    input_ = []
    output_ = []
    
    with torch.no_grad():
        for img, label, _ in dataloader:
            img = img.to(device) 
            inputs = encoder(img)
            outputs = decoder(inputs[3],inputs[0:3],res)
            
            #print(inputs[0].shape)
            input_.append(inputs[0].cpu().numpy())
            
            output_.append(outputs[0].cpu().numpy())
            
            #print(outputs[0].shape)
            
            
            anomaly_map, a_map_list = cal_anomaly_map(inputs[0:3], outputs, img.shape[-1], amap_mode='a') # 需要的是最终的这个特征图，异常图可以用加法/乘法，显然加法更合理
            #print(inputs[0].shape)
            #print(inputs[1].shape)
            #print(inputs[2].shape)
            
            a_map_list[0] = gaussian_filter(a_map_list[0], sigma=4)
            a_map_list[1] = gaussian_filter(a_map_list[1], sigma=4)
            a_map_list[2] = gaussian_filter(a_map_list[1], sigma=4)
            
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
    print(auroc_sp)
    print(len(output_))
    print(len(input_))
    return auroc_sp

# 生成热度图
def evaluation_visualization_no_seg_redu(encoder, decoder, res, dataloader,device, print_canshu,score_num,img_path):
    count = 0
    decoder.eval()
    with torch.no_grad():
        for img, label, _  in dataloader:
            if (label.item() == 0):
                continue
            img = img.to(device)
            inputs = encoder(img)
            outputs = decoder(inputs[3],inputs[0:3],res)  

            anomaly_map, a_map_list = cal_anomaly_map(inputs[0:3], outputs, img.shape[-1], amap_mode='a') # 需要的是最终的这个特征图，异常图可以用加法/乘法，显然加法更合理
            
            a_map_list[0] = gaussian_filter(a_map_list[0], sigma=4)
            a_map_list[1] = gaussian_filter(a_map_list[1], sigma=4)
            a_map_list[2] = gaussian_filter(a_map_list[1], sigma=4)
            
            anomaly_map = gaussian_filter(anomaly_map, sigma=4)    # 高斯滤波
            
            #anomaly_map = a_map_list[0]
            #anomaly_map = a_map_list[1]
            #anomaly_map = a_map_list[2]
            
            ano_map = min_max_norm(anomaly_map)                    # 数据的标准化（normalization）是将数据按比例缩放，使之落入一个小的特定区间
            
            
            ano_map = cvt2heatmap(255-ano_map*255)                     # 转化为热度图
            img = cv2.cvtColor(img.permute(0, 2, 3, 1).cpu().numpy()[0] * 255, cv2.COLOR_BGR2RGB)
            
            img = np.uint8(min_max_norm(img)*255)
            ano_map = show_cam_on_image(img, ano_map)              # 热度图和原图片叠加
            
            # 先画热度图
            #anomaly_map = a_map_list[0]
            #anomaly_map = a_map_list[1]
            #anomaly_map = a_map_list[2]
            #anomaly_map, a_map_list = cal_anomaly_map(inputs[0:3], outputs, img.shape[-1], amap_mode='a') # 需要的是最终的这个特征图，异常图可以用加法/乘法，显然加法更合理
            ano_map = min_max_norm(anomaly_map)                    # 数据的标准化（normalization）是将数据按比例缩放，使之落入一个小的特定区间
            
            
            ano_map = cvt2heatmap(255-ano_map*255)                     # 转化为热度图
            #img = cv2.cvtColor(img.permute(0, 2, 3, 1).cpu().numpy()[0] * 255, cv2.COLOR_BGR2RGB)
            
            #img = np.uint8(min_max_norm(img)*255)
            ano_map = show_cam_on_image(img, ano_map)              # 热度图和原图片叠加
            
            plt.subplot(1,3,1)
            plt.imshow(ano_map)
            plt.axis('off')

            # 再画ground_truth
            #gt = gt.cpu().numpy().astype(int)[0][0]*255
            anomaly_map = a_map_list[1]
            #anomaly_map = a_map_list[1]
            #anomaly_map = a_map_list[2]
            
            ano_map = min_max_norm(anomaly_map)                    # 数据的标准化（normalization）是将数据按比例缩放，使之落入一个小的特定区间
            
            
            ano_map = cvt2heatmap(255-ano_map*255)                     # 转化为热度图
            #img = cv2.cvtColor(img.permute(0, 2, 3, 1).cpu().numpy()[0] * 255, cv2.COLOR_BGR2RGB)
            
            #img = np.uint8(min_max_norm(img)*255)
            ano_map = show_cam_on_image(img, ano_map)              # 热度图和原图片叠加
            
            plt.subplot(1,3,2)
            plt.imshow(ano_map)
            plt.axis('off')
                    
            # 最后画原图
            anomaly_map = a_map_list[2]
            #anomaly_map = a_map_list[1]
            #anomaly_map = a_map_list[2]
            
            ano_map = min_max_norm(anomaly_map)                    # 数据的标准化（normalization）是将数据按比例缩放，使之落入一个小的特定区间
            
            
            ano_map = cvt2heatmap(255-ano_map*255)                     # 转化为热度图
            #img = cv2.cvtColor(img.permute(0, 2, 3, 1).cpu().numpy()[0] * 255, cv2.COLOR_BGR2RGB)
            
            #img = np.uint8(min_max_norm(img)*255)
            ano_map = show_cam_on_image(img, ano_map)              # 热度图和原图片叠加
            
            plt.subplot(1,3,3)
            plt.imshow(ano_map)
            plt.axis('off')
            
          
            
            if (os.path.exists(img_path)==0):
                os.mkdir(img_path)
                
            # 保存图片
            print(img_path+str(count).replace('/','_')+'.png')
            plt.savefig(img_path+str(count).replace('/','_')+'.png')
            
            count += 1

# 生成mask的热度图
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

            anomaly_map, a_map_list = cal_anomaly_map(inputs[0:3], outputs, img.shape[-1], amap_mode='a') # 需要的是最终的这个特征图，异常图可以用加法/乘法，显然加法更合理
            
            a_map_list[0] = gaussian_filter(a_map_list[0], sigma=4)
            a_map_list[1] = gaussian_filter(a_map_list[1], sigma=4)
            a_map_list[2] = gaussian_filter(a_map_list[1], sigma=4)
            
            anomaly_map = gaussian_filter(anomaly_map, sigma=4)    # 高斯滤波
            
            #anomaly_map = a_map_list[0]
            #anomaly_map = a_map_list[1]
            #anomaly_map = a_map_list[2]
            
            ano_map = min_max_norm(anomaly_map)                    # 数据的标准化（normalization）是将数据按比例缩放，使之落入一个小的特定区间
            
            # print(ano_map.shape)
            from skimage import morphology
            mask = ano_map
            mask[mask > 0.8] = 1
            mask[mask <= 0.8] = 0
            kernel = morphology.disk(4)
            mask = morphology.opening(mask, kernel)
            mask *= 255
            
            
            img = cv2.cvtColor(img.permute(0, 2, 3, 1).cpu().numpy()[0] * 255, cv2.COLOR_BGR2RGB)
            
            img = np.uint8(min_max_norm(img)*255)
            print(img.shape)
            vis_img = mark_boundaries(img, mask, color=(1, 0, 0), mode='thick')
            
            ano_map = cvt2heatmap(255-ano_map*255)                     # 转化为热度图
            #img = cv2.cvtColor(img.permute(0, 2, 3, 1).cpu().numpy()[0] * 255, cv2.COLOR_BGR2RGB)
            
            #img = np.uint8(min_max_norm(img)*255)
            
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
            plt.imshow(vis_img)
            plt.axis('off')
            
            
            
            if (os.path.exists(img_path)==0):
                os.mkdir(img_path)
                
            # 保存图片
            print(img_path+str(count).replace('/','_')+'.png')
            plt.savefig(img_path+str(count).replace('/','_')+'.png')
            
            count += 1
            
# 生成不需要mask的热度图
def evaluation_visualization_no_seg_(encoder, decoder, res, dataloader,device, print_canshu,score_num,img_path):
    count = 0
    decoder.eval()
    with torch.no_grad():
        for img, label, _   in dataloader:
            #print(ip[0][-20:-4])
            if (label.item() == 0):
                continue
            img = img.to(device)
            inputs = encoder(img)
            outputs = decoder(inputs[3],inputs[0:3],res)  

            anomaly_map, a_map_list = cal_anomaly_map(inputs[0:3], outputs, img.shape[-1], amap_mode='a') # 需要的是最终的这个特征图，异常图可以用加法/乘法，显然加法更合理
            
            a_map_list[0] = gaussian_filter(a_map_list[0], sigma=4)
            a_map_list[1] = gaussian_filter(a_map_list[1], sigma=4)
            a_map_list[2] = gaussian_filter(a_map_list[1], sigma=4)
            
            anomaly_map = gaussian_filter(anomaly_map, sigma=4)    # 高斯滤波
            
            #anomaly_map = a_map_list[0]
            #anomaly_map = a_map_list[1]
            #anomaly_map = a_map_list[2]
            
            ano_map = min_max_norm(anomaly_map)                    # 数据的标准化（normalization）是将数据按比例缩放，使之落入一个小的特定区间
            
            
            ano_map = cvt2heatmap(255-ano_map*255)                     # 转化为热度图
            img = cv2.cvtColor(img.permute(0, 2, 3, 1).cpu().numpy()[0] * 255, cv2.COLOR_BGR2RGB)
            
            img = np.uint8(min_max_norm(img)*255)
            ano_map = show_cam_on_image(img, ano_map)              # 热度图和原图片叠加
            
            # 先画热度图
            
            # 绘制热力图
            h = sns.heatmap(
                data=ano_map[:,:,0], # 指定绘图数据
                cmap='jet', # 指定填充色'PuBuGn'，，jet
                # center=1,
                linewidths=.1, # 设置每个单元格边框的宽度
                annot=False,  # 显示数值
                cbar=False,
                # fmt='.3f',# 以科学计算法显示数据
                vmax=1,
                vmin=0,
                xticklabels=('1', '','','4','', '','7','','','10'),
                yticklabels=('1', '','','4','', '','7','','','10'),
            )
            # ------------设置颜色条刻度字体的大小-------------------------#
            cb = h.figure.colorbar(h.collections[0]) #显示colorbar
            cb.ax.tick_params(labelsize=20)  # 设置colorbar刻度字体大小。

            # 添加标题, fontweight='bold'
            plt.title('AAA', fontsize=35) # 'WS(m/s)'  u'T(°C)'  'P(mbar)'
            plt.xticks(fontsize=28)
            plt.yticks(fontsize=28)
            plt.xlabel('longitude', fontsize=34)  # 经度
            plt.ylabel('latitude', fontsize=34)   # 纬度
            plt.rcParams['savefig.dpi'] = 600
            plt.tight_layout()

            
            
            
            if (os.path.exists(img_path)==0):
                os.mkdir(img_path)    
            # 保存图片
            plt.savefig(img_path+str(count).replace('/','_')+'.png')
            
            count += 1
                            
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path',default='',type=str)          # 待测模型的路径
    parser.add_argument('--res', default=3,type=int)                 # 选择用多少个连接，可选择1、2、3，但是其实代表0、1、2个连接
    parser.add_argument('--seg', default=0,type=int)                 # 选择是否需要分割
    parser.add_argument('--data_path',default="/data/liumingxuan/dingwei/",type=str) # 选择数据集文件夹位置
    parser.add_argument('--print_canshu',default=1,type=int)                                         # 选择是否打印测试集的异常值
    parser.add_argument('--score_num',default=1,type=int)                                            # 选择使用异常图中的多少个异常值作为最终异常分数
    parser.add_argument('--img_path',default='./result_img/',type=str)                               # 如果需要分割，选择输出图像路径
    parser.add_argument('--vis',default=0,type=int)                                                  # 如果需要分割，是否可视化输出
    parser.add_argument('--net',default='wide_res50',type=str)                                       # 可使用的net类型，可选res18、res34、res50、wide_res50
    parser.add_argument('--class_',default='BUSI',type=str)                                          # 数据集
    args = parser.parse_args()
    
    
    print('--------args----------')
    for k in list(vars(args).keys()):
        print('%s: %s' % (k, vars(args)[k]))
    print('--------args----------\n')
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(device)
    
    test_path = args.data_path + args.class_ 
    data_transform, gt_transform = get_data_transforms(256, 256) 
    # 是否需要分割图像
    if args.seg==0:  
        test_data = MVTecDataset(root=test_path, transform=data_transform,phase="test") 
        test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=False,num_workers=8)
    if args.seg==1:
        test_data = MVTecDataset_seg(root=test_path, transform=data_transform,gt_transform = gt_transform,phase="test") 
        test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=False,num_workers=8)
    print('test data finish!')
        
    # 选择使用什么网络
    model_path = args.model_path
    if args.net == 'wide_res50':   
        encoder = wide_resnet50_2(pretrained=True) # 编码器
        encoder = encoder.to(device)
        encoder.eval() # 固定编码器模型参数    
        decoder = de_wide_resnet50_2(pretrained=False) # 解码器，为编码器的反向结构
        decoder = decoder.to(device)
        
        # 加载预训练参数
        ckp = torch.load(model_path)
        new_state_dict = OrderedDict()
        for k, v in ckp.items():
            name = k[0:] # remove `module.`，表面从第7个key值字符取到最后一个字符，正好去掉了module.
            new_state_dict[name] = v #新字典的key值对应的value为一一对应的值。  
        decoder.load_state_dict(new_state_dict)
        decoder.eval()
        
        
    if args.net == 'res18':
        encoder = resnet18(pretrained=True) # 编码器
        encoder = encoder.to(device)
        encoder.eval() # 固定编码器模型参数    
        decoder = de_resnet18(pretrained=False) # 解码器，为编码器的反向结构
        decoder = decoder.to(device)
        
        # 加载预训练参数
        ckp = torch.load(model_path)
        new_state_dict = OrderedDict()
        for k, v in ckp.items():
            name = k[0:] # remove `module.`，表面从第7个key值字符取到最后一个字符，正好去掉了module.
            new_state_dict[name] = v #新字典的key值对应的value为一一对应的值。  
        decoder.load_state_dict(new_state_dict)
        decoder.eval()
        
    if args.net == 'res34':
        encoder = resnet34(pretrained=True) # 编码器
        encoder = encoder.to(device)
        encoder.eval() # 固定编码器模型参数    
        decoder = de_resnet34(pretrained=False) # 解码器，为编码器的反向结构
        decoder = decoder.to(device)
        
        # 加载预训练参数
        ckp = torch.load(model_path)
        new_state_dict = OrderedDict()
        for k, v in ckp.items():
            name = k[0:] # remove `module.`，表面从第7个key值字符取到最后一个字符，正好去掉了module.
            new_state_dict[name] = v #新字典的key值对应的value为一一对应的值。  
        decoder.load_state_dict(new_state_dict)
        decoder.eval()
        
    if args.net == 'res50':
        encoder = resnet50(pretrained=True) # 编码器
        encoder = encoder.to(device)
        encoder.eval() # 固定编码器模型参数    
        decoder = de_resnet50(pretrained=False) # 解码器，为编码器的反向结构
        decoder = decoder.to(device)
        
        # 加载预训练参数
        ckp = torch.load(model_path)
        new_state_dict = OrderedDict()
        for k, v in ckp.items():
            name = k[0:] # remove `module.`，表面从第7个key值字符取到最后一个字符，正好去掉了module.
            new_state_dict[name] = v #新字典的key值对应的value为一一对应的值。  
        decoder.load_state_dict(new_state_dict)
        decoder.eval()
        
    if args.vis == 0 or args.vis ==1 :
        print('eval start!')
        evaluation_me(encoder,decoder, args.res,test_dataloader,device,args.print_canshu,args.score_num)
        
    if args.vis == 1 and args.seg == 1:
        print('vis with ground_truth start!')
        evaluation_visualization(encoder, decoder, args.res, test_dataloader,device, args.print_canshu,args.score_num,args.img_path)
    
    if args.vis == 1 and args.seg == 0:
        print('vis without ground_truth start!')
        evaluation_visualization_no_seg(encoder, decoder, args.res, test_dataloader,device, args.print_canshu,args.score_num,args.img_path)