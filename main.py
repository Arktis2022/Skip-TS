#-*-coding:gb2312-*-
import torch
from dataset import get_data_transforms 
from torchvision.datasets import ImageFolder
import numpy as np
import random
import os
from torch.utils.data import DataLoader
from resnet import resnet18, resnet34, resnet50, wide_resnet50_2
from de_resnet import de_resnet18, de_resnet34, de_wide_resnet50_2, de_resnet50 
from dataset import MVTecDataset, MVTecDataset_seg 
import torch.backends.cudnn as cudnn
import argparse
from test import evaluation_me,evaluation_visualization,evaluation,evaluation_visualization_no_seg
from torch.nn import functional as F
import argparse
import sys
# 设置随机数种子
def setup_seed(seed): 
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
# 损失函数，这里可以做消融研究
def loss_fucntion(a, b, L2): # 输入两个张量数组
    cos_loss = torch.nn.CosineSimilarity()
    #print(a[0].size()) # a[0] = [16,256,64,64]
    #print(a[1].size()) # a[1] = [16,512,32,32]
    #print(a[2].size()) # a[2] = [16,1024,16,16]
    loss = 0
    
    # 使用cosloss
    if L2 == 0:
        for item in range(len(a)): # 根据a中的每一个张量
            #print( torch.mean((1-cos_loss(a[item].view(a[item].shape[0],-1),b[item].view(b[item].shape[0],-1)))) ) # 计算得到的16个数字（batch.size是16）
            loss += torch.mean(1-cos_loss(a[item].view(a[item].shape[0],-1),b[item].view(b[item].shape[0],-1)))  # 按通道，计算损失函数图，求其均值，之后把各个数组的loss直接加起来；mean()函数的参数：dim=0,按列求平均值，返回的形状是（1，列数）；dim=1,按行求平均值，返回的形状是（行数，1）,默认不设置dim的时候，返回的是所有元素的平均值
    
    # 使用l2loss+cosloss
    if L2 == 2:
        l2_loss = torch.nn.MSELoss()
        for item in range(len(a)):
             loss += 0.5*torch.mean(l2_loss(a[item].view(a[item].shape[0],-1),b[item].view(b[item].shape[0],-1)))       
             loss += 0.5*torch.mean(1-cos_loss(a[item].view(a[item].shape[0],-1),b[item].view(b[item].shape[0],-1)))
    loss2 = loss_fucntion_2(a,b)
    
    # 使用l2loss
    if L2 == 1:
        l2_loss = torch.nn.MSELoss()
        for item in range(len(a)):
             loss += torch.mean(l2_loss(a[item].view(a[item].shape[0],-1),b[item].view(b[item].shape[0],-1)))       
    loss2 = loss_fucntion_2(a,b)
    #print(loss)
    #print(loss2) 
    #sys.exit()
    return loss,loss2
    
# 尝试计算组间一致性损失

def loss_fucntion_2(a, b): # 输入两个张量数组
    mse_loss = torch.nn.MSELoss()
    # 将a2和b2上采样得到的结果和a1、b1没上采样得到的结果比较
    a2 = F.interpolate(a[2], size=32, mode='bilinear', align_corners=True)
    b2 = F.interpolate(b[2], size=32, mode='bilinear', align_corners=True)
    l2 = torch.mean(mse_loss(a2.view(a2.shape[0],-1),b2.view(b2.shape[0],-1)))
    l1 = torch.mean(mse_loss(a[1].view(a[1].shape[0],-1),b[1].view(b[1].shape[0],-1)))
    loss2_1 = torch.abs(l2-l1)
    
    # 将a1和b1上采样得到结果，与a0和b0没上采用得到的结果比较
    l0 = torch.mean(mse_loss(a[0].view(a[0].shape[0],-1),b[0].view(b[0].shape[0],-1)))
    
    a1 = F.interpolate(a[1], size=64, mode='bilinear', align_corners=True)
    b1 = F.interpolate(b[1], size=64, mode='bilinear', align_corners=True)
    l1 = torch.mean(mse_loss(a1.view(a1.shape[0],-1),b1.view(b1.shape[0],-1)))
    loss2_2 = torch.abs(l1-l0)
    
    
    #print(loss2_1,loss2_2)
    #sys.exit()
    loss2 = loss2_1+loss2_2
    return loss2
        
    
def train(class_,epochs,learning_rate,res,batch_size,print_epoch,seg,data_path,save_path,print_canshu,score_num,print_loss,img_path,vis,cut,layerloss,rate,print_max,net,L2,seed): 
    image_size = 256
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(device)
    if (os.path.exists(save_path)==0):
        os.mkdir(save_path)
    data_transform, gt_transform = get_data_transforms(image_size, image_size) 
    
    train_path = data_path + class_ + '/train'
    test_path = data_path + class_ 
    ckp_path = save_path + net +class_ 
    
    train_data = ImageFolder(root=train_path, transform=data_transform)
    train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True,num_workers=8)
    
    # 是否需要分割图像
    if seg==0:  
        test_data = MVTecDataset(root=test_path, transform=data_transform,phase="test") 
        test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=False,num_workers=8)
    if seg==1:
        test_data = MVTecDataset_seg(root=test_path, transform=data_transform,gt_transform = gt_transform,phase="test") 
        test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=False,num_workers=8)
        
    # 选择使用什么网络
    if net == 'wide_res50':   
        encoder = wide_resnet50_2(pretrained=True) # 编码器
        encoder = encoder.to(device)
        encoder.eval() # 固定编码器模型参数    
        decoder = de_wide_resnet50_2(pretrained=False) # 解码器，为编码器的反向结构
        decoder = decoder.to(device)
    if net == 'res18':
        encoder = resnet18(pretrained=True) # 编码器
        encoder = encoder.to(device)
        encoder.eval() # 固定编码器模型参数    
        decoder = de_resnet18(pretrained=False) # 解码器，为编码器的反向结构
        decoder = decoder.to(device)
    if net == 'res34':
        encoder = resnet34(pretrained=True) # 编码器
        encoder = encoder.to(device)
        encoder.eval() # 固定编码器模型参数    
        decoder = de_resnet34(pretrained=False) # 解码器，为编码器的反向结构
        decoder = decoder.to(device)
    if net == 'res50':
        encoder = resnet50(pretrained=True) # 编码器
        encoder = encoder.to(device)
        encoder.eval() # 固定编码器模型参数    
        decoder = de_resnet50(pretrained=False) # 解码器，为编码器的反向结构
        decoder = decoder.to(device)
    
    optimizer = torch.optim.Adam(list(decoder.parameters()), lr=learning_rate, betas=(0.5,0.999)) # 给它一个可进行迭代优化的包含了所有参数（所有的参数必须是变量s）的列表
    
    max_auc = []
    max_auc_epoch = []
    max_pr = []
    max_pr_epoch = []
    
    # 开始训练
    for epoch in range(epochs):
        decoder.train()
        loss_list = []
        for img, label in train_dataloader:
            img = img.to(device) 
            inputs = encoder(img)
            outputs = decoder(inputs[3],inputs[0:3],res)  
            
            # 选择损失函数  
            if layerloss==0:
                loss = loss_fucntion(inputs[0:3], outputs,L2)[0] 
            if layerloss==1:
                loss = loss_fucntion(inputs[0:3], outputs,L2)[0]+rate*loss_fucntion(inputs[0:3], outputs,L2)[1]
                
                 
            optimizer.zero_grad() 
            loss.backward()
            optimizer.step()
            loss_list.append(loss.item()) 
            
             
        if print_loss==1:
            print('epoch [{}/{}], loss:{:.4f}'.format(epoch + 1, epochs, np.mean(loss_list)))
        if (epoch + 1) % print_epoch == 0:
            # 测试集没有mask
            if seg==0:
                auroc_sp = evaluation_me(encoder, decoder, res, test_dataloader, device,print_canshu,score_num)
                print('epoch：', (epoch+1))
                print('Sample Auroc{:.3f}'.format(auroc_sp))
                max_auc.append(auroc_sp)
                max_auc_epoch.append(epoch+1)
                if print_max==1:
                  print('max_auc = ' , max(max_auc) )
                  print('max_epoch = ', max_auc_epoch[max_auc.index(max(max_auc))])
                print('------------------')
                torch.save(decoder.state_dict(), ckp_path+str(epoch+1)+str(seed)+'auc='+str(auroc_sp)+'.pth')
                if vis==1: # 没有mask时的可视化输出
                    evaluation_visualization_no_seg(encoder, decoder, res, test_dataloader,device, print_canshu,score_num,img_path)
                    
            # 测试集有mask且需要定位
            if seg==1:
                # 正常流程走一遍
                # 绘图
                if vis==1:
                    evaluation_visualization(encoder, decoder, res, test_dataloader,device, print_canshu,score_num,img_path)
                auroc_px, auroc_sp, aupro_px = evaluation(encoder,decoder,res,test_dataloader, device,img_path)
                print('Pixel Auroc:{:.3f}, Sample Auroc{:.3f}, Pixel Aupro{:.3}'.format(auroc_px, auroc_sp, aupro_px))
                max_auc.append(auroc_px)
                max_auc_epoch.append(epoch+1)
                max_pr.append(aupro_px)
                max_pr_epoch.append(epoch+1)
                print('max_auc = ' , max(max_auc) )
                print('max_epoch = ', max_auc_epoch[max_auc.index(max(max_auc))])
                print('max_pr = ' , max(max_pr) )
                print('max_epoch = ', max_pr_epoch[max_pr.index(max(max_pr))])
                torch.save(decoder.state_dict(), ckp_path+str(epoch+1)+str(seed)+'auc='+str(auroc_sp)+'.pth')
    return auroc_sp

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', default=200,type=int) # 训练周期
    parser.add_argument('--res', default=3,type=int)      # 选择用多少个连接，可选择1、2、3，但是其实代表0、1、2个连接
    parser.add_argument('--learning_rate', default=0.005,type=float) # 学习率
    parser.add_argument('--batch_size', default=16,type=int)         # 不解释
    parser.add_argument('--seed', default=[111,250,444,999,114514],nargs='+',type=int)              # 随机数种子
    parser.add_argument('--class_', default='head_ct',type=str)      # 选择子数据集
    parser.add_argument('--seg', default=0,type=int)                 # 选择是否需要分割
    parser.add_argument('--print_epoch', default=50,type=int)        # 选择过多少个epoch进行一次打印
    parser.add_argument('--data_path',default='/data/liumingxuan/dingwei/',type=str) # 选择数据集文件夹位置
    parser.add_argument('--save_path',default='./checkpoints/',type=str)                             # 选择模型文件保存位置
    parser.add_argument('--print_canshu',default=1,type=int)                                         # 选择是否打印测试集的异常值
    parser.add_argument('--score_num',default=1,type=int)                                            # 选择使用异常图中的多少个异常值作为最终异常分数
    parser.add_argument('--print_loss',default=1,type=int)
    parser.add_argument('--img_path',default='./result_img/',type=str)                              # 如果需要分割，选择路径
    parser.add_argument('--vis',default=0,type=int)                              # 如果需要分割，是否可视化输出
    parser.add_argument('--cut',default=0,type=int)                         # 是否使用cutpaste数据增强
    parser.add_argument('--layerloss',default=1,type=int)                         # 是否使用组间一致性损失
    parser.add_argument('--rate',default=0.05,type=float)                         # 组间一致性损失占比
    parser.add_argument('--print_max',default=1,type=int)                         # 是否打印最佳auc
    parser.add_argument('--net',default='wide_res50',type=str)                    # 可使用的net类型，可选res18、res34、res50、wide_res50
    parser.add_argument('--L2',default=0,type=int)                                # 是否使用L2损失函数
    args = parser.parse_args()
    
    
    print('--------args----------')
    for k in list(vars(args).keys()):
        print('%s: %s' % (k, vars(args)[k]))
    print('--------args----------\n')
    
    if args.class_ == 'all':
        all = ['head_ct2','brainMRI2','covid5','BUSI_true','breast','OCT','CP','CP2','NAFLD2','OTH','covidxray','cell','funi','boluo_2']
        epoch_ = [200,400,200,400,200,200,200,200,100,200,200,200,200,200]
        rate_ = [1e-5,0.05,1e-6,0.05,0.0001,0.05,0.05,0.05,0.0001,0.01,0.01,0.005,0.05,0.01]
        for class_,epoch,rate in zip(all,epoch_,rate_):
            print(class_)
            print(epoch)
            print(rate)
            print_epoch = epoch
            for seed in args.seed:
                print('*************************')
                print('seed:',seed)
                setup_seed(seed)
                train(class_,epoch,args.learning_rate,args.res,args.batch_size,print_epoch,args.seg,args.data_path,args.save_path,args.print_canshu,args.score_num,args.print_loss,args.img_path,args.vis,args.cut,args.layerloss,rate,args.print_max,args.net,args.L2,seed)
                print('*************************')  
    
    if args.class_ != 'all':
            for seed in args.seed:
                print('*************************')
                print('seed:',seed)
                setup_seed(seed)
                train(args.class_,args.epochs,args.learning_rate,args.res,args.batch_size,args.print_epoch,args.seg,args.data_path,args.save_path,args.print_canshu,args.score_num,args.print_loss,args.img_path,args.vis,args.cut,args.layerloss,args.rate,args.print_max,args.net,args.L2,seed)
                print('*************************') 
