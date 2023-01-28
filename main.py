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
import cutpaste
import sys
# �������������
def setup_seed(seed): 
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
# ��ʧ��������������������о�
def loss_fucntion(a, b, L2): # ����������������
    cos_loss = torch.nn.CosineSimilarity()
    #print(a[0].size()) # a[0] = [16,256,64,64]
    #print(a[1].size()) # a[1] = [16,512,32,32]
    #print(a[2].size()) # a[2] = [16,1024,16,16]
    loss = 0
    
    # ʹ��cosloss
    if L2 == 0:
        for item in range(len(a)): # ����a�е�ÿһ������
            #print( torch.mean((1-cos_loss(a[item].view(a[item].shape[0],-1),b[item].view(b[item].shape[0],-1)))) ) # ����õ���16�����֣�batch.size��16��
            loss += torch.mean(1-cos_loss(a[item].view(a[item].shape[0],-1),b[item].view(b[item].shape[0],-1)))  # ��ͨ����������ʧ����ͼ�������ֵ��֮��Ѹ��������lossֱ�Ӽ�������mean()�����Ĳ�����dim=0,������ƽ��ֵ�����ص���״�ǣ�1����������dim=1,������ƽ��ֵ�����ص���״�ǣ�������1��,Ĭ�ϲ�����dim��ʱ�򣬷��ص�������Ԫ�ص�ƽ��ֵ
    
    # ʹ��l2loss+cosloss
    if L2 == 2:
        l2_loss = torch.nn.MSELoss()
        for item in range(len(a)):
             loss += 0.5*torch.mean(l2_loss(a[item].view(a[item].shape[0],-1),b[item].view(b[item].shape[0],-1)))       
             loss += 0.5*torch.mean(1-cos_loss(a[item].view(a[item].shape[0],-1),b[item].view(b[item].shape[0],-1)))
    loss2 = loss_fucntion_2(a,b)
    
    # ʹ��l2loss
    if L2 == 1:
        l2_loss = torch.nn.MSELoss()
        for item in range(len(a)):
             loss += torch.mean(l2_loss(a[item].view(a[item].shape[0],-1),b[item].view(b[item].shape[0],-1)))       
    loss2 = loss_fucntion_2(a,b)
    #print(loss)
    #print(loss2) 
    #sys.exit()
    return loss,loss2
    
# ���Լ������һ������ʧ

def loss_fucntion_2(a, b): # ����������������
    mse_loss = torch.nn.MSELoss()
    # ��a2��b2�ϲ����õ��Ľ����a1��b1û�ϲ����õ��Ľ���Ƚ�
    a2 = F.interpolate(a[2], size=32, mode='bilinear', align_corners=True)
    b2 = F.interpolate(b[2], size=32, mode='bilinear', align_corners=True)
    l2 = torch.mean(mse_loss(a2.view(a2.shape[0],-1),b2.view(b2.shape[0],-1)))
    l1 = torch.mean(mse_loss(a[1].view(a[1].shape[0],-1),b[1].view(b[1].shape[0],-1)))
    loss2_1 = torch.abs(l2-l1)
    
    # ��a1��b1�ϲ����õ��������a0��b0û�ϲ��õõ��Ľ���Ƚ�
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
    
    # �Ƿ���Ҫ�ָ�ͼ��
    if seg==0:  
        test_data = MVTecDataset(root=test_path, transform=data_transform,phase="test") 
        test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=False,num_workers=8)
    if seg==1:
        test_data = MVTecDataset_seg(root=test_path, transform=data_transform,gt_transform = gt_transform,phase="test") 
        test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=False,num_workers=8)
        
    # ѡ��ʹ��ʲô����
    if net == 'wide_res50':   
        encoder = wide_resnet50_2(pretrained=True) # ������
        encoder = encoder.to(device)
        encoder.eval() # �̶�������ģ�Ͳ���    
        decoder = de_wide_resnet50_2(pretrained=False) # ��������Ϊ�������ķ���ṹ
        decoder = decoder.to(device)
    if net == 'res18':
        encoder = resnet18(pretrained=True) # ������
        encoder = encoder.to(device)
        encoder.eval() # �̶�������ģ�Ͳ���    
        decoder = de_resnet18(pretrained=False) # ��������Ϊ�������ķ���ṹ
        decoder = decoder.to(device)
    if net == 'res34':
        encoder = resnet34(pretrained=True) # ������
        encoder = encoder.to(device)
        encoder.eval() # �̶�������ģ�Ͳ���    
        decoder = de_resnet34(pretrained=False) # ��������Ϊ�������ķ���ṹ
        decoder = decoder.to(device)
    if net == 'res50':
        encoder = resnet50(pretrained=True) # ������
        encoder = encoder.to(device)
        encoder.eval() # �̶�������ģ�Ͳ���    
        decoder = de_resnet50(pretrained=False) # ��������Ϊ�������ķ���ṹ
        decoder = decoder.to(device)
    
    optimizer = torch.optim.Adam(list(decoder.parameters()), lr=learning_rate, betas=(0.5,0.999)) # ����һ���ɽ��е����Ż��İ��������в��������еĲ��������Ǳ���s�����б�
    
    max_auc = []
    max_auc_epoch = []
    max_pr = []
    max_pr_epoch = []
    
    # ��ʼѵ��
    for epoch in range(epochs):
        decoder.train()
        loss_list = []
        for img, label in train_dataloader:
            if cut==1:
                img = cutpaste.cutpaste(img)
            img = img.to(device) 
            inputs = encoder(img)
            outputs = decoder(inputs[3],inputs[0:3],res)  
            
            # ѡ����ʧ����  
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
            # ���Լ�û��mask
            if seg==0:
                auroc_sp = evaluation_me(encoder, decoder, res, test_dataloader, device,print_canshu,score_num)
                print('epoch��', (epoch+1))
                print('Sample Auroc{:.3f}'.format(auroc_sp))
                max_auc.append(auroc_sp)
                max_auc_epoch.append(epoch+1)
                if print_max==1:
                  print('max_auc = ' , max(max_auc) )
                  print('max_epoch = ', max_auc_epoch[max_auc.index(max(max_auc))])
                print('------------------')
                torch.save(decoder.state_dict(), ckp_path+str(epoch+1)+str(seed)+'auc='+str(auroc_sp)+'.pth')
                if vis==1: # û��maskʱ�Ŀ��ӻ����
                    evaluation_visualization_no_seg(encoder, decoder, res, test_dataloader,device, print_canshu,score_num,img_path)
                    
            # ���Լ���mask����Ҫ��λ
            if seg==1:
                # ����������һ��
                # ��ͼ
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
    parser.add_argument('--epochs', default=200,type=int) # ѵ������
    parser.add_argument('--res', default=3,type=int)      # ѡ���ö��ٸ����ӣ���ѡ��1��2��3��������ʵ����0��1��2������
    parser.add_argument('--learning_rate', default=0.005,type=float) # ѧϰ��
    parser.add_argument('--batch_size', default=16,type=int)         # ������
    parser.add_argument('--seed', default=[111,250,444,999,114514],nargs='+',type=int)              # ���������
    parser.add_argument('--class_', default='head_ct',type=str)      # ѡ�������ݼ�
    parser.add_argument('--seg', default=0,type=int)                 # ѡ���Ƿ���Ҫ�ָ�
    parser.add_argument('--print_epoch', default=50,type=int)        # ѡ������ٸ�epoch����һ�δ�ӡ
    parser.add_argument('--data_path',default='/data/liumingxuan/dingwei/',type=str) # ѡ�����ݼ��ļ���λ��
    parser.add_argument('--save_path',default='./checkpoints/',type=str)                             # ѡ��ģ���ļ�����λ��
    parser.add_argument('--print_canshu',default=1,type=int)                                         # ѡ���Ƿ��ӡ���Լ����쳣ֵ
    parser.add_argument('--score_num',default=1,type=int)                                            # ѡ��ʹ���쳣ͼ�еĶ��ٸ��쳣ֵ��Ϊ�����쳣����
    parser.add_argument('--print_loss',default=1,type=int)
    parser.add_argument('--img_path',default='./result_img/',type=str)                              # �����Ҫ�ָѡ��·��
    parser.add_argument('--vis',default=0,type=int)                              # �����Ҫ�ָ�Ƿ���ӻ����
    parser.add_argument('--cut',default=0,type=int)                         # �Ƿ�ʹ��cutpaste������ǿ
    parser.add_argument('--layerloss',default=1,type=int)                         # �Ƿ�ʹ�����һ������ʧ
    parser.add_argument('--rate',default=0.05,type=float)                         # ���һ������ʧռ��
    parser.add_argument('--print_max',default=1,type=int)                         # �Ƿ��ӡ���auc
    parser.add_argument('--net',default='wide_res50',type=str)                    # ��ʹ�õ�net���ͣ���ѡres18��res34��res50��wide_res50
    parser.add_argument('--L2',default=0,type=int)                                # �Ƿ�ʹ��L2��ʧ����
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



"""

                                _ooOoo_
//                             o8888888o
//                             88" . "88
//                             (| -_- |)
//                              O\ = /O
//                           ____/`---'\____
//                        .   ' \\| |// `.
//                         / \\||| : |||// \
//                        / _||||| -:- |||||- \
//                         | | \\\ - /// | |
//                       | \_| ''\---/'' | |
//                        \ .-\__ `-` ___/-. /
//                    ___`. .' /--.--\ `. . __
//                  ."" '< `.___\_<|>_/___.' >'"".
//                 | | : `- \`.;`\ _ /`;.`/ - ` : | |
//                    \ \ `-. \_ __\ /__ _/ .-` / /
//           ======`-.____`-.___\_____/___.-`____.-'======
//                              `=---='
//
//           .............................................
//                     ���汣��             ����BUG
//            ��Ի:
//                     д��¥��д�ּ䣬д�ּ������Ա��
//                     ������Աд�������ó��򻻾�Ǯ��
//                     ����ֻ���������������������ߣ�
//                     ��������ո��գ����������긴�ꡣ
//                     ��Ը�������Լ䣬��Ը�Ϲ��ϰ�ǰ��
//                     ���۱������Ȥ���������г���Ա��
//                     ����Ц��߯��񲣬��Ц�Լ���̫����
//                     ��������Ư���ã��ĸ���ó���Ա��
"""