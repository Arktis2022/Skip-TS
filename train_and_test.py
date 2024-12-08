
import torch
import os
from dataset import get_data_transforms, MVTecDataset
from encoder import wide_resnet50_2
from decoder import de_wide_resnet50_2
from torchvision.datasets import ImageFolder
from loss_function import loss_fucntion
import tqdm
import numpy as np
from eval_func import evaluation

def train(class_,epochs,learning_rate,res,
          batch_size,print_epoch,data_path,save_path,
          score_num,print_loss,
          layerloss,rate,print_max,net,L2,seed): 
    image_size = 256
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(device)
    if (os.path.exists(save_path)==0):
        os.mkdir(save_path)
    data_transform, gt_transform = get_data_transforms(image_size, image_size) 
    
    train_path = data_path + '/train'
    test_path = data_path 
    ckp_path = save_path + net
    
    train_data = ImageFolder(root=train_path, transform=data_transform)
    train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True,num_workers=8)
    
    test_data = MVTecDataset(root=test_path, transform=data_transform,phase="test") 
    test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=False,num_workers=8)

    encoder = wide_resnet50_2(pretrained= True) 
    encoder = encoder.to(device)
    encoder.eval() 
    decoder = de_wide_resnet50_2(pretrained = False) 
    decoder = decoder.to(device)

    optimizer = torch.optim.Adam(list(decoder.parameters()), 
                                 lr=learning_rate, 
                                 betas=(0.5,0.999)) 
    
    max_auc = []
    max_auc_epoch = []

    for epoch in range(epochs):
        decoder.train()
        loss_list = []
        for img, label in tqdm.tqdm(train_dataloader):
            img = img.to(device) 
            inputs = encoder(img)
            outputs = decoder(inputs[3],inputs[0:3],res)  
            
            # In the paper we use layerloss = 1
            if layerloss==0:
                loss = loss_fucntion(inputs[0:3], outputs,L2)[0]  

            if layerloss==1:
                loss = loss_fucntion(inputs[0:3], outputs,L2)[0] + rate * loss_fucntion(inputs[0:3], outputs,L2)[1]
                
            optimizer.zero_grad() 
            loss.backward()
            optimizer.step()
            loss_list.append(loss.item()) 
             
        if print_loss==1:
            if((epoch + 1) % 10 == 0):
                print('epoch [{}/{}], loss:{:.4f}'.format(epoch + 1, epochs, np.mean(loss_list)))
        if (epoch + 1) % print_epoch == 0:
                
                # Auroc is calculated every 10 epochs, and the max is returned
                auroc_sp, aupr_sp = evaluation(encoder, decoder, res, test_dataloader, device, score_num)
                print('epoch:', (epoch+1))
                print('Sample Auroc{:.3f}'.format(auroc_sp))
                print('Sample Aupr{:.3f}'.format(aupr_sp))
                max_auc.append(auroc_sp)
                max_auc_epoch.append(epoch+1)
                if print_max==1:
                    print('max_auc = ' , max(max_auc) )
                    print('max_epoch = ', max_auc_epoch[max_auc.index(max(max_auc))])
                print('------------------')
                torch.save(decoder.state_dict(), ckp_path+str(epoch+1)+str(seed)+'auc='+str(auroc_sp)+'.pth')              
    return auroc_sp