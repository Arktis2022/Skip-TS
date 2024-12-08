
import torch
import torch.nn.functional as F
def loss_fucntion(a, b, L2): 
    cos_loss = torch.nn.CosineSimilarity()
    l2_loss = torch.nn.MSELoss()
    loss = 0
    
    # Use cosloss only
    if L2 == 0:
        for item in range(len(a)): 
            loss += torch.mean(1 - cos_loss(a[item].view(a[item].shape[0], -1), b[item].view(b[item].shape[0], -1)))  
    
    # Use l2loss and cosloss
    if L2 == 2:
        for item in range(len(a)):
             loss += 0.5*torch.mean(l2_loss(a[item].view(a[item].shape[0],-1),b[item].view(b[item].shape[0],-1)))       
             loss += 0.5*torch.mean(1-cos_loss(a[item].view(a[item].shape[0],-1),b[item].view(b[item].shape[0],-1)))

    # Use l2loss only
    if L2 == 1:
        l2_loss = torch.nn.MSELoss()
        for item in range(len(a)):
             loss += torch.mean(l2_loss(a[item].view(a[item].shape[0],-1),b[item].view(b[item].shape[0],-1)))       

    loss2 = loss_fucntion_2(a, b)

    return loss, loss2

def loss_fucntion_2(a, b): 
    mse_loss = torch.nn.MSELoss()

    a2 = F.interpolate(a[2], size=64, mode='bilinear', align_corners=True)
    b2 = F.interpolate(b[2], size=64, mode='bilinear', align_corners=True)
    l1_1 = torch.mean(mse_loss(a2.view(a2.shape[0],-1),b2.view(b2.shape[0],-1)))
    l1_2 = torch.mean(mse_loss(a[1].view(a[1].shape[0],-1),b[1].view(b[1].shape[0],-1)))
    loss2_1 = torch.abs(l1_1 - l1_2)
    
    a1 = F.interpolate(a[1], size=64, mode='bilinear', align_corners=True)
    b1 = F.interpolate(b[1], size=64, mode='bilinear', align_corners=True)
    l2_1 = torch.mean(mse_loss(a1.view(a1.shape[0],-1),b1.view(b1.shape[0],-1)))
    l2_2 = torch.mean(mse_loss(a[0].view(a[0].shape[0],-1),b[0].view(b[0].shape[0],-1)))
    loss2_2 = torch.abs(l2_1 - l2_2)
    
    double_a2 = F.interpolate(a2, size = 64, mode='bilinear', align_corners=True)
    double_b2 = F.interpolate(b2, size = 64, mode='bilinear', align_corners=True)
    l3_1 = l2_1 = torch.mean(mse_loss(double_a2.view(double_a2.shape[0],-1), double_b2.view(double_b2.shape[0],-1)))
    l3_2 = l2_2
    loss2_3 = torch.abs(l3_1 - l3_2)

    loss2 = loss2_1 + loss2_2 + loss2_3
    return loss2