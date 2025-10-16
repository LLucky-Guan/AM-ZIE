import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import torchvision.transforms as transforms
import numpy as np
from torchvision.models.vgg import vgg16, vgg19

class MyLoss(nn.Module):
    def __init__(self):
        super(MyLoss, self).__init__()
        # 定义你的各类损失
        # self.l2_loss = nn.MSELoss()
        self.L_color = L_color()
        self.L_spa = L_spa()
        self.L_exp = L_exp()
        self.L_TV = L_TV()
        
    def forward(self, img_lowlight, enhance_image, A):
        # 计算各项损失
        Loss_TV =  200*self.L_TV(A)  
        loss_spa = 5*torch.mean(self.L_spa(enhance_image, img_lowlight))
        loss_col = torch.mean(self.L_color(enhance_image))
        loss_exp = 5*torch.mean(self.L_exp(enhance_image,A))
        # loss_l2 = 0*torch.mean(self.L_Per(enhance_image,img_lowlight))

        # 总损失为各个损失的和
        total_loss = Loss_TV + loss_exp + loss_spa  + loss_col 
        # 返回总损失以及各个损失的字典
        losses = {
            "tv_loss": Loss_TV.item(),
            "spa_loss": loss_spa.item(),
            "color_loss": loss_col.item(),
            "exp_loss": loss_exp.item(),
            "total_loss": total_loss.item(),
        }
        return total_loss, losses


class L_exp(nn.Module):
    def __init__(self, patch_size=16, a=0.5,b=0.1):  # 默认a=0.5,b=0.3
        super(L_exp, self).__init__()
        self.pool = nn.AvgPool2d(patch_size)
        self.a=a
        self.b=b

    def forward(self, x, A):
        B, C, H, W = x.shape
    
        # 1. 亮度分段编码（0-1归一化）
        global_mean = x.mean((1,2,3))  # [B]
        local_mean = self.pool(x)

        # 分段：暗区(<0.4)、正常(0.4-0.6)、亮区(>0.6)
        dark_mask = torch.sigmoid(10*(0.4 - global_mean))  # 暗区权重[0,1]
        bright_mask = torch.sigmoid(10*(global_mean - 0.6))  # 亮区权重[0,1]
        mid_mask = 1 - dark_mask - bright_mask  # 正常区权重
        
        # 2. 动态基准目标（结合A的物理意义）
        A_scaled = (A + 1) / 2  # [-1,1]→[0,1]，暗图A≈0，亮图A≈1
        # print(A_scaled.mean())
        # 3. 分段补偿策略
        target_mean = (
            dark_mask * (self.a + self.b*A_scaled.mean()) +   # 暗区目标
            mid_mask * self.a  +  # 正常区保底
            bright_mask * (self.a-self.b*A_scaled.mean())  # 亮区目标
        ).view(B, 1, 1, 1)  # [B,1,1,1]适配池化结果

        
        exp_loss = torch.pow(local_mean - target_mean, 2)
        # exp_loss = torch.pow(local_mean - 0.6, 2)
        
        return exp_loss

    
class L_spa(nn.Module):

    def __init__(self):
        super(L_spa, self).__init__()
        # print(1)kernel = torch.FloatTensor(kernel).unsqueeze(0).unsqueeze(0)
        kernel_left = torch.FloatTensor( [[0,0,0],[-1,1,0],[0,0,0]]).cuda().unsqueeze(0).unsqueeze(0)
        kernel_right = torch.FloatTensor( [[0,0,0],[0,1,-1],[0,0,0]]).cuda().unsqueeze(0).unsqueeze(0)
        kernel_up = torch.FloatTensor( [[0,-1,0],[0,1, 0 ],[0,0,0]]).cuda().unsqueeze(0).unsqueeze(0)
        kernel_down = torch.FloatTensor( [[0,0,0],[0,1, 0],[0,-1,0]]).cuda().unsqueeze(0).unsqueeze(0)
        self.weight_left = nn.Parameter(data=kernel_left, requires_grad=False)
        self.weight_right = nn.Parameter(data=kernel_right, requires_grad=False)
        self.weight_up = nn.Parameter(data=kernel_up, requires_grad=False)
        self.weight_down = nn.Parameter(data=kernel_down, requires_grad=False)
        self.pool = nn.AvgPool2d(8)
    def forward(self, org , enhance ):
        b,c,h,w = org.shape

        org_mean = torch.mean(org,1,keepdim=True)
        enhance_mean = torch.mean(enhance,1,keepdim=True)

        org_pool =  self.pool(org_mean)			
        enhance_pool = self.pool(enhance_mean)	

        weight_diff =torch.max(torch.FloatTensor([1]).cuda() + 10000*torch.min(org_pool - torch.FloatTensor([0.3]).cuda(),torch.FloatTensor([0]).cuda()),torch.FloatTensor([0.5]).cuda())
        E_1 = torch.mul(torch.sign(enhance_pool - torch.FloatTensor([0.5]).cuda()) ,enhance_pool-org_pool)


        D_org_letf = F.conv2d(org_pool , self.weight_left, padding=1)
        D_org_right = F.conv2d(org_pool , self.weight_right, padding=1)
        D_org_up = F.conv2d(org_pool , self.weight_up, padding=1)
        D_org_down = F.conv2d(org_pool , self.weight_down, padding=1)

        D_enhance_letf = F.conv2d(enhance_pool , self.weight_left, padding=1)
        D_enhance_right = F.conv2d(enhance_pool , self.weight_right, padding=1)
        D_enhance_up = F.conv2d(enhance_pool , self.weight_up, padding=1)
        D_enhance_down = F.conv2d(enhance_pool , self.weight_down, padding=1)

        D_left = torch.pow(D_org_letf - D_enhance_letf,2)
        D_right = torch.pow(D_org_right - D_enhance_right,2)
        D_up = torch.pow(D_org_up - D_enhance_up,2)
        D_down = torch.pow(D_org_down - D_enhance_down,2)
        E = (D_left + D_right + D_up +D_down)

        return E

class L_TV(nn.Module):
    def __init__(self,TVLoss_weight=1):
        super(L_TV,self).__init__()
        self.TVLoss_weight = TVLoss_weight

    def forward(self,x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h =  (x.size()[2]-1) * x.size()[3]
        count_w = x.size()[2] * (x.size()[3] - 1)
        h_tv = torch.pow((x[:,:,1:,:]-x[:,:,:h_x-1,:]),2).sum()
        w_tv = torch.pow((x[:,:,:,1:]-x[:,:,:,:w_x-1]),2).sum()
        return self.TVLoss_weight*2*(h_tv/count_h+w_tv/count_w)/batch_size

class L_color(nn.Module):
    def __init__(self):
        super(L_color, self).__init__()

    def forward(self, x):
        b, c, h, w = x.shape
        # 计算每个通道的平均值
        mean_rgb = torch.mean(x, [2, 3], keepdim=True)
        mr, mg, mb = torch.split(mean_rgb, 1, dim=1)
        # 计算各个颜色通道之间的距离，约束色彩平衡
        Drg = torch.pow(mr - mg, 2)
        Drb = torch.pow(mr - mb, 2)
        Dgb = torch.pow(mb - mg, 2)
        k = torch.pow(torch.pow(Drg, 2) + torch.pow(Drb, 2) + torch.pow(Dgb, 2), 0.5)
        # 返回色彩平衡损失
        return k.mean()



