import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import model
from Myloss import MyLoss
from torchvision.utils import save_image
from dataloader import LLdataset
from evaluation import Evaluation
from torch.optim.lr_scheduler import StepLR
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from thop import profile  # 新增

def print_model_profile(model, input_tensor):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"模型总参数量: {total_params:,}, 可训练参数量: {trainable_params:,}")

    flops, params = profile(model, inputs=(input_tensor,))
    print(f"FLOPs: {flops / 1e9:.4f} GFLOPs")  # 转换成 GigaFLOPs

plt.rc('font', family='serif')             # 设置为衬线字体（罗马字体）
plt.rc('mathtext', fontset='cm')           # 使用 Computer Modern 字体（LaTeX 默认的罗马字体）
plt.rc('axes', titlesize=18)               # 调整标题字号（可选）
plt.rc('axes', labelsize=14)               # 调整坐标轴标签字号（可选）

class Trainer:
    def __init__(self, config):
        self.config = config
        self.best_psnr = 0.0
        self.best_ssim = 0.0
        self.best_piqe = float('inf')
        self.best_niqe = float('inf')
        self.best_brisque = float('inf')

        # Initialize model
        self.DCE_net = model.EnhanceNet().cuda()
        self.evaluation = Evaluation()

        if config.load_pretrain:
            self.DCE_net.load_state_dict(torch.load(config.pretrain_dir))

        # Initialize dataset and dataloader using LLdataset
        data_loader = LLdataset(config)
        self.train_loader, self.val_loader = data_loader.get_loaders()

        # Initialize loss function and optimizer
        self.loss_fn = MyLoss().cuda()
        self.optimizer = optim.Adam(self.DCE_net.parameters(), lr=config.lr, weight_decay=config.weight_decay)
        self.scheduler = StepLR(self.optimizer, step_size=10, gamma=0.9)  

        # Create folders to save models and results
        os.makedirs(config.snapshots_folder, exist_ok=True)
        os.makedirs(config.output_folder, exist_ok=True)
    
    def train(self):
        self.DCE_net.train()

        for epoch in range(self.config.num_epochs):
            epoch_loss = 0.0
            for iteration, (img_lowlight, _) in enumerate(self.train_loader):
                self.DCE_net.train()
                img_lowlight = img_lowlight.cuda()
                
                # Forward pass
                enhanced_image, A = self.DCE_net(img_lowlight)

                loss, losses = self.loss_fn(img_lowlight, enhanced_image, A)

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.DCE_net.parameters(), self.config.grad_clip_norm)
                self.optimizer.step()

                # Accumulate the epoch loss for tracking
                epoch_loss += loss.item() / len(self.train_loader)

                # Display training progress
                if (iteration + 1) % self.config.display_iter == 0:
                    loss_details = ", ".join([f"{key}: {val:.6f}" for key, val in losses.items()])
                    print(f"Epoch [{epoch + 1}/{self.config.num_epochs}], Iteration [{iteration + 1}/{len(self.train_loader)}], {loss_details}")
            
            self.scheduler.step() 
            print(f"Epoch {epoch+1}: Learning Rate = {self.optimizer.param_groups[0]['lr']}")
            # Test the model on the validation set at the end of each epoch
            self.test_model(epoch)
            self.verify(epoch, paired=True)
            self.save_model()

    def save_color_feature_map(self, A, image, enhanced_image, img_id, output_folder):
        """将特征图A转换为彩色图像并保存，显示原始图像、R、G、B及增强结果"""
        A = A[0].detach().cpu().numpy()  # 取 batch 里的第一张

        # 提取R、G、B通道
        R = A[0]  # 红色通道
        G = A[1] if A.shape[0] > 1 else A[0]  # 绿色通道
        B = A[2] if A.shape[0] > 2 else A[0]  # 蓝色通道

        # 归一化到 0-1 范围
        def normalize(img):
            return (img - img.min()) / (img.max() - img.min() + 1e-8)

        R, G, B = map(normalize, [R, G, B])

        # 创建一个网格布局
        fig = plt.figure(figsize=(20, 5))
        gs = GridSpec(1, 5, width_ratios=[1,1.08,1.08,1.08,1], height_ratios=[1])  # 自定义宽度比例，高度比例为 1

        # 原始图像
        ax0 = fig.add_subplot(gs[0])
        ax0.imshow(image[0].permute(1, 2, 0).cpu().numpy())
        # ax0.set_title("(a) Input Image", pad=10)
        ax0.axis('off')
        plt.text(0.5, -0.05, "(a) Input", transform=plt.gca().transAxes, fontsize=18, verticalalignment='top',horizontalalignment='center')

        # R、G、B通道
        channels = [R, G, B]
        titles = [r"(b) $A^R$", r"(c) $A^G$", r"(d) $A^B$"]
        for i, (channel, title) in enumerate(zip(channels, titles)):
            ax = fig.add_subplot(gs[i + 1])
            im = ax.imshow(channel, cmap='jet', vmin=0, vmax=1)
            # ax.set_title(title, pad=10)
            ax.axis('off')
            fig.colorbar(im, ax=ax, fraction=0.0455, pad=0.0345)
            plt.text(0.5, -0.05, title, transform=plt.gca().transAxes, fontsize=18, verticalalignment='top',horizontalalignment='center')

            
        # 增强后图像
        ax4 = fig.add_subplot(gs[4])
        ax4.imshow(enhanced_image[0].permute(1, 2, 0).cpu().numpy())
        # ax4.set_title("(e) Enhanced Image", pad=10)
        ax4.axis('off')
        plt.text(0.5, -0.05, "(e) Output", transform=plt.gca().transAxes, fontsize=18, verticalalignment='top',horizontalalignment='center')

        # 保存
        feature_map_path = os.path.join(output_folder, f"feature_{os.path.basename(img_id[0])}")
        plt.savefig(feature_map_path, bbox_inches='tight', pad_inches=0.1, dpi=300)
        plt.close()

    def test_model(self, epoch):
        self.DCE_net.eval()
        output_folder = os.path.join(self.config.output_folder)
        feature_folder = os.path.join(self.config.feature_folder)
        os.makedirs(output_folder, exist_ok=True)
        
        printed = False  # 只打印一次参数量和 FLOPs

        with torch.no_grad():
            for i, (image, img_id) in enumerate(self.val_loader):
                image = image.cuda()
                
                # 只在第一张图像时输出模型参数和 FLOPs
                if not printed:
                    dummy_input = image.clone()
                    print_model_profile(self.DCE_net, dummy_input)
                    printed = True

                enhanced_image, A = self.DCE_net(image)

                result_path = os.path.join(output_folder, os.path.basename(img_id[0]))
                save_image(enhanced_image, result_path)
        print(f"Saved enhanced images for epoch {epoch + 1}.")

        
    def verify(self, epoch, paired=True):
        result_file = os.path.join("eval.txt")
        dirB = os.path.join(self.config.eval_dir)
        dirA = os.path.join(self.config.output_folder) 

        if paired:
            # 使用 measure_dirs 计算 PSNR、SSIM 和 LPIPS
            psnr, ssim, lpips, niqe, brisque, piqe= self.evaluation.measure_dirs(dirA, dirB)
            result_str = (f"Epoch: {epoch+1}:PSNR: {psnr:.3f}, SSIM: {ssim:.3f}, LPIPS: {lpips:.3f}, "
                            f"NIQE: {niqe:.3f}, BRISQUE: {brisque:.3f}, PIQE: {piqe:.3f}\n")
            
            # 保存最佳模型
            if psnr > self.best_psnr:
                self.best_psnr = psnr
                self.save_model(best="psnr")
            if ssim > self.best_ssim:
                self.best_ssim = ssim
                self.save_model(best="ssim")
            if piqe < self.best_piqe:  # piqe 越低越好
                self.best_piqe = piqe
                self.save_model(best="piqe")
            if niqe < self.best_niqe:  # NIQE 越低越好
                self.best_niqe = niqe
                self.save_model(best="niqe")
            if brisque < self.best_brisque:  # BRISQUE 越低越好
                self.best_brisque = brisque
                self.save_model(best="brisque")

        print(result_str)
        # 将结果写入文件
        with open(result_file, "a") as f:
            f.write(result_str)

    def save_model(self, best=None):
        if best:
            filename = f"best_{best}_model.pth"
        else:
            filename = "model_last.pth"
        torch.save(self.DCE_net.state_dict(), os.path.join(self.config.snapshots_folder, filename))
        print(f"{filename} has been saved.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Input Parameters
    parser.add_argument('--train_dir', type=str, default="../Mydata/Coal_Train")
    # parser.add_argument('--val_dir', type=str, default="../Mydata/SICE_test/Lowlight_img")
    # parser.add_argument('--eval_dir', type=str, default="../Mydata/SICE_test/Lowlight_img_Label")
    # parser.add_argument('--val_dir', type=str, default="../data/data/LOL_v2/Real_captured/Test/low")
    # # parser.add_argument('--eval_dir', type=str, default="../data/data/LOL_v2/Real_captured/Test/high")
    # parser.add_argument('--val_dir', type=str, default="../Mydata/LOL_test/low")
    # parser.add_argument('--eval_dir', type=str, default="../Mydata/LOL_test/high")
    parser.add_argument('--val_dir', type=str, default="../Mydata/DSOD_test")
    parser.add_argument('--eval_dir', type=str, default="../Mydata/DSOD_test")
    parser.add_argument('--patch_size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--weight_decay', type=float, default=0.0001)
    parser.add_argument('--grad_clip_norm', type=float, default=1.0)
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--display_iter', type=int, default=10)
    parser.add_argument('--snapshot_iter', type=int, default=50)
    parser.add_argument('--snapshots_folder', type=str, default="ckpt/test/")
    parser.add_argument('--output_folder', type=str, default="train_result/DSOD-test/")
    parser.add_argument('--feature_folder', type=str, default="feature_result/")
    parser.add_argument('--load_pretrain', type=bool, default=False)
    parser.add_argument('--pretrain_dir', type=str, default="ckpt/Epoch99.pth")

    config = parser.parse_args()

    trainer = Trainer(config)
    trainer.train()
