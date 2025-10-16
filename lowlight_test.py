import torch
import torch.nn as nn
import torchvision
import torch.backends.cudnn as cudnn
import torch.optim
import os
import sys
import argparse
import time
import dataloader
import model
import numpy as np
from torchvision import transforms
from PIL import Image
import glob
import time
from thop import profile  

def print_model_profile(model, input_tensor):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"模型总参数量: {total_params:,}, 可训练参数量: {trainable_params:,}")

    flops, params = profile(model, inputs=(input_tensor,))
    print(f"FLOPs: {flops / 1e9:.4f} GFLOPs")  # 转换成 GigaFLOPs


def lowlight(test_image_folder, save_folder, ckpt_path):
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    total_start_time = time.time()
    with torch.no_grad():
        for file_name in os.listdir(test_image_folder):
            torch.cuda.empty_cache()  # 清空 GPU 缓存，确保记录准确
            start_time = time.time()
            if file_name.lower().endswith(('.jpg', '.png', '.jpeg', '.bmp')):
                image_path = os.path.join(test_image_folder, file_name)
                image = Image.open(image_path)
                image = (np.asarray(image) / 255.0)
                image = torch.from_numpy(image).float().permute(2, 0, 1).cuda().unsqueeze(0)
                DCE_net = model.EnhanceNet().cuda()
                DCE_net.load_state_dict(torch.load(ckpt_path))

                enhanced_image, _ = DCE_net(image)
                # 获取原始文件名
                file_name = os.path.basename(image_path)
                # 创建完整的保存路径
                result_path = os.path.join(save_folder, file_name)
                # 确保保存目录存在
                if not os.path.exists(save_folder):
                    os.makedirs(save_folder)
                # 保存处理后的图像
                torchvision.utils.save_image(enhanced_image, result_path)
                print(f"Image saved at {result_path}")

            # 记录单张图像推理的时间
            end_time = time.time()
            single_inference_time = end_time - start_time
            print(f"Processed {file_name }, inference time: {single_inference_time:.4f} seconds")

        # 获取 GPU 内存使用情况
        current_memory = torch.cuda.memory_allocated('cuda') / (1024 ** 2)  # 转换为 MB
        peak_memory = torch.cuda.max_memory_allocated('cuda') / (1024 ** 2)  # 转换为 MB
        
        print(f"Current GPU memory: {current_memory:.2f} MB, Peak GPU memory: {peak_memory:.2f} MB")
        
        # 记录总的测试时间
        total_end_time = time.time()
        total_inference_time = total_end_time - total_start_time
        print(f"Total testing time: {total_inference_time:.4f} seconds")

        print_model_profile(DCE_net, image)


if __name__ == '__main__':
        test_image_folder = "../Mydata/DSOD_640"
        save_folder = './results/DSOD'  # 你想保存图像的目标文件夹
        ckpt_path = 'ckpt/SICE/best_psnr_model.pth'
        lowlight(test_image_folder, save_folder, ckpt_path)


