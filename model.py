import torch
import torch.nn as nn
import torch.nn.functional as F


class DepthConv(nn.Module):
	def __init__(self, in_ch, out_ch):
		super(DepthConv, self).__init__()
		self.depth_conv = nn.Conv2d(in_channels=in_ch, out_channels=in_ch, kernel_size=3, padding=1, groups=in_ch)
		self.point_conv = nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=1)
	def forward(self, x):
		return self.point_conv(self.depth_conv(x))

class SpatialAttention(nn.Module):

	def __init__(self):
		super(SpatialAttention, self).__init__()
		self.conv = DepthConv(2,1)
		self.sigmoid = nn.Sigmoid()
 
	def forward(self, x):
		avg_out = torch.mean(x, dim=1, keepdim=True)
		max_out, _ = torch.max(x, dim=1, keepdim=True)
		x_out = torch.cat([avg_out, max_out], dim=1)
		out = self.sigmoid(self.conv(x_out))
		return out
	
class ChannelAttention(nn.Module):
	def __init__(self, channel, reduction=16):
		super(ChannelAttention, self).__init__()
		self.avg_pool = nn.AdaptiveAvgPool2d(1)
		self.fc = nn.Sequential(
			nn.Linear(channel, channel // reduction, bias=False),
			nn.ReLU(inplace=True),
			nn.Linear(channel // reduction, channel, bias=False),
			nn.Sigmoid()
		)

	def forward(self, x):
		b, c, _, _ = x.size()
		avg_out = self.avg_pool(x).view(b, c)
		y = self.fc(avg_out).view(b, c, 1, 1)  # 信息融合
		return y.expand_as(x)

class CSCA(nn.Module):
	def __init__(self, channel, reduction=16):
		super(CSCA, self).__init__()
		self.channel_attention = ChannelAttention(channel, reduction)
		self.spatial_attention = SpatialAttention()
		self.conv = DepthConv(2*channel, channel)

	def forward(self, x):
		# 第一次通道注意力
		ca1 = x * self.channel_attention(x)
		# 第一次空间注意力
		sa1 = ca1 * self.spatial_attention(ca1)
		# cs= torch.cat([ca1,sa1],dim=1)
		# 第二次通道注意力
		out = x * self.channel_attention(sa1)
		return out

class EnhanceNet(nn.Module):
	def __init__(self, channels=32):
		super(EnhanceNet, self).__init__()

		# 原始的卷积层
		self.conv1 = DepthConv(3, channels) 
		self.conv2 = DepthConv(channels, channels)
		self.conv3 = DepthConv(channels, channels)
		self.conv4 = DepthConv(channels, channels)
		# 新增的卷积层
		self.conv5 = DepthConv(channels*2, channels)  
		self.conv6 = DepthConv(channels*2, channels)  
		self.conv7 = DepthConv(channels*2, 3)        

		# Shuffle Attention和ReLU激活
		self.Attention = CSCA(channels)
		self.relu = nn.ReLU()
		# 多尺度池化用于光照调整
		self.M_LAC = Mutil_LAC()

	def feature_extract(self, x):

		# 特征提取：原始卷积层
		x_1 = self.conv1(x)
		x_1 = self.Attention(x_1)
		# x_1 = self.relu(x_1)
		x_2 = self.conv2(x_1)
		x_2 = self.Attention(x_2)
		# x_2 = self.relu(x_2)
		x_3 = self.conv3(x_2)
		x_3 = self.Attention(x_3)
		# x_3 = self.relu(x_3)
		x_4 = self.conv3(x_3)
		x_4 = self.Attention(x_4)
		# x_4 = self.relu(x_4)

		# 新的卷积层，进行特征融合
		x_5 = self.conv5(torch.cat([x_3, x_4], 1))  # 连接x_2和x_3的特征进行处理
		x_5 = self.relu(x_5)

		x_6 = self.conv6(torch.cat([x_2, x_5], 1))  # 连接x_4和x_5的特征进行处理
		x_6 = self.relu(x_6)

		x_7 = self.conv7(torch.cat([x_1, x_6], 1))  # 连接x_5和x_6的特征进行输出
		A = torch.tanh(x_7)  # 输出值在[-1, 1]之间
		return A
	
	def forward(self, x):
		# 初始化增强后的图像为输入图像
		A = self.feature_extract(x)
		enhanced_image = x
		# 使用LAC函数对图像进行递归增强
		for _ in range(8):	
			enhanced_image = self.M_LAC(enhanced_image, A)  

		return enhanced_image, A
# weights=[0.7,0.8,0.4,0.1]
class Mutil_LAC(nn.Module):
	def __init__(self, pool_sizes=[1,2,3,6], weights=[0.8,0.7,0.4,0.1]):
		super(Mutil_LAC, self).__init__()
		
		self.pool_sizes = pool_sizes
		self.weights = weights
		self.pools = nn.ModuleList([
			nn.AvgPool2d(kernel_size=size, stride=1, padding=size // 2)
			for size in self.pool_sizes
		])

	def forward(self, x, A):
		# lac = x + A * (torch.pow(x, 2) - x)
		lac = x
		_, _, h, w = x.size()

		for i, pool in enumerate(self.pools):
			pooled_x = pool(x)  # 执行池化操作
			xr = F.interpolate(pooled_x, size=(h, w), mode='bilinear', align_corners=False)
			lac_x = A*(1-torch.mean(xr))* (xr-torch.pow(xr, 2))
			# 根据给定的权重来加权不同尺度的特征图
			lac = lac + self.weights[i] * lac_x

		return lac

