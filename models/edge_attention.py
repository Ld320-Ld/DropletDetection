import torch
import torch.nn as nn
import torch.nn.functional as F

class ChannelAttention(nn.Module):
    def __init__(self, in_channels, ratio=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.fc1 = nn.Conv2d(in_channels, in_channels // ratio, 1)
        self.relu = nn.ReLU()
        self.fc2 = nn.Conv2d(in_channels // ratio, in_channels, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu(self.fc1(self.max_pool(x))))
        out = self.sigmoid(avg_out + max_out)
        return out

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_cat = torch.cat([avg_out, max_out], dim=1)
        out = self.conv(x_cat)
        return self.sigmoid(out)

class EdgeAttention(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv2 = nn.Conv2d(in_channels, 1, 1)
        self.sigmoid = nn.Sigmoid()
        
        # 可学习的边缘卷积核
        self.edge_conv_x = nn.Conv2d(1, 1, 3, padding=1, bias=False)
        self.edge_conv_y = nn.Conv2d(1, 1, 3, padding=1, bias=False)
        
        # 初始化为近似 Sobel 算子
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]).float().view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]]).float().view(1, 1, 3, 3)
        self.edge_conv_x.weight = nn.Parameter(sobel_x)
        self.edge_conv_y.weight = nn.Parameter(sobel_y)

    def forward(self, x):
        # 计算可学习边缘
        edge_x = self.edge_conv_x(x.mean(dim=1, keepdim=True))
        edge_y = self.edge_conv_y(x.mean(dim=1, keepdim=True))
        edge = torch.sqrt(edge_x**2 + edge_y**2)
        
        # 生成注意力权重
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.conv2(out)
        out = self.sigmoid(out)
        
        return out * edge

class CBAM(nn.Module):
    def __init__(self, in_channels, ratio=16, kernel_size=7):
        super().__init__()
        # 确保ratio不会导致通道数为0
        ratio = min(ratio, in_channels)
        if in_channels // ratio < 1:
            ratio = in_channels
            
        self.channel_att = ChannelAttention(in_channels, ratio)
        self.spatial_att = SpatialAttention(kernel_size)
        self.edge_att = EdgeAttention(in_channels)
        
        # 确保输出通道数与输入相同
        self.in_channels = in_channels
        self.out_channels = in_channels
        
        # 添加dropout来防止过拟合
        self.dropout = nn.Dropout2d(0.1)

    def forward(self, x):
        # 验证输入张量
        if x.size(1) != self.in_channels:
            raise ValueError(f"Expected input with {self.in_channels} channels, got {x.size(1)}")
            
        identity = x
        
        # 通道注意力
        x = x * self.channel_att(x)
        
        # 空间注意力  
        x = x * self.spatial_att(x)
        
        # 边缘注意力
        x = x * self.edge_att(x)
        
        # 应用dropout
        x = self.dropout(x)
        
        # 残差连接
        return x + identity