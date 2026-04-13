import torch
import torch.nn as nn
import torch.nn.functional as F

class AttentionBlock(nn.Module):
    """
    Attention Block for focusing on relevant features
    """
    def __init__(self, F_g, F_l, F_int):
        super(AttentionBlock, self).__init__()
        
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi


class ResidualBlock(nn.Module):
    """
    Residual Block with two convolutions
    """
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        residual = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        out += self.shortcut(residual)
        out = self.relu(out)
        
        return out


class ASPP(nn.Module):
    """
    Atrous Spatial Pyramid Pooling (ASPP) module
    """
    def __init__(self, in_channels, out_channels):
        super(ASPP, self).__init__()
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=6, dilation=6, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=12, dilation=12, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=18, dilation=18, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        self.global_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        self.conv_out = nn.Sequential(
            nn.Conv2d(out_channels * 5, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        size = x.shape[2:]
        
        feat1 = self.conv1(x)
        feat2 = self.conv2(x)
        feat3 = self.conv3(x)
        feat4 = self.conv4(x)
        feat5 = F.interpolate(self.global_pool(x), size=size, mode='bilinear', align_corners=True)
        
        out = torch.cat([feat1, feat2, feat3, feat4, feat5], dim=1)
        out = self.conv_out(out)
        
        return out


class ResUNetPlusPlus(nn.Module):
    """
    ResUNet++ with Attention Mechanism for Steel Defect Detection
    """
    def __init__(self, in_channels=3, num_classes=4):
        super(ResUNetPlusPlus, self).__init__()
        
        # Encoder
        self.input_block = ResidualBlock(in_channels, 64)
        self.pool1 = nn.MaxPool2d(2)
        
        self.enc1 = ResidualBlock(64, 128)
        self.pool2 = nn.MaxPool2d(2)
        
        self.enc2 = ResidualBlock(128, 256)
        self.pool3 = nn.MaxPool2d(2)
        
        self.enc3 = ResidualBlock(256, 512)
        self.pool4 = nn.MaxPool2d(2)
        
        # Bridge with ASPP
        self.bridge = ASPP(512, 1024)
        
        # Decoder with Attention
        self.up1 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.att1 = AttentionBlock(F_g=512, F_l=512, F_int=256)
        self.dec1 = ResidualBlock(1024, 512)
        
        self.up2 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.att2 = AttentionBlock(F_g=256, F_l=256, F_int=128)
        self.dec2 = ResidualBlock(512, 256)
        
        self.up3 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.att3 = AttentionBlock(F_g=128, F_l=128, F_int=64)
        self.dec3 = ResidualBlock(256, 128)
        
        self.up4 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.att4 = AttentionBlock(F_g=64, F_l=64, F_int=32)
        self.dec4 = ResidualBlock(128, 64)
        
        # Output
        self.output = nn.Conv2d(64, num_classes, kernel_size=1)
    
    def forward(self, x):
        # Encoder
        x1 = self.input_block(x)  # 64 channels
        p1 = self.pool1(x1)
        
        x2 = self.enc1(p1)  # 128 channels
        p2 = self.pool2(x2)
        
        x3 = self.enc2(p2)  # 256 channels
        p3 = self.pool3(x3)
        
        x4 = self.enc3(p3)  # 512 channels
        p4 = self.pool4(x4)
        
        # Bridge
        bridge = self.bridge(p4)  # 1024 channels
        
        # Decoder
        d1 = self.up1(bridge)
        x4_att = self.att1(d1, x4)
        d1 = torch.cat([d1, x4_att], dim=1)
        d1 = self.dec1(d1)
        
        d2 = self.up2(d1)
        x3_att = self.att2(d2, x3)
        d2 = torch.cat([d2, x3_att], dim=1)
        d2 = self.dec2(d2)
        
        d3 = self.up3(d2)
        x2_att = self.att3(d3, x2)
        d3 = torch.cat([d3, x2_att], dim=1)
        d3 = self.dec3(d3)
        
        d4 = self.up4(d3)
        x1_att = self.att4(d4, x1)
        d4 = torch.cat([d4, x1_att], dim=1)
        d4 = self.dec4(d4)
        
        # Output
        output = self.output(d4)
        
        return output
