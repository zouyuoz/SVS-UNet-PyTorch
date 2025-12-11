import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import torch
import os
import auraloss

"""
    This script define the structure and update schema of U-Net

    @Reference: https://github.com/Jeongseungwoo/Singing-Voice-Separation
    @Revise: SunnerLi
"""

class WeightedL1Loss(nn.Module):
    def __init__(self, reduction='mean'):
        super().__init__()
        self.reduction = reduction
        
    def absoluteError(self, pred_spec, target_spec):
        return torch.abs(pred_spec - target_spec)
        
    def forward(self, target_vocal, target_mix, mask):
        pred_vocal = mask * target_mix
        pred_accomp = (1 - mask) * target_mix
        target_accomp = torch.clamp(target_mix - target_vocal, min=0.0)
        
        loss_v = self.absoluteError(pred_vocal, target_vocal)
        loss_a = self.absoluteError(pred_accomp, target_accomp)
        loss = loss_v + loss_a
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        
        return loss

class SelfAttention(nn.Module):
    """ 
    Self attention Layer optimized with torch.nn.functional.scaled_dot_product_attention
    """
    def __init__(self, in_dim):
        super(SelfAttention, self).__init__()
        self.chanel_in = in_dim
        
        # 根據論文架構，Query 和 Key 的通道數縮減為原本的 1/8
        self.head_dim = in_dim // 8
        
        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=self.head_dim, kernel_size=1)
        self.key_conv   = nn.Conv2d(in_channels=in_dim, out_channels=self.head_dim, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        """
        inputs : x (B, C, W, H)
        """
        m_batchsize, C, width, height = x.size()
        N = width * height
        
        # 1. Projections & Reshape
        # 我們將 (W, H) 展平成 N
        # Permute 調整為 (Batch, N, Channel) 以符合 Attention 輸入格式
        q = self.query_conv(x).view(m_batchsize, -1, N).permute(0, 2, 1) # (B, N, C/8)
        k = self.key_conv(x).view(m_batchsize, -1, N).permute(0, 2, 1)   # (B, N, C/8)
        v = self.value_conv(x).view(m_batchsize, -1, N).permute(0, 2, 1) # (B, N, C)

        # 2. 增加 Head 維度 (Batch, Head, SeqLen, Dim)
        # 雖然論文是單頭架構，但為了使用 SDPA 加速，我們將其視為 1 個 Head
        q = q.unsqueeze(1) # (B, 1, N, C/8)
        k = k.unsqueeze(1) # (B, 1, N, C/8)
        v = v.unsqueeze(1) # (B, 1, N, C)

        # 3. 執行 PyTorch 內建加速 Attention
        # 這行程式碼會自動啟用 FlashAttention (若硬體支援) 或其他優化核心
        # 預設會進行 Scale 操作 (除以 sqrt(dim))，這對訓練穩定性很有幫助
        attn_out = F.scaled_dot_product_attention(q, k, v)
        
        # 4. 還原形狀
        # (B, 1, N, C) -> (B, N, C) -> (B, C, N) -> (B, C, W, H)
        out = attn_out.squeeze(1).permute(0, 2, 1).view(m_batchsize, C, width, height)
        
        # 5. Residual Connection
        out = self.gamma * out + x
        return out

class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()

        # Define the network components
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=(5, 5), stride=(2, 2), padding=2),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(inplace=True, negative_slope=0.2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=(5, 5), stride=(2, 2), padding=2),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(inplace=True, negative_slope=0.2)
        )
        self.attn_enc2 = SelfAttention(32)
        self.conv3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=(5, 5), stride=(2, 2), padding=2),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(inplace=True, negative_slope=0.2)
        )
        self.attn_enc3 = SelfAttention(64)
        self.conv4 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=(5, 5), stride=(2, 2), padding=2),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(inplace=True, negative_slope=0.2)
        )
        self.attn_enc4 = SelfAttention(128)
        self.conv5 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=(5, 5), stride=(2, 2), padding=2),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(inplace=True, negative_slope=0.2)
        )
        self.attn_enc5 = SelfAttention(256)
        self.conv6 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=(5, 5), stride=(2, 2), padding=2),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(inplace=True, negative_slope=0.2)
        )
        self.attn_enc6 = SelfAttention(512)
        
        # Deconv layers
        self.deconv1 = nn.ConvTranspose2d(512, 256, kernel_size=(5, 5), stride=(2, 2), padding=2)
        self.deconv1_BAD = nn.Sequential(
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.Dropout2d(0.5)
        )
        self.attn_dec1 = SelfAttention(256)
        self.deconv2 = nn.ConvTranspose2d(512, 128, kernel_size=(5, 5), stride=(2, 2), padding=2)
        self.deconv2_BAD = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.Dropout2d(0.5)
        )
        self.attn_dec2 = SelfAttention(128)
        self.deconv3 = nn.ConvTranspose2d(256, 64, kernel_size=(5, 5), stride=(2, 2), padding=2)
        self.deconv3_BAD = nn.Sequential(
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.Dropout2d(0.5)
        )
        self.attn_dec3 = SelfAttention(64)
        self.deconv4 = nn.ConvTranspose2d(128, 32, kernel_size=(5, 5), stride=(2, 2), padding=2)
        self.deconv4_BAD = nn.Sequential(
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.Dropout2d(0.5)
        )
        self.attn_dec4 = SelfAttention(32)
        self.deconv5 = nn.ConvTranspose2d(64, 16, kernel_size=(5, 5), stride=(2, 2), padding=2)
        self.deconv5_BAD = nn.Sequential(
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            nn.Dropout2d(0.5)
        )
        self.deconv6 = nn.ConvTranspose2d(32, 1, kernel_size=(5, 5), stride=(2, 2), padding=2)
        
        self.optim = torch.optim.Adam(self.parameters(), lr=5e-3)
        self.crit = nn.L1Loss()

    # ==============================================================================
    #   Set & Get
    # ==============================================================================
    def forward(self, mix):
        """
            Generate the mask for the given mixture audio spectrogram

            Arg:    mix     (torch.Tensor)  - The mixture spectrogram which size is (B, 1, 512, 128)
            Ret:    The soft mask which size is (B, 1, 512, 128)
        """
        conv1_out = self.conv1(mix)
        
        conv2_out = self.conv2(conv1_out)
        conv2_out = self.attn_enc2(conv2_out)
        
        conv3_out = self.conv3(conv2_out)
        conv3_out = self.attn_enc3(conv3_out)
        
        conv4_out = self.conv4(conv3_out)
        conv4_out = self.attn_enc4(conv4_out)
        
        conv5_out = self.conv5(conv4_out)
        conv5_out = self.attn_enc5(conv5_out)
        
        conv6_out = self.conv6(conv5_out)
        conv6_out = self.attn_enc6(conv6_out)
        
        deconv1_out = self.deconv1(conv6_out, output_size=conv5_out.size())
        deconv1_out = self.deconv1_BAD(deconv1_out)
        deconv1_out = self.attn_dec1(deconv1_out)
        
        deconv2_out = self.deconv2(torch.cat([deconv1_out, conv5_out], 1), output_size=conv4_out.size())
        deconv2_out = self.deconv2_BAD(deconv2_out)
        deconv2_out = self.attn_dec2(deconv2_out)
        
        deconv3_out = self.deconv3(torch.cat([deconv2_out, conv4_out], 1), output_size=conv3_out.size())
        deconv3_out = self.deconv3_BAD(deconv3_out)
        deconv3_out = self.attn_dec3(deconv3_out)
        
        deconv4_out = self.deconv4(torch.cat([deconv3_out, conv3_out], 1), output_size=conv2_out.size())
        deconv4_out = self.deconv4_BAD(deconv4_out)
        deconv4_out = self.attn_dec4(deconv4_out)
        
        deconv5_out = self.deconv5(torch.cat([deconv4_out, conv2_out], 1), output_size=conv1_out.size())
        deconv5_out = self.deconv5_BAD(deconv5_out)
        
        deconv6_out = self.deconv6(torch.cat([deconv5_out, conv1_out], 1), output_size=mix.size())
        
        out = torch.sigmoid(deconv6_out)
        return out

    def backward(self, mix, voc):
        """
            Update the parameters for the given mixture spectrogram and the pure vocal spectrogram

            Arg:    mix     (torch.Tensor)  - The mixture spectrogram which size is (B, 1, 512, 128)
                    voc     (torch.Tensor)  - The pure vocal spectrogram which size is (B, 1, 512, 128)
        """
        self.optim.zero_grad()
        mask = self.forward(mix)
        loss = self.crit(mask * mix, voc)
        loss.backward()
        self.optim.step()
        return loss.item()