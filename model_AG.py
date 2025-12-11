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
        loss = nn.L1Loss()
        return loss(pred_spec, target_spec)
        
    def forward(self, target_vocal, target_mix, mask):
        pred_vocal = mask * target_mix
        pred_accomp = (1 - mask) * target_mix
        target_accomp = torch.clamp(target_mix - target_vocal, min=0.0)
        
        loss_v = self.absoluteError(pred_vocal, target_vocal)
        loss_a = self.absoluteError(pred_accomp, target_accomp)
        loss = loss_v + loss_a
        
        if self.reduction == 'mean': return loss.mean()
        elif self.reduction == 'sum': return loss.sum()
        return loss

class AttentionGate(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        """
        Attention Gate
        :param F_g: Gating Signal 的通道數 (來自 Decoder)
        :param F_l: Skip Connection 的通道數 (來自 Encoder)
        :param F_int: 中間層的通道數 (通常設為 F_l / 2)
        """
        super(AttentionGate, self).__init__()
        
        # 對 Gating Signal 進行卷積 (W_g)
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        
        # 對 Skip Connection 進行卷積 (W_x)
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        # 產生 Attention Coefficients (Psi)
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, g, x):
        # g: Gating Signal (Decoder)
        # x: Skip Connection (Encoder)
        
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        
        return x * psi

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
        self.conv3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=(5, 5), stride=(2, 2), padding=2),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(inplace=True, negative_slope=0.2)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=(5, 5), stride=(2, 2), padding=2),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(inplace=True, negative_slope=0.2)
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=(5, 5), stride=(2, 2), padding=2),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(inplace=True, negative_slope=0.2)
        )
        self.conv6 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=(5, 5), stride=(2, 2), padding=2),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(inplace=True, negative_slope=0.2)
        )
        
        # Deconv layers
        self.deconv1 = nn.ConvTranspose2d(512, 256, kernel_size=(5, 5), stride=(2, 2), padding=2)
        self.deconv1_BAD = nn.Sequential(
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.Dropout2d(0.5)
        )
        self.att1 = AttentionGate(F_g=256, F_l=256, F_int=128)
        
        self.deconv2 = nn.ConvTranspose2d(512, 128, kernel_size=(5, 5), stride=(2, 2), padding=2)
        self.deconv2_BAD = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.Dropout2d(0.5)
        )
        self.att2 = AttentionGate(F_g=128, F_l=128, F_int=64)
        
        self.deconv3 = nn.ConvTranspose2d(256, 64, kernel_size=(5, 5), stride=(2, 2), padding=2)
        self.deconv3_BAD = nn.Sequential(
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.Dropout2d(0.5)
        )
        self.att3 = AttentionGate(F_g=64, F_l=64, F_int=32)
        
        self.deconv4 = nn.ConvTranspose2d(128, 32, kernel_size=(5, 5), stride=(2, 2), padding=2)
        self.deconv4_BAD = nn.Sequential(
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.Dropout2d(0.5)
        )
        self.att4 = AttentionGate(F_g=32, F_l=32, F_int=16)
        
        self.deconv5 = nn.ConvTranspose2d(64, 16, kernel_size=(5, 5), stride=(2, 2), padding=2)
        self.deconv5_BAD = nn.Sequential(
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            nn.Dropout2d(0.5)
        )
        self.att5 = AttentionGate(F_g=16, F_l=16, F_int=8)
        
        self.deconv6 = nn.ConvTranspose2d(32, 1, kernel_size=(5, 5), stride=(2, 2), padding=2)
        
        self.optim = torch.optim.Adam(self.parameters(), lr=5e-4)
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
        conv3_out = self.conv3(conv2_out)
        conv4_out = self.conv4(conv3_out)
        conv5_out = self.conv5(conv4_out)
        conv6_out = self.conv6(conv5_out)
        
        deconv1_out = self.deconv1(conv6_out, output_size=conv5_out.size())
        deconv1_out = self.deconv1_BAD(deconv1_out)
        conv5_att = self.att1(g=deconv1_out, x=conv5_out)
        
        deconv2_out = self.deconv2(torch.cat([deconv1_out, conv5_att], 1), output_size=conv4_out.size())
        deconv2_out = self.deconv2_BAD(deconv2_out)
        conv4_att = self.att1(g=deconv2_out, x=conv4_out)
        
        deconv3_out = self.deconv3(torch.cat([deconv2_out, conv4_att], 1), output_size=conv3_out.size())
        deconv3_out = self.deconv3_BAD(deconv3_out)
        conv3_att = self.att1(g=deconv3_out, x=conv3_out)
        
        deconv4_out = self.deconv4(torch.cat([deconv3_out, conv3_att], 1), output_size=conv2_out.size())
        deconv4_out = self.deconv4_BAD(deconv4_out)
        conv2_att = self.att1(g=deconv4_out, x=conv2_out)
        
        deconv5_out = self.deconv5(torch.cat([deconv4_out, conv2_att], 1), output_size=conv1_out.size())
        deconv5_out = self.deconv5_BAD(deconv5_out)
        conv1_att = self.att1(g=deconv5_out, x=conv1_out)
        
        deconv6_out = self.deconv6(torch.cat([deconv5_out, conv1_att], 1), output_size=mix.size())
        
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
        
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
        self.optim.step()
        return loss.item()