import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import torch
import os

"""
    This script define the structure and update schema of U-Net

    @Reference: https://github.com/Jeongseungwoo/Singing-Voice-Separation
    @Revise: SunnerLi
"""

class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()

        # Define the network components
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=(5, 5), stride=(2, 2), padding=2),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=(5, 5), stride=(2, 2), padding=2),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(True)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=(5, 5), stride=(2, 2), padding=2),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(True)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=(5, 5), stride=(2, 2), padding=2),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(True)
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=(5, 5), stride=(2, 2), padding=2),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(True)
        )
        self.conv6 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=(5, 5), stride=(2, 2), padding=2),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(True)
        )
        
        # Deconv layers
        self.deconv1 = nn.ConvTranspose2d(512, 256, kernel_size=(5, 5), stride=(2, 2), padding=2)
        self.deconv1_BAD = nn.Sequential(
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.Dropout2d(0.5)
        )
        self.deconv2 = nn.ConvTranspose2d(512, 128, kernel_size=(5, 5), stride=(2, 2), padding=2)
        self.deconv2_BAD = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.Dropout2d(0.5)
        )
        self.deconv3 = nn.ConvTranspose2d(256, 64, kernel_size=(5, 5), stride=(2, 2), padding=2)
        self.deconv3_BAD = nn.Sequential(
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.Dropout2d(0.5)
        )
        self.deconv4 = nn.ConvTranspose2d(128, 32, kernel_size=(5, 5), stride=(2, 2), padding=2)
        self.deconv4_BAD = nn.Sequential(
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.Dropout2d(0.5)
        )
        self.deconv5 = nn.ConvTranspose2d(64, 16, kernel_size=(5, 5), stride=(2, 2), padding=2)
        self.deconv5_BAD = nn.Sequential(
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            nn.Dropout2d(0.5)
        )
        self.deconv6 = nn.ConvTranspose2d(32, 1, kernel_size=(5, 5), stride=(2, 2), padding=2)

        # Define loss list
        self.loss_list_vocal = []
        
        # Define the criterion and optimizer
        self.optim = torch.optim.Adam(self.parameters(), lr=1e-3)
        # self.crit = nn.L1Loss()
        self.crit = nn.SmoothL1Loss()
        
        # We handle device movement externally or via .to(device)

    # ==============================================================================
    #   IO
    # ==============================================================================
    def load(self, path):
        if os.path.exists(path):
            print("Load the pre-trained model from {}".format(path))
            # map_location='cpu' ensures we can load even without CUDA
            state = torch.load(path, map_location='cpu') 
            
            # Load simple attributes
            for (key, obj) in state.items():
                if 'loss_list' in key: # Fix for string matching
                    setattr(self, key, obj)
            
            # Load weights
            self.load_state_dict(state['model_state_dict'], strict=False)
            if 'optim' in state:
                self.optim.load_state_dict(state['optim'])
        else:
            print("Pre-trained model {} is not exist...".format(path))

    def save(self, path):
        # Record the parameters
        # Use standard state_dict saving which is cleaner than saving individual layers
        state = {
            'model_state_dict': self.state_dict(),
            'optim': self.optim.state_dict(),
        }

        # Record the loss history
        for key in self.__dict__:
            if 'loss_list' in key:
                state[key] = getattr(self, key)
        torch.save(state, path)

    # ==============================================================================
    #   Set & Get
    # ==============================================================================
    def getLoss(self, normalize=False):
        loss_dict = {}
        for key in self.__dict__:
            if 'loss_list' in key:
                val = getattr(self, key)
                if len(val) > 0:
                    if not normalize:
                        loss_dict[key] = round(val[-1], 6)
                    else:
                        loss_dict[key] = np.mean(val)
        return loss_dict

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
        
        deconv2_out = self.deconv2(torch.cat([deconv1_out, conv5_out], 1), output_size=conv4_out.size())
        deconv2_out = self.deconv2_BAD(deconv2_out)
        
        deconv3_out = self.deconv3(torch.cat([deconv2_out, conv4_out], 1), output_size=conv3_out.size())
        deconv3_out = self.deconv3_BAD(deconv3_out)
        
        deconv4_out = self.deconv4(torch.cat([deconv3_out, conv3_out], 1), output_size=conv2_out.size())
        deconv4_out = self.deconv4_BAD(deconv4_out)
        
        deconv5_out = self.deconv5(torch.cat([deconv4_out, conv2_out], 1), output_size=conv1_out.size())
        deconv5_out = self.deconv5_BAD(deconv5_out)
        
        deconv6_out = self.deconv6(torch.cat([deconv5_out, conv1_out], 1), output_size=mix.size())
        
        out = torch.sigmoid(deconv6_out) # Updated from F.sigmoid
        return out

    def backward(self, mix, voc):
        """
            Update the parameters for the given mixture spectrogram and the pure vocal spectrogram

            Arg:    mix     (torch.Tensor)  - The mixture spectrogram which size is (B, 1, 512, 128)
                    voc     (torch.Tensor)  - The pure vocal spectrogram which size is (B, 1, 512, 128)
        """
        self.optim.zero_grad()
        msk = self.forward(mix)
        loss = self.crit(msk * mix, voc)
        self.loss_list_vocal.append(loss.item())
        loss.backward()
        self.optim.step()