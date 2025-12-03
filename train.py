from model import UNet
import torch.utils.data as Data
import numpy as np
import argparse
import torch
import os
from tqdm import tqdm
import random

"""
    This script defines the training procedure of SVS-UNet
    Updated for Python 3.9+
"""

# Determine device
device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

class SpectrogramDataset(Data.Dataset):
    def __init__(self, path):
        self.path = path
        self.mixture_path = os.path.join(path, 'mixture')
        self.vocal_path = os.path.join(path, 'vocal')
        self.files = sorted([f for f in os.listdir(self.mixture_path) if 'spec' in f])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file_name = self.files[idx]
        
        # Load the spectrogram
        mix = np.load(os.path.join(self.mixture_path, file_name))
        voc = np.load(os.path.join(self.vocal_path, file_name))

        # Random crop
        # Ensure we have enough width
        if mix.shape[-1] > 129:
            start = random.randint(0, mix.shape[-1] - 128 - 1)
            mix = mix[1:, start:start + 128] # Crop frequency bins if needed, original code mix[1:]
            voc = voc[1:, start:start + 128]
        else:
            # Padding if too short (optional, based on your data)
            pass

        # Add channel dimension
        mix = mix[np.newaxis, :, :]
        voc = voc[np.newaxis, :, :]
        
        mix = np.asarray(mix, dtype=np.float32)
        voc = np.asarray(voc, dtype=np.float32)

        # To tensor (Pytorch expects C, H, W)
        # Original code did permute(2,0,1) which implies input was (Freq, Time, 1)?
        # If np.load returns (Freq, Time), adding newaxis at end makes (Freq, Time, 1).
        # Permute(2, 0, 1) -> (1, Freq, Time). Correct.
        # But here I did newaxis at 0. So shape is (1, Freq, Time). No permute needed.
        # Let's stick to original logic strictly:
        
        # Reload to be sure
        mix_raw = np.load(os.path.join(self.mixture_path, file_name))
        voc_raw = np.load(os.path.join(self.vocal_path, file_name))
        
        if mix_raw.shape[-1] > 129:
            start = random.randint(0, mix_raw.shape[-1] - 128 - 1)
            mix_crop = mix_raw[1:, start:start + 128, np.newaxis] # (Freq, Time, 1)
            voc_crop = voc_raw[1:, start:start + 128, np.newaxis]
        else:
            # Fallback for short files
            mix_crop = mix_raw[1:, :, np.newaxis]
            voc_crop = voc_raw[1:, :, np.newaxis]

        mix_tensor = torch.from_numpy(mix_crop).permute(2, 0, 1) # (1, Freq, Time)
        voc_tensor = torch.from_numpy(voc_crop).permute(2, 0, 1)
        
        return mix_tensor, voc_tensor

# =========================================================================================
# 1. Parse the direction and related parameters
# =========================================================================================
"""
                                    Parameter Explain
    --------------------------------------------------------------------------------------------
        --train_folder      The root of the training folder. You can generate via data.py
        --load_path         The path of pre-trained model
        --save_path         The model path you want to save 
        --epoch             How many epoch you want to train
    --------------------------------------------------------------------------------------------
"""
parser = argparse.ArgumentParser()
parser.add_argument('--train_folder', type = str, default = './data/vocals')
parser.add_argument('--load_path'   , type = str, default = 'result.pth')
parser.add_argument('--save_path'   , type = str, default = 'result.pth')
parser.add_argument('--epoch'       , type = int, default = 2)
args = parser.parse_args()

# =========================================================================================
# 2. Training
# =========================================================================================
# Create the data loader
loader = Data.DataLoader(
    dataset=SpectrogramDataset(args.train_folder),
    batch_size=4, # Increased batch size slightly
    num_workers=4, 
    shuffle=True,
    pin_memory=True if device.type == 'cuda' else False
)

model = UNet()
model.to(device)

if os.path.exists(args.load_path):
    model.load(args.load_path)

print("Start training...")
for ep in range(args.epoch):
    model.train()
    loop = tqdm(loader, desc=f"Epoch {ep+1}/{args.epoch}")
    
    epoch_loss = 0
    for i, (mix, voc) in enumerate(loop):
        mix, voc = mix.to(device), voc.to(device)
        
        # model.backward handles zero_grad, forward, loss, backward, step
        model.backward(mix, voc)
        
        # Get loss for display
        loss_dict = model.getLoss()
        current_loss = loss_dict.get('loss_list_vocal', 0)
        epoch_loss += current_loss
        
        loop.set_postfix(loss=current_loss)
    
    print(f"Epoch {ep+1} Average Loss: {epoch_loss / len(loader):.6f}")
    model.save(args.save_path)

print("Finish training!")