from model import UNet
import numpy as np
import argparse
import torch
import os
from tqdm import tqdm

def num2str(n):
    return str(n).zfill(4)

"""
    This script defines the inference procedure of SVS-UNet

    @Author: SunnerLi
"""
# =========================================================================================
# 1. Parse the direction and related parameters
# =========================================================================================
"""
                                    Parameter Explain
    --------------------------------------------------------------------------------------------
        --model_path        The path of pre-trained model
        --mixture_folder    The root of the testing folder. You can generate via data.py 
        --tar               The folder where you want to save the splited magnitude in
    --------------------------------------------------------------------------------------------
"""
parser = argparse.ArgumentParser()
parser.add_argument('--model_path', type=str, default='result.pth')
parser.add_argument('--mixture_folder', type=str, default='inference/mixture')
parser.add_argument('--tar', type=str, default='inference/split')
args = parser.parse_args()

if not os.path.exists(args.tar):
    os.makedirs(args.tar, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

model = UNet()
model.to(device)
model.load(args.model_path)
model.eval()

print(f"Inference using {device}...")

with torch.no_grad():
    files = sorted([f for f in os.listdir(args.mixture_folder) if 'spec' in f])
    bar = tqdm(files)
    
    for idx, name in enumerate(bar):
        # Optional: limit number of files for testing
        # if idx > 5: break
        
        filepath = os.path.join(args.mixture_folder, name)
        mix = np.load(filepath)
        
        # Prepare to reconstruct
        spec_sum = None
        
        # Sliding window inference
        # Original logic assumes simple concatenation.
        # We process in chunks of 128 time steps.
        
        num_segments = mix.shape[-1] // 128
        
        for i in range(num_segments):
            # Get segment
            seg = mix[1:, i * 128 : i * 128 + 128, np.newaxis] # (Freq, Time, 1)
            seg_tensor = torch.from_numpy(seg).permute(2, 0, 1).unsqueeze(0) # (B, C, F, T) -> (1, 1, F, T)
            seg_tensor = seg_tensor.to(device)

            # Generate mask
            msk = model(seg_tensor)

            # Apply mask
            # Vocal = Mix * (1 - Mask) based on original paper logic usually, 
            # but check original code: vocal_ = seg * (1 - msk)
            vocal_tensor = seg_tensor * (1 - msk)
            
            # Back to numpy
            # (1, 1, F, T) -> (F, T)
            vocal_np = vocal_tensor.squeeze().cpu().numpy()
            
            # Restore the first frequency bin if it was dropped (original code mix[1:])
            # Original code: vocal_ = np.vstack((np.zeros((128)), vocal_)) -> This adds a row of zeros at freq 0
            # Dimensions of vocal_np should be (512, 128)
            # vstack along axis 0
            vocal_np = np.vstack((np.zeros((1, 128), dtype=np.float32), vocal_np))
            
            spec_sum = vocal_np if spec_sum is None else np.concatenate((spec_sum, vocal_np), axis=1)
            
        if spec_sum is not None:
            save_name = f"{num2str(idx)}_{name.replace('_spec.npy', '')}_vocal_spec"
            np.save(os.path.join(args.tar, save_name), spec_sum)

print("Inference finished!")