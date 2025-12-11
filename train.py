# from model import UNet
from model_AG import UNet
import torch.utils.data as Data
import numpy as np
import argparse
import torch
import os
from tqdm import tqdm
import random
from utils import *
from sampleLoader import SpectrogramDataset

"""
    SVS-UNet Training Script with Validation
    Updated for Python 3.10+ & PyTorch 2.x
"""

# Determine device
device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

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
parser.add_argument('--train_folder', type = str, default = 'unet_spectrograms/train')
parser.add_argument('--valid_folder', type = str, default = 'unet_spectrograms/valid')
parser.add_argument('--label'       , type = str, required = True)
parser.add_argument('--batch_size'  , type = int, default = 8)
parser.add_argument('--epoch'       , type = int, default = 10)
parser.add_argument('--val_interval', type = int, default = 20)
parser.add_argument('--load_path'   , type = str)

args = parser.parse_args()

log_file    = f'LOG/log_{args.label}.txt'
best_weight = f'CKPT/svs_{args.label}_best.pth'
ckpt_weight = f'CKPT/svs_{args.label}.pth'

# =========================================================================================
# 2. Training Setup
# =========================================================================================
WAV_ROOT = '.MUSDB18/' 

# 替換 Train Loader
train_loader = Data.DataLoader(
    dataset = SpectrogramDataset(path=args.train_folder, samples_per_song=SAMPLES_PER_SONG, augment=True),
    batch_size=args.batch_size, num_workers=8, shuffle=True, pin_memory=True
)

# 替換 Valid Loader
valid_loader = Data.DataLoader(
    dataset = SpectrogramDataset(path=args.valid_folder, samples_per_song=SAMPLES_PER_SONG),
    batch_size=args.batch_size, num_workers=4, shuffle=False, pin_memory=True
)

model = UNet()
model.to(device)

best_val_loss = 100.
train_loss_history = []
valid_loss_history = []
scheduler = None
start_epoch = 0

# =========================================================================================
# Load Checkpoint
# =========================================================================================
if os.path.exists(args.load_path):
    print(f"Loading checkpoint from {args.load_path}")
    checkpoint = torch.load(args.load_path, map_location=device)
    
    # 1. 載入模型權重
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # 2. 載入優化器狀態
    if 'optim' in checkpoint:
        model.optim.load_state_dict(checkpoint['optim'])
        
    # 3. 載入 Epoch (接續訓練關鍵)
    start_epoch = checkpoint.get('epoch', 0)
    
    # 4. 載入 Scheduler (如果有)
    if scheduler is not None and 'scheduler' in checkpoint and checkpoint['scheduler'] is not None:
        scheduler.load_state_dict(checkpoint['scheduler'])
        
    # 5. [修改] 載入 Loss History & 恢復 Best Loss
    if 'train_loss_history' in checkpoint:
        train_loss_history = checkpoint['train_loss_history']
        
    if 'valid_loss_history' in checkpoint:
        valid_loss_history = checkpoint['valid_loss_history']
        # [關鍵] 從歷史紀錄中恢復最佳 loss，否則會被重置為 100
        if len(valid_loss_history) > 0:
            best_val_loss = min(valid_loss_history)
            print(f"Restored best_val_loss: {best_val_loss:.6f}")

# =========================================================================================
# 3. Main Loop
# =========================================================================================

print(f"Start training for {args.epoch - start_epoch} epochs...")

for ep in range(start_epoch, args.epoch):
    # --- Training Phase ---
    model.train()
    loop = tqdm(train_loader, desc=f"Epoch {ep+1}/{args.epoch} [Train]", leave=False)
    train_loss_sum = 0
    
    # if ep == 401 or ep == 901:
    #     new_lr = 5e-4 if ep == 401 else 1e-4
    #     for param_group in model.optim.param_groups:
    #         param_group['lr'] = new_lr
    #     torch.save(checkpoint, f'CKPT/svs_{args.label}_{ep}.pth')
    #     print(f"\n[Info] Epoch {ep}: Learning rate manually changed to {new_lr}!\n")
    
    for i, (mix, voc) in enumerate(loop):
        
        mix, voc = mix.to(device), voc.to(device)
        current_loss = model.backward(mix, voc)
        train_loss_sum += current_loss
        
        loop.set_postfix(loss=current_loss)
    
    avg_train_loss = train_loss_sum / len(train_loader)
    train_loss_history.append(avg_train_loss)
    
    # --- Validation Phase (Every 10 epochs) ---
    avg_val_loss = 1000.
    if valid_loader and (ep + 1) % args.val_interval == 0:
        model.eval() # 切換到評估模式
        val_loss_sum = 0
        
        # 不計算梯度
        with torch.no_grad():
            val_loop = tqdm(valid_loader, desc=f"Epoch {ep+1} [Valid]", leave=False)
            
            for mix, voc in val_loop:
                mix, voc = mix.to(device), voc.to(device)
                
                # 計算 Loss
                mask = model(mix)
                loss = model.crit(mix * mask, voc)
                val_loss_sum += loss.item()
        
        avg_val_loss = val_loss_sum / len(valid_loader)
        valid_loss_history.append(avg_val_loss)

        # 儲存 Checkpoint
        checkpoint = {
            'epoch': ep + 1,
            'model_state_dict': model.state_dict(),
            'optim': model.optim.state_dict(),
            'scheduler': scheduler.state_dict() if scheduler is not None else None,
            'train_loss_history': train_loss_history,
            'valid_loss_history': valid_loss_history
        }
        torch.save(checkpoint, ckpt_weight)
        
        is_best_val_loss = ''
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            is_best_val_loss = '*'
            torch.save(checkpoint, best_weight)
            
        print(f"\n[Epoch {ep+1}] Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f} " + is_best_val_loss)
        
    else: print(f"Epoch {ep+1} Avg Loss: {avg_train_loss:.6f}")

print("Finish training!")

"""
1208 midnight:
python train.py \
    --train_folder unet_spectrograms/train \
    --valid_folder unet_spectrograms/valid \
    --label ag_mid_aug \
    --batch_size 32 \
    --epoch 500 \
    --val_interval 10 \
    --load_path CKPT/svs_attn_gate.pth
    
想法：

"""