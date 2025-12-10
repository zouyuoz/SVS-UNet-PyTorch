from model import UNet, WeightedL1Loss
import torch.utils.data as Data
import numpy as np
import argparse
import torch
import os
from tqdm import tqdm
import random
from utils import *
from sampleLoader import DynamicDataset

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
# parser.add_argument('--train_folder', type = str, default = './data/vocals')
parser.add_argument('--load_path'   , type = str, default = 'result.pth')
parser.add_argument('--label'       , type = str, required = True)
parser.add_argument('--epoch'       , type = int, default = 2)
parser.add_argument('--batch_size'  , type = int, default = 2)

# parser.add_argument('--valid_folder', type = str, default = 'unet_spectrograms/valid', help="驗證集路徑")
parser.add_argument('--val_interval', type = int, default = 20, help="每幾輪做一次驗證")

args = parser.parse_args()

log_file = f'LOG/log_{args.label}.txt'
best_weight = f'CKPT/svs_best_{args.label}.pth'
ckpt_weight = f'CKPT/svs_{args.label}.pth'

# =========================================================================================
# 2. Training Setup
# =========================================================================================
WAV_ROOT = '.MUSDB18/' 

# 替換 Train Loader
train_loader = Data.DataLoader(
    # samples_per_epoch 設為 6400 (假設 100 首歌 * 64 採樣)
    dataset = DynamicDataset(WAV_ROOT, split='train', samples_per_song=SAMPLES_PER_SONG),
    batch_size=args.batch_size, num_workers=8, shuffle=True, pin_memory=True
)

# 替換 Valid Loader
valid_loader = Data.DataLoader(
    dataset = DynamicDataset(WAV_ROOT, split='valid', samples_per_song=SAMPLES_PER_SONG),
    batch_size=args.batch_size, num_workers=4, shuffle=False, pin_memory=True
)

model = UNet()
model.to(device)

best_val_loss = 100.
log_buffer = []
scheduler = None
start_epoch = 0

# 這是給您參考的載入寫法 (放在 train.py 初始化階段)
if os.path.exists(args.load_path):
    print(f"Loading checkpoint from {args.load_path}")
    checkpoint = torch.load(args.load_path, map_location=device)
    
    # 1. 載入模型權重
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # 2. 載入優化器狀態
    if 'optim' in checkpoint:
        model.optim.load_state_dict(checkpoint['optim'])
        
    # 3. 載入 Epoch (如果要接續訓練)
    start_epoch = checkpoint.get('epoch', 0)
    
    # 4. 載入 Scheduler (如果有)
    if scheduler is not None and 'scheduler' in checkpoint and checkpoint['scheduler'] is not None:
        scheduler.load_state_dict(checkpoint['scheduler'])
        
    # 5. 載入 Loss 紀錄 (選用)
    for key in checkpoint:
        if key.startswith('loss_list'):
            setattr(model, key, checkpoint[key])

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
    #     checkpoint = {
    #         'epoch': ep + 1,                           # 當前訓練到的 Epoch
    #         'model_state_dict': model.state_dict(),    # 模型權重
    #         'optim': model.optim.state_dict(),         # 優化器狀態 (包含 momentum 等資訊)
    #         'scheduler': scheduler.state_dict() if scheduler is not None else None, # 排程器狀態
    #     }
    #     torch.save(checkpoint, f'CKPT/svs_{args.label}_{ep}.pth')
    #     print(f"\n[Info] Epoch {ep}: Learning rate manually changed to {new_lr}!\n")
    
    for i, (mix, voc) in enumerate(loop):
        mix, voc = mix.to(device), voc.to(device)
        model.backward(mix, voc) # backward 內含 zero_grad, forward, loss, optim.step
        
        # 取得當前 batch loss
        loss_dict = model.getLoss()
        current_loss = loss_dict.get('loss_list_total', 0)
        train_loss_sum += current_loss
        
        loop.set_postfix(loss=current_loss)
    
    avg_train_loss = train_loss_sum / len(train_loader)
    log_buffer.append(f"{avg_train_loss}\n")
    
    # --- Validation Phase (Every 10 epochs) ---
    if valid_loader and (ep + 1) % args.val_interval == 0:
        model.eval() # 切換到評估模式 (關閉 Dropout, BatchNorm 變動)
        val_loss_sum = 0
        
        with torch.no_grad(): # 不計算梯度，節省記憶體
            # 使用 tqdm 顯示驗證進度
            val_loop = tqdm(valid_loader, desc=f"Epoch {ep+1} [Valid]", leave=False)
            
            for mix, voc in val_loop:
                mix, voc = mix.to(device), voc.to(device)
                
                # 手動計算 Loss (因為 model.backward 是訓練用的)
                mask = model(mix)
                loss = model.crit(voc, mix, mask)
                val_loss_sum += loss.item()
        
        avg_val_loss = val_loss_sum / len(valid_loader)
        log_buffer.append(f"Val {avg_val_loss}\n")
        print(f"\n[Epoch {ep+1}] Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f}")
        
        if avg_val_loss < best_val_loss:
            model.save(best_weight)
            
        # [新增] 觸發 Validation 時，將累積的 Buffer 寫入 log.txt
        try:
            with open(log_file, 'a') as f:
                f.writelines(log_buffer)
            print(f"已將 {len(log_buffer)} 筆 Loss 紀錄寫入 {log_file}")
            log_buffer = [] # 清空 Buffer
        except Exception as e:
            print(f"寫入 {log_file} 失敗: {e}")
    
    else:
        # 平常只印 Train Loss
        print(f"Epoch {ep+1} Avg Loss: {avg_train_loss:.6f}")

    # [修改] 儲存 Checkpoint (取代原本的 model.save)
    checkpoint = {
        'epoch': ep + 1,                           # 當前訓練到的 Epoch
        'model_state_dict': model.state_dict(),    # 模型權重
        'optim': model.optim.state_dict(),         # 優化器狀態 (包含 momentum 等資訊)
        'scheduler': scheduler.state_dict() if scheduler is not None else None, # 排程器狀態
    }

    # 為了保持與舊版相容，我們也把 loss history 存進去
    for key in model.__dict__:
        if key.startswith('loss_list'):
            checkpoint[key] = getattr(model, key)

    # 執行儲存
    torch.save(checkpoint, ckpt_weight)

if log_buffer:
    with open(log_file, 'a') as f:
        f.writelines(log_buffer)
    print(f"剩餘 Log 已寫入 {log_file}")

print("Finish training!")

"""
1208 midnight:
python train.py \
    --label aug \
    --batch_size 32 \
    --epoch 900 \
    --val_interval 10 \
    --load_path CKPT/svs_L1_1.pth
    
想法：
    --train_folder unet_spectrograms/train \
    --valid_folder unet_spectrograms/valid \

"""