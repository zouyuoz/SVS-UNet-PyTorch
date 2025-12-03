from model import UNet
import torch.utils.data as Data
import numpy as np
import argparse
import torch
import os
from tqdm import tqdm
import random

"""
    SVS-UNet Training Script with Validation
    Updated for Python 3.10+ & PyTorch 2.x
"""

# Determine device
device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

# =========================================================================================
# Dataset Definition
# =========================================================================================
class SpectrogramDataset(Data.Dataset):
    def __init__(self, path):
        """
        path: 指向 unet_spectrogram/train 或 unet_spectrogram/valid
        """
        self.path = path
        self.mixture_path = os.path.join(path, 'mixture')
        self.vocal_path = os.path.join(path, 'vocal')

        if not os.path.exists(self.mixture_path):
            raise FileNotFoundError(f"找不到 Mixture 資料夾: {self.mixture_path}")
        if not os.path.exists(self.vocal_path):
            raise FileNotFoundError(f"找不到 Vocals 資料夾: {self.vocal_path}")

        # 讀取檔名
        mix_files = sorted([f for f in os.listdir(self.mixture_path) if f.endswith('_spec.npy')])
        
        # 雙重確認對應檔案存在
        self.files = [f for f in mix_files if os.path.exists(os.path.join(self.vocal_path, f))]
        
        print(f"[{os.path.basename(path)}] 載入 {len(self.files)} 筆資料")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file_name = self.files[idx]
        
        # 1. Load (Freq=513, Time=T)
        mix = np.load(os.path.join(self.mixture_path, file_name))
        voc = np.load(os.path.join(self.vocal_path, file_name))

        # 2. Crop Freq (513 -> 512)
        mix = mix[1:, :]
        voc = voc[1:, :]

        # 3. Random Crop Time (Time -> 128)
        target_len = 128
        curr_len = mix.shape[1]
        
        if curr_len > target_len:
            start = random.randint(0, curr_len - target_len)
            mix = mix[:, start:start + target_len]
            voc = voc[:, start:start + target_len]
        else:
            pad_width = target_len - curr_len
            mix = np.pad(mix, ((0, 0), (0, pad_width)), mode='constant')
            voc = np.pad(voc, ((0, 0), (0, pad_width)), mode='constant')

        # 4. To Tensor (1, 512, 128)
        mix = torch.from_numpy(mix[np.newaxis, :, :].astype(np.float32))
        voc = torch.from_numpy(voc[np.newaxis, :, :].astype(np.float32))
        
        return mix, voc

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

parser.add_argument('--valid_folder', type=str, default='unet_spectrograms/valid', help="驗證集路徑")
parser.add_argument('--batch_size', type=int, default=4)
parser.add_argument('--val_interval', type=int, default=20, help="每幾輪做一次驗證")

args = parser.parse_args()
# =========================================================================================
# 2. Training Setup
# =========================================================================================
# Train Loader
train_dataset = SpectrogramDataset(args.train_folder)
train_loader = Data.DataLoader(
    dataset=train_dataset,
    batch_size=args.batch_size,
    num_workers=4, 
    shuffle=True,
    pin_memory=True if device.type == 'cuda' else False
)

# Valid Loader (如果不為空)
valid_loader = None
if os.path.exists(args.valid_folder):
    valid_dataset = SpectrogramDataset(args.valid_folder)
    if len(valid_dataset) > 0:
        valid_loader = Data.DataLoader(
            dataset=valid_dataset,
            batch_size=args.batch_size, # 驗證時 batch size 可以大一點，這裡先保持一致
            num_workers=2,
            shuffle=False, # 驗證不需要洗牌
            pin_memory=True if device.type == 'cuda' else False
        )
else:
    print(f"Warning: 找不到驗證資料夾 {args.valid_folder}，將跳過驗證步驟。")

model = UNet()
model.to(device)

if os.path.exists(args.load_path):
    model.load(args.load_path)

# =========================================================================================
# Main Loop
# =========================================================================================
print(f"Start training for {args.epoch} epochs...")

for ep in range(args.epoch):
    # --- Training Phase ---
    model.train()
    loop = tqdm(train_loader, desc=f"Epoch {ep+1}/{args.epoch} [Train]")
    train_loss_sum = 0
    
    for i, (mix, voc) in enumerate(loop):
        mix, voc = mix.to(device), voc.to(device)
        model.backward(mix, voc) # backward 內含 zero_grad, forward, loss, optim.step
        
        # 取得當前 batch loss
        loss_dict = model.getLoss()
        current_loss = loss_dict.get('loss_list_vocal', 0)
        train_loss_sum += current_loss
        
        loop.set_postfix(loss=current_loss)
    
    avg_train_loss = train_loss_sum / len(train_loader)
    
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
                loss = model.crit(mask * mix, voc)
                val_loss_sum += loss.item()
        
        avg_val_loss = val_loss_sum / len(valid_loader)
        
        # 印出結果
        print(f"\n[Epoch {ep+1}] Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f}")
        
        # 這裡可以加入「若 Val Loss 創新低則存檔」的邏輯 (Early Stopping 基礎)
        # 目前先照舊，每輪都存
    
    else:
        # 平常只印 Train Loss
        print(f"Epoch {ep+1} Avg Loss: {avg_train_loss:.6f}")

    # Save Model
    model.save(args.save_path)

print("Finish training!")