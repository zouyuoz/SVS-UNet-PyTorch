from model import UNet
import torch.utils.data as Data
import numpy as np
import argparse
import torch
import os
from tqdm import tqdm
import random
from utils import *
import auraloss

"""
    SVS-UNet Training Script with Validation
    Updated for Python 3.10+ & PyTorch 2.x
"""

# Determine device
device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")
mrstft_loss = auraloss.freq.MRSTFTLoss().to(device)
alpha = 0.1

# =========================================================================================
# Dataset Definition
# =========================================================================================
class SpectrogramDataset(Data.Dataset):
    def __init__(self, path, samples_per_song=SAMPLES_PER_SONG):
        self.path = path
        self.mixture_path = os.path.join(path, 'mixture')
        self.vocal_path = os.path.join(path, 'vocal')
        self.samples_per_song = samples_per_song

        if not os.path.exists(self.mixture_path):
            raise FileNotFoundError(f"找不到 Mixture 資料夾: {self.mixture_path}")

        # 讀取所有檔名
        self.file_names = sorted([f for f in os.listdir(self.mixture_path) if f.endswith('_spec.npy')])
        
        # 確保對應檔案存在
        self.file_names = [f for f in self.file_names if os.path.exists(os.path.join(self.vocal_path, f))]
        
        print(f"[{os.path.basename(path)}] 載入 {len(self.file_names)} 首歌曲，每輪採樣 {self.samples_per_song} 次，共 {len(self)} 筆資料。")

    def __len__(self):
        # 讓 Dataset 的長度變長 (歌曲數 * 每首歌採樣次數)
        return len(self.file_names) * self.samples_per_song

    def __getitem__(self, idx):
        # 透過取餘數來決定現在要讀哪首歌
        file_name = self.file_names[idx % len(self.file_names)]
        
        # 1. 讀取 .npy
        mix = np.load(os.path.join(self.mixture_path, file_name))
        voc = np.load(os.path.join(self.vocal_path, file_name))

        # 2. 裁切頻率 (513 -> 512)
        mix = mix[1:, :]
        voc = voc[1:, :]

        # 3. 隨機裁切時間軸 (Time -> 128)
        target_len = INPUT_LEN
        curr_len = mix.shape[1]
        
        if curr_len > target_len:
            # 隨機選一個起點
            start = random.randint(0, curr_len - target_len)
            mix = mix[:, start:start + target_len]
            voc = voc[:, start:start + target_len]
        else:
            # Padding
            pad_width = target_len - curr_len
            mix = np.pad(mix, ((0, 0), (0, pad_width)), mode='constant')
            voc = np.pad(voc, ((0, 0), (0, pad_width)), mode='constant')

        # 4. 轉 Tensor
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
parser.add_argument('--label'       , type = str, required = True)
parser.add_argument('--epoch'       , type = int, default = 2)
parser.add_argument('--batch_size'  , type = int, default = 2)

parser.add_argument('--valid_folder', type = str, default = 'unet_spectrograms/valid', help="驗證集路徑")
parser.add_argument('--val_interval', type = int, default = 20, help="每幾輪做一次驗證")

args = parser.parse_args()

log_file = f'LOG/log_{args.label}.txt'
best_weight = f'CKPT/svs_best_{args.label}.pth'
ckpt_weight = f'CKPT/svs_{args.label}.pth'

# =========================================================================================
# 2. Training Setup
# =========================================================================================

# Train Loader
train_dataset = SpectrogramDataset(args.train_folder)
train_loader = Data.DataLoader(
    dataset=train_dataset,
    batch_size=args.batch_size,
    num_workers=8, 
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
    print(f"Loaded checkpoint from {args.load_path}")

best_val_loss = 100.
log_buffer = []
scheduler = None
start_epoch = 0
optimizer = model.optimizer

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
    
    # --- Validation Phase (Every 20 epochs) ---
    if valid_loader and (ep + 1) % args.val_interval == 0:
        model.eval() # 切換到評估模式 (關閉 Dropout, BatchNorm 變動)
        val_loss_sum = 0
        
        with torch.no_grad(): # 不計算梯度，節省記憶體
            # 使用 tqdm 顯示驗證進度
            val_loop = tqdm(valid_loader, desc=f"Epoch {ep+1} [Valid]", leave=False)
            
            for mix, voc in val_loop:
                mix, voc = mix.to(device), voc.to(device)
                mask = model(mix)
                loss_tensor = model.crit(mask * mix, voc) + model.crit((1 - mask) * mix, torch.clamp(mix - voc, min=0.0))
                val_loss_sum += loss_tensor.item()
        
        avg_val_loss = val_loss_sum / len(valid_loader)
        log_buffer.append(f"Val {avg_val_loss}\n")
        print(f"\n[Epoch {ep+1}] Train Loss: {avg_train_loss:.4e} | Val Loss: {avg_val_loss:.4e}")
        
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
        print(f"Epoch {ep+1} Avg Loss: {avg_train_loss:.4e}")

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
    --train_folder unet_spectrograms/train \
    --valid_folder unet_spectrograms/valid \
    --label L1_ft16 \
    --batch_size 16 \
    --epoch 500 \
    --val_interval 20 \
    --load_path CKPT/svs_1209_L1.ckpt
    
想法：

"""