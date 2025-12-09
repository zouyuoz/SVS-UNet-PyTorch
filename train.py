from model import UNet
import torch.utils.data as Data
import numpy as np
import argparse
import torch
import torch.nn.functional as F
import os
from tqdm import tqdm
import random
from config import *
import auraloss

"""
    SVS-UNet Training Script with Validation
    Updated for Python 3.10+ & PyTorch 2.x
"""

# Determine device
device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

# 1. 初始化 MR-STFT Loss
# 建議參數：alpha 通常設小一點 (e.g., 0.1 或 1.0)，視 L1 Loss 的數值量級而定
alpha_L1 = 166.66
alpha_MR = .66
mrstft_loss_fn = auraloss.freq.MultiResolutionSTFTLoss(sample_rate=SAMPLE_RATE,device=device)

# 2. 準備 iSTFT 需要的 Window (避免在迴圈內重複建立，節省資源)
# 確保 window 放在正確的 device 上
stft_window = torch.hann_window(WINDOW_SIZE).to(device)

# 定義一個 iSTFT 輔助函數，方便轉換
def specific_istft(magnitude, phase):
    # magnitude shape: [Batch, 1, 512, Time]
    # phase shape:     [Batch, 1, 512, Time]
    
    # 1. 在頻率軸 (Dim 2) 的最前面補一個 0，變回 513
    # F.pad 格式為 (left, right, top, bottom, front, back...)
    # 對應最後兩維 (Time, Freq)，所以是 (0, 0, 1, 0) -> Time不補, Freq上面補1
    magnitude = F.pad(magnitude, (0, 0, 1, 0), "constant", 0)
    phase     = F.pad(phase,     (0, 0, 1, 0), "constant", 0)
    
    # 2. 轉 Complex
    complex_tensor = torch.polar(magnitude, phase)
    
    # 3. 調整維度給 istft (需要 [Batch, Freq, Time])
    # 移除 Channel 維度 (Batch, 1, 513, Time) -> (Batch, 513, Time)
    complex_tensor = complex_tensor.squeeze(1)
    
    # 4. iSTFT
    waveform = torch.istft(
        complex_tensor,
        n_fft=1024,
        hop_length=HOP_SIZE,
        win_length=WINDOW_SIZE,
        window=stft_window,
        return_complex=False
    )
    
    return waveform.unsqueeze(1)

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

        # 讀取所有檔名 (以前綴 _spec.npy 為主)
        self.file_names = sorted([f for f in os.listdir(self.mixture_path) if f.endswith('_spec.npy')])
        
        # 確保對應的 Vocal 檔案存在
        self.file_names = [f for f in self.file_names if os.path.exists(os.path.join(self.vocal_path, f))]
        
        print(f"[{os.path.basename(path)}] 載入 {len(self.file_names)} 首歌曲，每輪採樣 {self.samples_per_song} 次，共 {len(self)} 筆資料。")

    def __len__(self):
        return len(self.file_names) * self.samples_per_song

    def __getitem__(self, idx):
        # 透過取餘數來決定現在要讀哪首歌
        file_name = self.file_names[idx % len(self.file_names)]
        
        # 建構 Phase 的檔名 (將 _spec.npy 替換為 _phase.npy)
        phase_name = file_name.replace('_spec.npy', '_phase.npy')
        
        # 1. 讀取 .npy (Mag + Phase)
        mix_path = os.path.join(self.mixture_path, file_name)
        voc_path = os.path.join(self.vocal_path, file_name)
        mix_phase_path = os.path.join(self.mixture_path, phase_name)
        voc_phase_path = os.path.join(self.vocal_path, phase_name)
        
        mix = np.load(mix_path)
        voc = np.load(voc_path)
        
        # [新增] 讀取 Phase，若找不到則報錯或給預設值(建議報錯以確保資料正確)
        mix_phase = np.load(mix_phase_path)
        voc_phase = np.load(voc_phase_path)
        mix_phase = np.angle(mix_phase).astype(np.float32)
        voc_phase = np.angle(voc_phase).astype(np.float32)

        # 2. 裁切頻率 (513 -> 512) [注意：Phase 也要跟著裁切]
        # 這裡切掉了第 0 個 bin (DC component)，之後 iSTFT 前記得補回來
        mix = mix[1:, :]
        voc = voc[1:, :]
        mix_phase = mix_phase[1:, :]
        voc_phase = voc_phase[1:, :]

        # 3. 隨機裁切時間軸 (Time -> 128)
        target_len = INPUT_LEN
        curr_len = mix.shape[1]
        
        if curr_len > target_len:
            # [關鍵] 隨機選一個起點，Mag 和 Phase 必須共用這個 start
            start = random.randint(0, curr_len - target_len)
            
            mix = mix[:, start:start + target_len]
            voc = voc[:, start:start + target_len]
            mix_phase = mix_phase[:, start:start + target_len]
            voc_phase = voc_phase[:, start:start + target_len]
        else:
            # Padding
            pad_width = target_len - curr_len
            # Mag 補 0
            mix = np.pad(mix, ((0, 0), (0, pad_width)), mode='constant')
            voc = np.pad(voc, ((0, 0), (0, pad_width)), mode='constant')
            # Phase 補 0 (數學上 Magnitude 為 0 時 Phase 無意義，補 0 即可)
            mix_phase = np.pad(mix_phase, ((0, 0), (0, pad_width)), mode='constant')
            voc_phase = np.pad(voc_phase, ((0, 0), (0, pad_width)), mode='constant')

        # 4. 轉 Tensor
        mix = torch.from_numpy(mix[np.newaxis, :, :].astype(np.float32))
        voc = torch.from_numpy(voc[np.newaxis, :, :].astype(np.float32))
        mix_phase = torch.from_numpy(mix_phase[np.newaxis, :, :].astype(np.float32))
        voc_phase = torch.from_numpy(voc_phase[np.newaxis, :, :].astype(np.float32))
        
        return mix, voc, mix_phase, voc_phase

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
optimizer = model.optim

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
    
    # 修改：這裡假設 loader 回傳四個值，請根據你的 dataset 調整順序
    for i, (mix, voc, mix_phase, voc_phase) in enumerate(loop):
        # 1. 資料搬移到 Device
        mix, voc = mix.to(device), voc.to(device)
        mix_phase, voc_phase = mix_phase.to(device), voc_phase.to(device)
        
        # 2. 清空梯度
        model.optim.zero_grad() # 或是 optimizer.zero_grad()
        
        # 3. 模型 Forward (預測 Mask)
        mask = model(mix)
        pred_vocal = mask * mix
        pred_accomp = (1 - mask) * mix
        target_accomp = torch.clamp(mix - voc, min=0.0)
        
        # 4. 計算原本的 L1 Loss (Spectrogram Domain)
        # 你的舊 Loss: model.crit 可能是 L1Loss
        loss_v = model.crit(pred_vocal, voc)
        loss_a = model.crit(pred_accomp, target_accomp)
        l1_loss = loss_v + loss_a
        
        # 5. 計算 MR-STFT Loss (Time Domain)
        # (A) 重建預測波形：使用 預測Magnitude + 混合音Phase (標準 SVS 作法)
        pred_voc_wav = specific_istft(pred_vocal, mix_phase)
        
        # (B) 重建目標波形：使用 真實Magnitude + 真實Phase
        target_voc_wav = specific_istft(voc, voc_phase)
        
        # (C) 計算 auraloss
        mr_loss = mrstft_loss_fn(pred_voc_wav, target_voc_wav)
        
        # 6. 結合 Loss
        total_loss = (alpha_L1 * l1_loss) + (alpha_MR * mr_loss)
        
        # 7. Backward & Update
        total_loss.backward()
        model.optim.step() # 或是 optimizer.step()
        
        # 紀錄
        current_loss = total_loss.item()
        train_loss_sum += current_loss
        
        # 顯示 L1 和 MR 的個別數值，方便監控比例
        loop.set_postfix(
            L1=f"{l1_loss.item():.4f}", 
            MR=f"{mr_loss.item():.4f}", 
            Total=f"{current_loss:.4f}"
        )
    
    avg_train_loss = train_loss_sum / len(train_loader)
    log_buffer.append(f"{avg_train_loss}\n")
    
    # --- Validation Phase ---
    if valid_loader and (ep + 1) % args.val_interval == 0:
        model.eval()
        val_loss_sum = 0
        
        with torch.no_grad():
            val_loop = tqdm(valid_loader, desc=f"Epoch {ep+1} [Valid]", leave=False)
            
            # 修改：同樣需要 Phase 資訊
            for mix, voc, mix_phase, voc_phase in val_loop:
                mix, voc = mix.to(device), voc.to(device)
                mix_phase, voc_phase = mix_phase.to(device), voc_phase.to(device)
                
                mask = model(mix)
                pred_vocal = mask * mix
                
                # 計算 L1 (包含 Vocal 和 Instrumental)
                # Instrumental Mag = (1-Mask) * Mix
                # Instrumental Target = Mix - Voc (或是你有單獨載入 inst)
                pred_inst_mag = (1 - mask) * mix
                inst_target = torch.clamp(mix - voc, min=0.0)
                
                loss_spec = model.crit(pred_vocal, voc) + model.crit(pred_inst_mag, inst_target)
                
                # 計算 MR-STFT (這裡只針對 Vocal 做優化，也可以對 Inst 做，看需求)
                pred_voc_wav = specific_istft(pred_vocal, mix_phase)
                target_voc_wav = specific_istft(voc, voc_phase)
                loss_mr = mrstft_loss_fn(pred_voc_wav, target_voc_wav)
                
                # 總 Val Loss
                val_total_loss = (alpha_L1 * loss_spec) + (alpha_MR * loss_mr)
                val_loss_sum += val_total_loss.item()
        
        avg_val_loss = val_loss_sum / len(valid_loader)
        log_buffer.append(f"Val {avg_val_loss}\n")
        print(f"\n[Epoch {ep+1}] Train Loss: {avg_train_loss:.4e} | Val Loss: {avg_val_loss:.4e}")
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss # 記得更新 best_val_loss
            model.save(best_weight)
            
        try:
            with open(log_file, 'a') as f:
                f.writelines(log_buffer)
            print(f"已將 {len(log_buffer)} 筆 Loss 紀錄寫入 {log_file}")
            log_buffer = []
        except Exception as e:
            print(f"寫入失敗: {e}")
    
    else:
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
    --label L1+SL_test \
    --batch_size 32 \
    --epoch 100 \
    --val_interval 10
    
想法：

"""