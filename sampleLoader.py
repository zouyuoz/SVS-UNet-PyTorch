import torch
import torch.utils.data as Data
import numpy as np
import os
import random
import librosa
from tqdm import tqdm
import warnings
from utils import *

warnings.filterwarnings("ignore")

import torch.nn.functional as F

class SpectrogramDataset(Data.Dataset):
    def __init__(self, path, samples_per_song=64, augment=False, nmixx=False):
        self.path = path
        self.mixture_path = os.path.join(path, 'mixture')
        self.vocal_path = os.path.join(path, 'vocal')
        self.samples_per_song = samples_per_song
        self.augment = augment
        self.nmixx = nmixx 

        if not os.path.exists(self.mixture_path):
            raise FileNotFoundError(f"找不到 Mixture 資料夾: {self.mixture_path}")

        self.file_names = sorted([f for f in os.listdir(self.mixture_path) if f.endswith('_spec.npy')])
        self.file_names = [f for f in self.file_names if os.path.exists(os.path.join(self.vocal_path, f))]
        
        mode_str = "Train (w/ Aug)" if augment else "Valid"
        print(f"[{os.path.basename(path)}] [{mode_str}] 載入 {len(self.file_names)} 首歌曲，每輪採樣 {self.samples_per_song} 次。")

    def __len__(self):
        return len(self.file_names) * self.samples_per_song

    def _random_crop(self, mat, crop_width):
        """ 輔助函式：對單一頻譜圖進行隨機裁切或 Padding """
        curr_len = mat.shape[1]
        if curr_len > crop_width:
            start = random.randint(0, curr_len - crop_width)
            return mat[:, start:start + crop_width]
        else:
            pad_width = crop_width - curr_len
            return np.pad(mat, ((0, 0), (0, pad_width)), mode='constant')

    def __getitem__(self, idx):
        # 1. 取得檔案索引
        idx_a = idx % len(self.file_names)
        file_name_a = self.file_names[idx_a]
        
        # 2. 決定 Augmentation 參數 (先決定要切多長)
        scale_f = 1.0
        scale_t = 1.0
        if self.augment:
            scale_f = random.uniform(0.7, 1.3)
            scale_t = random.uniform(0.7, 1.3)
            
        target_width = INPUT_LEN
        crop_width = int(target_width * scale_t) # 根據 Time Stretch 比例決定裁切長度

        # 3. 讀取與處理
        # 讀取 Song A (作為 Base 或 Accompaniment 來源)
        mix_a = np.load(os.path.join(self.mixture_path, file_name_a))[1:, :] # (512, T)
        voc_a = np.load(os.path.join(self.vocal_path, file_name_a))[1:, :] # (512, T)

        # === 核心修改邏輯 ===
        if self.nmixx and random.random() < 0.98: # 100% 機率觸發 Remix
            # 隨機選 Song B (提供 Vocal)
            idx_b = (idx_a + random.randint(1, len(self.file_names) - 1)) % len(self.file_names)
            file_name_b = self.file_names[idx_b]
            voc_b = np.load(os.path.join(self.vocal_path, file_name_b))[1:, :] # (512, T_b)

            # A. 計算 Song A 的伴奏 (Approximation: Mix - Voc)
            acc_a = np.maximum(0, mix_a - voc_a)

            # B. 獨立裁切 (解決維度不匹配問題)
            acc_crop = self._random_crop(acc_a, crop_width)
            voc_crop = self._random_crop(voc_b, crop_width)

            # C. 合成新數據 (New Mix = Acc A + Voc B)
            mix_crop = acc_crop + voc_crop
            target_crop = voc_crop # 目標是 Song B 的人聲

        else:
            # === 一般情況 (同一首歌) ===
            # 必須使用「相同」的起點裁切，所以不能用分開的 _random_crop
            curr_len = mix_a.shape[1]
            if curr_len > crop_width:
                start = random.randint(0, curr_len - crop_width)
                mix_crop = mix_a[:, start:start + crop_width]
                target_crop = voc_a[:, start:start + crop_width]
            else:
                pad_width = crop_width - curr_len
                mix_crop = np.pad(mix_a, ((0, 0), (0, pad_width)), mode='constant')
                target_crop = np.pad(voc_a, ((0, 0), (0, pad_width)), mode='constant')

        # --- 4. 執行縮放 (Interpolation) ---
        if self.augment and (scale_f != 1.0 or scale_t != 1.0):
            # 轉 Tensor 並增加維度 (B, C, H, W)
            mix_tensor = torch.from_numpy(mix_crop).unsqueeze(0).unsqueeze(0)
            voc_tensor = torch.from_numpy(target_crop).unsqueeze(0).unsqueeze(0)
            
            target_height_raw = int(512 * scale_f)
            
            # 雙線性插值
            mix_resized = F.interpolate(mix_tensor, size=(target_height_raw, target_width), mode='bilinear', align_corners=False)
            voc_resized = F.interpolate(voc_tensor, size=(target_height_raw, target_width), mode='bilinear', align_corners=False)
            
            # 修復頻率維度
            if target_height_raw > 512:
                mix_final = mix_resized[:, :, :512, :]
                voc_final = voc_resized[:, :, :512, :]
            elif target_height_raw < 512:
                pad_h = 512 - target_height_raw
                mix_final = F.pad(mix_resized, (0, 0, 0, pad_h), mode='constant', value=0)
                voc_final = F.pad(voc_resized, (0, 0, 0, pad_h), mode='constant', value=0)
            else:
                mix_final = mix_resized
                voc_final = voc_resized
            
            mix = mix_final.squeeze(0)
            voc = voc_final.squeeze(0)
        else:
            mix = torch.from_numpy(mix_crop[np.newaxis, :, :].astype(np.float32))
            voc = torch.from_numpy(target_crop[np.newaxis, :, :].astype(np.float32))

        return mix.float(), voc.float()