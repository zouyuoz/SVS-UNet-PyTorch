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

# class DynamicDataset(Data.Dataset):
#     def __init__(
#     		self, path_to_wav_root, split='train',
#       		samples_per_song=SAMPLES_PER_SONG, win_size=WINDOW_SIZE, hop_size=HOP_SIZE, sr=SAMPLE_RATE
#         ):
#         """
#         path_to_wav_root: 原始 WAV 資料集的根目錄 (例如: musdb_wav/)
#         split: 'train' 或 'valid'
#         samples_per_epoch: 每個 Epoch 總共要生成多少筆資料
#         """
#         self.path = os.path.join(path_to_wav_root, split)
#         self.win_size = win_size
#         self.hop_size = hop_size
#         self.sr = sr
#         self.target_len = INPUT_LEN
        
#         # 掃描所有歌曲資料夾
#         self.song_dirs = sorted([
#             os.path.join(self.path, d)
#             for d in os.listdir(self.path) 
#             if os.path.isdir(os.path.join(self.path, d))
#         ])
#         self.samples_per_epoch = samples_per_song * len(self.song_dirs)
        
#         # 讀取所有 WAV 檔案的路徑
#         self.song_file_info = []
#         for song_dir in self.song_dirs:
#             mix_path = os.path.join(song_dir, 'mixture.wav')
#             voc_path = os.path.join(song_dir, 'vocals.wav')
#             if os.path.exists(mix_path) and os.path.exists(voc_path):
#                 self.song_file_info.append({'mix': mix_path, 'voc': voc_path})
            
#         print(f"Scanning complete. Found {len(self.song_file_info)} valid songs for '{split}'.")
#         if len(self.song_file_info) < 2: print("警告：歌曲數量不足，無法執行 Dynamic Remixing。")

#     def __len__(self):
#         # 每個 Epoch 總共生成 samples_per_epoch 筆資料
#         return self.samples_per_epoch

#     def __getitem__(self, idx):
#         # 隨機選取 Base Song (提供 Vocal) 和 Remix Song (提供 Accompaniment)
#         indices = random.sample(range(len(self.song_file_info)), 2)
#         base_song = self.song_file_info[indices[0]]
#         remix_song = self.song_file_info[indices[1]]
        
#         # --- I. 載入波形並計算 Accompaniment (處理長度差異) ---
        
#         # 載入 Base Song (提供 Vocal)
#         y_mix_base, _ = librosa.load(base_song['mix'], sr=self.sr, mono=True)
#         y_voc_base, _ = librosa.load(base_song['voc'], sr=self.sr, mono=True)
        
#         # 載入 Remix Song (提供 Accompaniment)
#         y_mix_remix, _ = librosa.load(remix_song['mix'], sr=self.sr, mono=True)
#         y_voc_remix, _ = librosa.load(remix_song['voc'], sr=self.sr, mono=True)
        
#         # 計算 Acc 波形 (Acc = Mix - Voc)
#         y_acc_base = y_mix_base - y_voc_base
#         y_acc_remix = y_mix_remix - y_voc_remix
        
#         # --- II. Data Augmentation ---
        
#         # 1. Pitch Shifting (+/- 3 semitones)
#         y_voc_final = y_voc_base
#         if random.random() < 0.5:
#             semitones = random.uniform(-3.0, 3.0)
#             y_voc_final = librosa.effects.pitch_shift(y_voc_final, sr=self.sr, n_steps=semitones, **{'res_type': 'soxr_hq'})
            
#         # 2. Dynamic Remixing (50% 機率使用 Base Acc, 50% 使用 Remix Acc)
#         if random.random() < 0.5 and indices[0] != indices[1]:
#             # 使用 Remix Acc: 必須先對齊長度
#             min_len_acc = min(len(y_voc_final), len(y_acc_remix))
#             y_acc_final = y_acc_remix[:min_len_acc]
#             y_voc_final = y_voc_final[:min_len_acc] # 對齊 Vocal
#         else:
#             # 使用 Base Acc: 必須先對齊長度
#             min_len_acc = min(len(y_voc_final), len(y_acc_base))
#             y_acc_final = y_acc_base[:min_len_acc]
#             y_voc_final = y_voc_final[:min_len_acc] # 對齊 Vocal
            
#         # 最終混合
#         y_mix_final = y_voc_final + y_acc_final
        
#         # --- III. Segmenting & STFT (裁切和轉換) ---
        
#         total_samples = len(y_mix_final)
#         start_sample = random.randint(0, total_samples - self.target_len)
        
#         # 裁切波形 Segment
#         y_mix_crop = y_mix_final[start_sample : start_sample + self.target_len]
#         y_voc_crop = y_voc_final[start_sample : start_sample + self.target_len]
        
#         # 轉換為 Spectrogram
#         stft_mix = librosa.stft(y_mix_crop, n_fft=self.win_size, hop_length=self.hop_size)
#         stft_voc = librosa.stft(y_voc_crop, n_fft=self.win_size, hop_length=self.hop_size)
        
#         spec_mix = np.abs(stft_mix).astype(np.float32)
#         spec_voc = np.abs(stft_voc).astype(np.float32)
        
#         # Normalization
#         norm = spec_mix.max()
#         if norm == 0: norm = 1
        
#         spec_mix /= norm
#         spec_voc /= norm

#         # 移除 DC component (513 -> 512)
#         mix_crop = spec_mix[1:, :]
#         voc_crop = spec_voc[1:, :]
        
#         # 轉 Tensor (1, 512, 128)
#         mix_tensor = torch.from_numpy(mix_crop[np.newaxis, :, :].astype(np.float32))
#         voc_tensor = torch.from_numpy(voc_crop[np.newaxis, :, :].astype(np.float32))
        
#         return mix_tensor, voc_tensor

# class SpectrogramDataset(Data.Dataset):
#     def __init__(self, path, samples_per_song=SAMPLES_PER_SONG):
#         self.path = path
#         self.mixture_path = os.path.join(path, 'mixture')
#         self.vocal_path = os.path.join(path, 'vocal')
#         self.samples_per_song = samples_per_song

#         if not os.path.exists(self.mixture_path):
#             raise FileNotFoundError(f"找不到 Mixture 資料夾: {self.mixture_path}")

#         # 讀取所有檔名
#         self.file_names = sorted([f for f in os.listdir(self.mixture_path) if f.endswith('_spec.npy')])
        
#         # 確保對應檔案存在
#         self.file_names = [f for f in self.file_names if os.path.exists(os.path.join(self.vocal_path, f))]
        
#         print(f"[{os.path.basename(path)}] 載入 {len(self.file_names)} 首歌曲，每輪採樣 {self.samples_per_song} 次，共 {len(self)} 筆資料。")

#     def __len__(self):
#         # 讓 Dataset 的長度變長 (歌曲數 * 每首歌採樣次數)
#         return len(self.file_names) * self.samples_per_song

#     def __getitem__(self, idx):
#         # 透過取餘數來決定現在要讀哪首歌
#         file_name = self.file_names[idx % len(self.file_names)]
        
#         # 1. 讀取 .npy
#         mix = np.load(os.path.join(self.mixture_path, file_name))
#         voc = np.load(os.path.join(self.vocal_path, file_name))

#         # 2. 裁切頻率 (513 -> 512)
#         mix = mix[1:, :]
#         voc = voc[1:, :]

#         # 3. 隨機裁切時間軸 (Time -> 128)
#         target_len = INPUT_LEN
#         curr_len = mix.shape[1]
        
#         start = random.randint(0, curr_len - target_len)
#         mix = mix[:, start:start + target_len]
#         voc = voc[:, start:start + target_len]

#         # 4. 轉 Tensor
#         mix = torch.from_numpy(mix[np.newaxis, :, :].astype(np.float32))
#         voc = torch.from_numpy(voc[np.newaxis, :, :].astype(np.float32))
        
#         return mix, voc

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
        if self.nmixx and random.random() < 0.5: # 50% 機率觸發 Remix (建議不要 100%，保留原始數據)
            # 隨機選 Song B (提供 Vocal)
            idx_b = (idx_a + random.randint(1, len(self.file_names) - 1)) % len(self.file_names)
            file_name_b = self.file_names[idx_b]
            voc_b = np.load(os.path.join(self.vocal_path, file_name_b))[1:, :] # (512, T_b)

            # A. 計算 Song A 的伴奏 (Approximation: Mix - Voc)
            # 使用 ReLU (maximum 0) 避免頻譜相減出現負值
            acc_a = np.maximum(0, mix_a - voc_a)

            # B. 獨立裁切 (解決維度不匹配問題)
            # Song A (Acc) 和 Song B (Voc) 長度不同，必須分開切
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