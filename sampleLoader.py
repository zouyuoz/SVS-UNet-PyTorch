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

class DynamicDataset(Data.Dataset):
    def __init__(
    		self, path_to_wav_root, split='train',
      		samples_per_song=SAMPLES_PER_SONG, win_size=WINDOW_SIZE, hop_size=HOP_SIZE, sr=SAMPLE_RATE
        ):
        """
        path_to_wav_root: 原始 WAV 資料集的根目錄 (例如: musdb_wav/)
        split: 'train' 或 'valid'
        samples_per_epoch: 每個 Epoch 總共要生成多少筆資料
        """
        self.path = os.path.join(path_to_wav_root, split)
        self.win_size = win_size
        self.hop_size = hop_size
        self.sr = sr
        self.target_len = INPUT_LEN
        
        # 掃描所有歌曲資料夾
        self.song_dirs = sorted([
            os.path.join(self.path, d)
            for d in os.listdir(self.path) 
            if os.path.isdir(os.path.join(self.path, d))
        ])
        self.samples_per_epoch = samples_per_song * len(self.song_dirs)
        
        # 讀取所有 WAV 檔案的路徑
        self.song_file_info = []
        for song_dir in self.song_dirs:
            mix_path = os.path.join(song_dir, 'mixture.wav')
            voc_path = os.path.join(song_dir, 'vocals.wav')
            if os.path.exists(mix_path) and os.path.exists(voc_path):
                self.song_file_info.append({'mix': mix_path, 'voc': voc_path})
            
        print(f"Scanning complete. Found {len(self.song_file_info)} valid songs for '{split}'.")
        if len(self.song_file_info) < 2: print("警告：歌曲數量不足，無法執行 Dynamic Remixing。")

    def __len__(self):
        # 每個 Epoch 總共生成 samples_per_epoch 筆資料
        return self.samples_per_epoch

    def __getitem__(self, idx):
        # 隨機選取 Base Song (提供 Vocal) 和 Remix Song (提供 Accompaniment)
        indices = random.sample(range(len(self.song_file_info)), 2)
        base_song = self.song_file_info[indices[0]]
        remix_song = self.song_file_info[indices[1]]
        
        # --- I. 載入波形並計算 Accompaniment (處理長度差異) ---
        
        # 載入 Base Song (提供 Vocal)
        y_mix_base, _ = librosa.load(base_song['mix'], sr=self.sr, mono=True)
        y_voc_base, _ = librosa.load(base_song['voc'], sr=self.sr, mono=True)
        
        # 載入 Remix Song (提供 Accompaniment)
        y_mix_remix, _ = librosa.load(remix_song['mix'], sr=self.sr, mono=True)
        y_voc_remix, _ = librosa.load(remix_song['voc'], sr=self.sr, mono=True)
        
        # 計算 Acc 波形 (Acc = Mix - Voc)
        y_acc_base = y_mix_base - y_voc_base
        y_acc_remix = y_mix_remix - y_voc_remix
        
        # --- II. Data Augmentation ---
        
        # 1. Pitch Shifting (+/- 3 semitones)
        y_voc_final = y_voc_base
        if random.random() < 0.5:
            semitones = random.uniform(-3.0, 3.0)
            y_voc_final = librosa.effects.pitch_shift(y_voc_final, sr=self.sr, n_steps=semitones, **{'res_type': 'soxr_hq'})
            
        # 2. Dynamic Remixing (50% 機率使用 Base Acc, 50% 使用 Remix Acc)
        if random.random() < 0.5 and indices[0] != indices[1]:
            # 使用 Remix Acc: 必須先對齊長度
            min_len_acc = min(len(y_voc_final), len(y_acc_remix))
            y_acc_final = y_acc_remix[:min_len_acc]
            y_voc_final = y_voc_final[:min_len_acc] # 對齊 Vocal
        else:
            # 使用 Base Acc: 必須先對齊長度
            min_len_acc = min(len(y_voc_final), len(y_acc_base))
            y_acc_final = y_acc_base[:min_len_acc]
            y_voc_final = y_voc_final[:min_len_acc] # 對齊 Vocal
            
        # 最終混合
        y_mix_final = y_voc_final + y_acc_final
        
        # --- III. Segmenting & STFT (裁切和轉換) ---
        
        total_samples = len(y_mix_final)
        start_sample = random.randint(0, total_samples - self.target_len)
        
        # 裁切波形 Segment
        y_mix_crop = y_mix_final[start_sample : start_sample + self.target_len]
        y_voc_crop = y_voc_final[start_sample : start_sample + self.target_len]
        
        # 轉換為 Spectrogram
        stft_mix = librosa.stft(y_mix_crop, n_fft=self.win_size, hop_length=self.hop_size)
        stft_voc = librosa.stft(y_voc_crop, n_fft=self.win_size, hop_length=self.hop_size)
        
        spec_mix = np.abs(stft_mix).astype(np.float32)
        spec_voc = np.abs(stft_voc).astype(np.float32)
        
        # Normalization
        norm = spec_mix.max()
        if norm == 0: norm = 1
        
        spec_mix /= norm
        spec_voc /= norm

        # 移除 DC component (513 -> 512)
        mix_crop = spec_mix[1:, :]
        voc_crop = spec_voc[1:, :]
        
        # 轉 Tensor (1, 512, 128)
        mix_tensor = torch.from_numpy(mix_crop[np.newaxis, :, :].astype(np.float32))
        voc_tensor = torch.from_numpy(voc_crop[np.newaxis, :, :].astype(np.float32))
        
        return mix_tensor, voc_tensor

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