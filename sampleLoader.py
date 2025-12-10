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
      		samples_per_epoch=SAMPLES_PER_SONG, win_size=WINDOW_SIZE, hop_size=HOP_SIZE, sr=SAMPLE_RATE
        ):
        """
        path_to_wav_root: 原始 WAV 資料集的根目錄 (例如: musdb_wav/)
        split: 'train' 或 'valid'
        samples_per_epoch: 每個 Epoch 總共要生成多少筆資料
        """
        self.path = os.path.join(path_to_wav_root, split)
        self.samples_per_epoch = samples_per_epoch
        self.win_size = win_size
        self.hop_size = hop_size
        self.sr = sr
        self.target_len = INPUT_LEN
        
        # 掃描所有歌曲資料夾
        self.song_folders = sorted([
            d for d in os.listdir(self.path) 
            if os.path.isdir(os.path.join(self.path, d))
        ])
        
        # 讀取所有 WAV 檔案的路徑
        self.vocal_paths = []
        self.mixture_paths = []
        self.accompaniment_paths = [] # 為了做 Remixing，我們預先計算伴奏波形
        
        print(f"Scanning {len(self.song_folders)} songs for dynamic processing...")
        for song_name in tqdm(self.song_folders):
            song_dir = os.path.join(self.path, song_name)
            mix_path = os.path.join(song_dir, 'mixture.wav')
            voc_path = os.path.join(song_dir, 'vocals.wav')
            
            # 確保兩個核心檔案都存在
            if os.path.exists(mix_path) and os.path.exists(voc_path):
                self.vocal_paths.append(voc_path)
                self.mixture_paths.append(mix_path)
                # 由於我們沒有 Acc.wav，這裡先用 Placeholder。Acc 必須在 __getitem__ 中用相減產生。
                self.accompaniment_paths.append(song_dir) # 存放目錄，用於計算 Acc
            
        print(f"Found {len(self.vocal_paths)} valid songs.")

    def __len__(self):
        # 每個 Epoch 總共生成 samples_per_epoch 筆資料
        return self.samples_per_epoch

    def __getitem__(self, idx):
        # 隨機選取歌曲作為基底
        base_idx = random.randint(0, len(self.vocal_paths) - 1)
        base_vocal_path = self.vocal_paths[base_idx]
        base_mix_path = self.mixture_paths[base_idx]
        
        # 隨機選取另一首歌的伴奏 (Dynamic Remixing 的目標)
        remix_idx = random.randint(0, len(self.vocal_paths) - 1)
        remix_song_dir = self.accompaniment_paths[remix_idx] # 這首歌提供伴奏

        # --- I. 讀取 WAV 波形 ---
        y_mix, _ = librosa.load(base_mix_path, sr=self.sr, mono=True)
        y_voc, _ = librosa.load(base_vocal_path, sr=self.sr, mono=True)
        
        # 計算 Acc 波形 (這是最準確的方式: Acc = Mix - Voc)
        # 注意：這依賴於 musdb18 的 wav 檔案是對齊且是線性疊加的
        y_acc = y_mix - y_voc 
        
        # --- II. Dynamic Remixing ---
        if random.random() < 0.5: # 假設 50% 機率執行 Remix
            # 隨機抽取一首歌的伴奏 y_acc_remix
            y_mix_remix, _ = librosa.load(os.path.join(remix_song_dir, 'mixture.wav'), sr=self.sr, mono=True)
            y_voc_remix, _ = librosa.load(os.path.join(remix_song_dir, 'vocals.wav'), sr=self.sr, mono=True)
            y_acc_remix = y_mix_remix - y_voc_remix
            
            # 使用 Remixed Acc + Base Voc
            y_acc = y_acc_remix
            y_mix = y_voc + y_acc
        
        # --- III. Pitch Shifting (音高調整) ---
        if random.random() < 0.5: # 假設 50% 機率執行 Pitch Shift
            semitones = random.uniform(-3.0, 3.0) # +-3 semitones 隨機
            y_voc = librosa.effects.pitch_shift(y_voc, sr=self.sr, n_steps=semitones)
            
            # 重新混合
            y_mix = y_voc + y_acc # 使用 Aug 之後的 Vocal + Acc

        # --- IV. Segmenting & STFT (轉頻譜) ---
        
        # 對齊長度 (如果經過 Pitch Shift 可能長度會變，需要對齊)
        min_len = min(len(y_mix), len(y_voc)) 
        y_mix = y_mix[:min_len]
        y_voc = y_voc[:min_len]

        # 計算 Segment 參數 (以 Samples 為單位)
        total_samples = min_len
        target_samples = self.target_len * self.hop_size # 128 frames * 256 hop = 32768 samples (約 0.74秒)
        
        if total_samples < target_samples:
            # Padding 如果太短
            y_mix = np.pad(y_mix, (0, target_samples - total_samples), mode='constant')
            y_voc = np.pad(y_voc, (0, target_samples - total_samples), mode='constant')
            start_sample = 0
        else:
            # 隨機選擇 segment 起點
            start_sample = random.randint(0, total_samples - target_samples)
        
        # 裁切波形
        y_mix_crop = y_mix[start_sample : start_sample + target_samples]
        y_voc_crop = y_voc[start_sample : start_sample + target_samples]
        
        # 轉換為 Spectrogram
        stft_mix = librosa.stft(y_mix_crop, n_fft=self.win_size, hop_length=self.hop_size)
        stft_voc = librosa.stft(y_voc_crop, n_fft=self.win_size, hop_length=self.hop_size)
        
        spec_mix = np.abs(stft_mix).astype(np.float32)
        spec_voc = np.abs(stft_voc).astype(np.float32)
        
        # --- V. Normalization (使用混合音頻最大值正規化兩者) ---
        norm = spec_mix.max()
        if norm == 0: norm = 1
        
        spec_mix /= norm
        spec_voc /= norm

        # --- VI. Final Formatting ---
        # 移除 DC component (513 -> 512)
        mix_crop = spec_mix[1:, :]
        voc_crop = spec_voc[1:, :]
        
        # 轉 Tensor (1, 512, 128)
        mix_tensor = torch.from_numpy(mix_crop[np.newaxis, :, :].astype(np.float32))
        voc_tensor = torch.from_numpy(voc_crop[np.newaxis, :, :].astype(np.float32))
        
        return mix_tensor, voc_tensor