import argparse
import librosa
import numpy as np
import os
import soundfile as sf
from tqdm import tqdm
import warnings
from utils import *

# 忽略不必要的警告
warnings.filterwarnings("ignore")

def num2str(n):
    return str(n).zfill(4)

# =========================================================================================
# 1. 參數設定
# =========================================================================================
parser = argparse.ArgumentParser()
parser.add_argument('--src',      type = str, required = True, help="來源資料夾 (包含多個歌曲資料夾)")
parser.add_argument('--tar',      type = str, required = True, help="目標資料夾 (存放 Spectrogram)")
parser.add_argument('--phase',    type = str, default = '-1',  help="Phase 資料夾 (僅用於 to_wave)")
parser.add_argument('--win_size', type = int, default = WINDOW_SIZE)
parser.add_argument('--hop_size', type = int, default = HOP_SIZE)
parser.add_argument('--sr',       type = int, default = SAMPLE_RATE)
parser.add_argument('--direction',            default = "to_spec", choices = ["to_spec", "to_wave"])
args = parser.parse_args()

# =========================================================================================
# 2. 轉換邏輯
# =========================================================================================

# 定義 MUSDB18 wav 檔名與專案目標資料夾的對應關係
# 格式: 'wav檔名': '目標資料夾名稱'
TRACK_MAP = {
    'mixture.wav': 'mixture',
    'vocals.wav':  'vocal'
} 

if args.direction == 'to_spec':
    # 建立目標資料夾結構
    if not os.path.exists(args.tar):
        os.makedirs(args.tar, exist_ok=True)
        
    for folder_name in TRACK_MAP.values():
        os.makedirs(os.path.join(args.tar, folder_name), exist_ok=True)

    print(f"正在掃描來源資料夾: {args.src}")
    song_folders = sorted([
        d for d in os.listdir(args.src) 
        if os.path.isdir(os.path.join(args.src, d))
    ])
    print(f"找到 {len(song_folders)} 首歌曲資料夾。")
    
    if len(song_folders) == 0:
        print("錯誤: 未找到任何歌曲資料夾，請確認 --src 路徑正確。")
        exit(1)

    loader = tqdm(song_folders)
    for audio_idx, song_name in enumerate(loader):
        loader.set_description(f"Processing {song_name}")
        song_path = os.path.join(args.src, song_name)
        
        # 用來正規化的基準 (通常使用 mixture 的最大值)
        norm = None# 為了確保正規化一致，我們先讀取 mixture
        
        mix_path = os.path.join(song_path, 'mixture.wav')
        if not os.path.exists(mix_path): continue
        
        try:
            # 第一步：先讀取 Mixture 確定 Normalization Factor
            y_mix, _ = librosa.load(mix_path, sr=args.sr, mono=True)
            stft_mix = librosa.stft(y_mix, n_fft=args.win_size, hop_length=args.hop_size)
            spec_mix, phase_mix = librosa.magphase(stft_mix)
            spec_mix = np.abs(spec_mix).astype(np.float32)
            
            # 計算正規化係數
            norm = spec_mix.max()
            if norm == 0: norm = 1# 處理所有軌道
            
            processing_list = ['mixture.wav', 'vocals.wav'] # 訓練只需要這兩個，省空間可只轉這兩軌
            for wav_file in processing_list:
                target_folder = TRACK_MAP.get(wav_file)
                if not target_folder: continue
                
                track_path = os.path.join(song_path, wav_file)
                if os.path.exists(track_path):
                    y, _ = librosa.load(track_path, sr=args.sr, mono=True)
                    
                    # 長度對齊
                    if len(y) > len(y_mix): y = y[:len(y_mix)]
                    else: y = np.pad(y, (0, len(y_mix) - len(y)))

                    stft = librosa.stft(y, n_fft=args.win_size, hop_length=args.hop_size)
                    spec, phase = librosa.magphase(stft)
                    spec = np.abs(spec).astype(np.float32)
                    
                    # [關鍵] 全部除以 Mixture 的最大值
                    spec /= norm
                    
                    save_name_base = f"{num2str(audio_idx)}_{song_name}"
                    np.save(os.path.join(args.tar, target_folder, f"{save_name_base}_spec.npy"), spec)
                    # 只有 mixture 需要存 phase 供還原使用 (或者全部存也可以)
                    if wav_file == 'mixture.wav':
                        np.save(os.path.join(args.tar, target_folder, f"{save_name_base}_phase.npy"), phase)

        except Exception as e:
            print(f"Error processing {song_name}: {e}")

# =========================================================================================
# 3. 逆轉換 (to_wave) - 保持原樣或根據需要微調
# =========================================================================================
elif args.direction == 'to_wave':
    if args.phase == '-1': raise Exception("需指定 --phase")
    if not os.path.exists(args.tar): os.makedirs(args.tar, exist_ok=True)
    
    files = sorted([f for f in os.listdir(args.src) if f.endswith('_spec.npy')])
    # files = files[:20]
    print(f"正在還原 {len(files)} 個檔案...")
    
    for spec_name in tqdm(files):
        try:
            mag = np.load(os.path.join(args.src, spec_name))
            
            # 嘗試讀取對應的相位檔
            # 注意：這裡假設 phase 資料夾結構是平的，或者與 src 結構對應
            # 如果 phase 參數指向的是原始 mixture 資料夾
            phase_name = spec_name.replace('_spec.npy', '_phase.npy')
            # 嘗試幾種可能的路徑組合
            possible_paths = [
                os.path.join(args.phase, phase_name),
                os.path.join(args.phase, 'mixture', phase_name) # 為了相容 train_data 結構
            ]
            
            phase = None
            for p in possible_paths:
                if os.path.exists(p):
                    phase = np.load(p)
                    break
            
            if phase is None:
                # 如果真的找不到相位，使用隨機相位（聽起來會怪怪的但有聲音）
                # print(f"Warning: No phase for {spec_name}, using random phase.")
                phase = np.exp(2j * np.pi * np.random.rand(*mag.shape))
            
            # 對齊尺寸
            min_len = min(mag.shape[1], phase.shape[1])
            mag = mag[:, :min_len]
            phase = phase[:, :min_len]
            
            # [修正點 2] 關於音量過小的修復
            # 因為我們遺失了原始 norm 值，直接還原出來的數值會在 0~1 之間（非常小聲）
            # 我們需要手動將還原後的音訊 Normalize 到 -1~1 區間
            
            y = librosa.istft(mag * phase, win_length=args.win_size, hop_length=args.hop_size)
            
            # 檢查最大振幅
            max_val = np.max(np.abs(y))
            if max_val > 0:
                y = y / max_val * 0.9 # 正規化到 0.9 (避免破音)
            
            sf.write(os.path.join(args.tar, spec_name.replace('_spec.npy', '.wav')), y, args.sr)
            
        except Exception as e:
            print(f"還原失敗 {spec_name}: {e}")

else:
    print("未知的方向，請使用 to_spec 或 to_wave")
    
"""
轉成 gt wav

python data.py \
    --direction to_wave \
    --src unet_spectrograms/test/mixture \
    --phase unet_spectrograms/test/mixture  \
    --tar test_results/gt_mixture_wav_low

python data.py \
    --direction to_wave \
    --src unet_spectrograms/test/vocal \
    --phase unet_spectrograms/test/mixture  \
    --tar test_results/gt_vocal_wav_low

"""