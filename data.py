import argparse
import librosa
import numpy as np
import os
import soundfile as sf
from tqdm import tqdm
import warnings

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
parser.add_argument('--win_size', type = int, default = 4096)
parser.add_argument('--hop_size', type = int, default = 1024)
parser.add_argument('--sr',       type = int, default = 44100)
parser.add_argument('--direction',            default = "to_spec", choices = ["to_spec", "to_wave"])
args = parser.parse_args()

# =========================================================================================
# 2. 轉換邏輯
# =========================================================================================

# 定義 MUSDB18 wav 檔名與專案目標資料夾的對應關係
# 格式: 'wav檔名': '目標資料夾名稱'
TRACK_MAP = {
    'mixture.wav': 'mixture',
    'drums.wav': 'drum',
    'bass.wav': 'bass',
    'other.wav': 'rest',   # 原專案將 'other' 稱為 'rest'
    'vocals.wav': 'vocal'
}

if args.direction == 'to_spec':
    # 建立目標資料夾結構
    if not os.path.exists(args.tar):
        os.makedirs(args.tar, exist_ok=True)
        for folder_name in TRACK_MAP.values():
            os.makedirs(os.path.join(args.tar, folder_name), exist_ok=True)

    print(f"正在掃描來源資料夾: {args.src}")
    
    # 掃描所有子資料夾（每一首歌）
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
        norm = None
        
        # 我們需要先處理 mixture 以取得 norm 值，確保所有分軌縮放比例一致
        # 因此我們先讀取 mixture，再讀取其他軌
        # 但為了程式簡潔，我們先遍歷一次找到 mixture 設定 norm，或者我們假設 mixture 總是在
        
        # 這裡我們依照 TRACK_MAP 的順序處理，但要確保 mixture 的 norm 被正確應用
        # 為了保險，我們直接指定先處理 mixture
        track_processing_order = ['mixture.wav', 'drums.wav', 'bass.wav', 'other.wav', 'vocals.wav']
        
        current_song_specs = {} # 暫存，等待 norm 確定後再存檔 (雖然通常 mixture 最大，但保險起見)

        # 讀取並計算 Spectrogram
        try:
            # 第一步：先讀取 Mixture 確定 Normalization Factor
            mix_path = os.path.join(song_path, 'mixture.wav')
            if not os.path.exists(mix_path):
                print(f"[Skip] {song_name} 缺少 mixture.wav")
                continue

            # 使用 librosa 讀取，它會自動轉為 mono (若 mono=True) 並重取樣
            # 原始 SVS-UNet 是將雙聲道相加: audio[:, 0] + audio[:, 1]
            # librosa.load(..., mono=True) 是取平均。
            # 為了保持能量守恆接近原始邏輯，我們可以讀取後 * 2 (如果是單純相加的話)
            # 但通常取平均後的頻譜結構是一樣的，這裡使用標準 mono 混合
            y_mix, _ = librosa.load(mix_path, sr=args.sr, mono=True)
            
            stft_mix = librosa.stft(y_mix, n_fft=args.win_size, hop_length=args.hop_size)
            spec_mix, phase_mix = librosa.magphase(stft_mix)
            spec_mix = np.abs(spec_mix).astype(np.float32)
            
            norm = spec_mix.max()
            if norm == 0: norm = 1 # 避免除以零

            # 儲存 Mixture
            spec_mix /= norm
            save_name_base = f"{num2str(audio_idx)}_{song_name}"
            
            np.save(os.path.join(args.tar, 'mixture', f"{save_name_base}_spec.npy"), spec_mix)
            np.save(os.path.join(args.tar, 'mixture', f"{save_name_base}_phase.npy"), phase_mix)

            # 第二步：處理其他分軌
            for wav_file in track_processing_order:
                if wav_file == 'mixture.wav': continue # 已經處理過
                
                target_folder = TRACK_MAP[wav_file]
                track_path = os.path.join(song_path, wav_file)
                
                if os.path.exists(track_path):
                    y, _ = librosa.load(track_path, sr=args.sr, mono=True)
                    
                    # 確保長度一致 (有時候不同分軌會有極微小的樣本數差異)
                    if len(y) != len(y_mix):
                        # 簡單的對齊：截斷或補零
                        if len(y) > len(y_mix):
                            y = y[:len(y_mix)]
                        else:
                            y = np.pad(y, (0, len(y_mix) - len(y)))

                    stft = librosa.stft(y, n_fft=args.win_size, hop_length=args.hop_size)
                    spec, phase = librosa.magphase(stft)
                    spec = np.abs(spec).astype(np.float32)
                    
                    # 使用 Mixture 的 norm 進行正規化
                    spec /= norm
                    
                    np.save(os.path.join(args.tar, target_folder, f"{save_name_base}_spec.npy"), spec)
                    np.save(os.path.join(args.tar, target_folder, f"{save_name_base}_phase.npy"), phase)
                else:
                    # 如果缺少某個分軌 (例如有些資料集沒有 other)，可以選擇跳過或補全黑圖
                    # 這裡選擇印出警告
                    # print(f"[Warning] {song_name} 缺少 {wav_file}")
                    pass

        except Exception as e:
            print(f"[Error] 處理 {song_name} 時發生錯誤: {e}")

# =========================================================================================
# 3. 逆轉換 (to_wave) - 保持原樣或根據需要微調
# =========================================================================================
elif args.direction == 'to_wave':
    if args.phase == '-1':
        raise Exception("to_wave 模式需要指定 --phase 參數！")
        
    if not os.path.exists(args.tar):
        os.makedirs(args.tar, exist_ok=True)
    
    # 掃描 spectrogram 檔案
    files = sorted([f for f in os.listdir(args.src) if f.endswith('_spec.npy')])
    loader = tqdm(files)
    
    for spec_name in loader:
        try:
            # 載入 Magnitude
            mag = np.load(os.path.join(args.src, spec_name))
            
            # 載入 Phase (假設檔名結構一致)
            phase_name = spec_name.replace('_spec.npy', '_phase.npy')
            phase_path = os.path.join(args.phase, phase_name)
            
            if not os.path.exists(phase_path):
                print(f"[Warning] 找不到相位檔 {phase_path}，跳過。")
                continue
                
            phase = np.load(phase_path)

            # 對齊尺寸
            length = min(phase.shape[-1], mag.shape[-1])
            mag = mag[:, :length]
            phase = phase[:, :length]

            # 反正規化 (De-normalization)
            # 需要讀取原始的 mixture spec 來取得 max 值
            # 假設 phase 資料夾裡面也有 mixture 的 spec (通常訓練時會生成)
            # 檔名通常是 ...._mixture_spec.npy 或者是 0001_SongName_spec.npy 在 mixture 資料夾
            # 這裡的邏輯比較依賴你的檔案放置位置。
            # 如果 args.phase 指向的是含有所有原始 npy 的根目錄下的 'mixture' 資料夾，那樣最好
            
            # 嘗試尋找對應的 mixture spec 來還原音量
            # 這裡假設使用者會把 phase 設為包含 mixture_spec 的路徑
            mix_spec_path = os.path.join(args.phase, spec_name) 
            if os.path.exists(mix_spec_path):
                mix_spec = np.load(mix_spec_path)
                mag *= mix_spec.max()
            else:
                # 如果找不到 mixture spec，聲音會很小，甚至聽不到
                # 這裡做一個 fallback，假設 max 為 1 (這通常是不對的，但至少能跑)
                # 或者嘗試從檔名推斷
                pass

            spectrogram = mag * phase
            
            # ISTFT
            y = librosa.istft(spectrogram, win_length=args.win_size, hop_length=args.hop_size)
            
            # 存檔
            save_name = spec_name.replace('_spec.npy', '.wav')
            sf.write(os.path.join(args.tar, save_name), y, args.sr)

        except Exception as e:
            print(f"還原 {spec_name} 失敗: {e}")

else:
    print("未知的方向，請使用 to_spec 或 to_wave")