import os
import glob
import soundfile as sf
import mir_eval
import numpy as np
from tqdm import tqdm
import argparse
from concurrent.futures import ProcessPoolExecutor

def evaluate_one_song(args):
    """
    單首歌曲的評估函式 (為了讓多行程調用，必須獨立出來)
    """
    est_path, ref_folder, mix_folder = args
    
    basename = os.path.basename(est_path)
    ref_path = os.path.join(ref_folder, basename)
    mix_path = os.path.join(mix_folder, basename)
    
    if not os.path.exists(ref_path) or not os.path.exists(mix_path):
        return None  # 跳過找不到檔案的

    try:
        # 讀取音訊
        y_est_voc, sr = sf.read(est_path)
        y_ref_voc, _  = sf.read(ref_path)
        y_mix, _      = sf.read(mix_path)

        # 確保長度一致
        min_len = min(len(y_est_voc), len(y_ref_voc), len(y_mix))
        y_est_voc = y_est_voc[:min_len]
        y_ref_voc = y_ref_voc[:min_len]
        y_mix     = y_mix[:min_len]

        # 靜音檢查 (跳過純音樂或清唱)
        if np.sum(np.abs(y_ref_voc)) < 1e-6:
            return None # Ref Vocal 靜音
        
        # 1. 準備訊號
        y_ref_acc = y_mix - y_ref_voc
        y_est_acc = y_mix - y_est_voc
        
        reference_sources = np.vstack([y_ref_voc, y_ref_acc])
        estimated_sources = np.vstack([y_est_voc, y_est_acc])
        
        # === 計算 SDR (模型預測) ===
        (sdr, sir, sar, _) = mir_eval.separation.bss_eval_sources(
            reference_sources, 
            estimated_sources, 
            compute_permutation=False
        )
        sdr_val = sdr[0] # 取 Vocal 的 SDR
        sir_val = sir[0]
        sar_val = sar[0]

        # === 計算 NSDR ===
        # NSDR = SDR(預測) - SDR(混音)
        # 我們要把 "混音" 當作 "預測結果" 丟進去算一次 Baseline SDR
        # 假設模型什麼都沒做，預測出來的人聲 = 混音，預測出來的伴奏 = 混音
        baseline_sources = np.vstack([y_mix, y_mix]) 
        
        (sdr_base, _, _, _) = mir_eval.separation.bss_eval_sources(
            reference_sources, 
            baseline_sources, 
            compute_permutation=False
        )
        nsdr_val = sdr_val - sdr_base[0]

        return sdr_val, nsdr_val, sir_val, sar_val

    except Exception as e:
        print(f"Error processing {basename}: {e}")
        return None

def evaluate_folder_parallel(ref_folder, est_folder, mix_folder, num_workers=None):
    # 搜尋所有 wav 檔案
    est_files = sorted(glob.glob(os.path.join(est_folder, "*.wav")))
    
    if len(est_files) == 0:
        print("錯誤：找不到預測檔案。")
        return

    # 準備參數列表
    tasks = [(f, ref_folder, mix_folder) for f in est_files]

    print(f"開始多核心評估 {len(est_files)} 首歌曲...")
    
    results = []
    # 使用 ProcessPoolExecutor 進行平行運算
    # max_workers=None 會自動使用 CPU 最大核心數
    with ProcessPoolExecutor(max_workers=num_workers) as pool:
        # 使用 tqdm 顯示進度條
        for res in tqdm(pool.map(evaluate_one_song, tasks), total=len(tasks)):
            if res is not None:
                results.append(res)

    if not results:
        print("沒有成功評估任何歌曲。")
        return

    # 轉換為 numpy array 方便計算平均
    results = np.array(results)
    avg_sdr = np.mean(results[:, 0])
    avg_nsdr = np.mean(results[:, 1])
    avg_sir = np.mean(results[:, 2])
    avg_sar = np.mean(results[:, 3])

    print("\n====== 人聲 (Vocals) 評估結果 (平均值) ======")
    print(f"SDR:  {avg_sdr:.4f} dB  (整體品質)")
    print(f"NSDR: {avg_nsdr:.4f} dB (進步幅度 - Normalized SDR)")
    print(f"SIR:  {avg_sir:.4f} dB  (分離乾淨度)")
    print(f"SAR:  {avg_sar:.4f} dB  (無雜音程度)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--ref', type=str, required=True, help="Ground Truth 人聲資料夾")
    parser.add_argument('--est', type=str, required=True, help="預測結果資料夾")
    parser.add_argument('--mix', type=str, required=True, help="原始 Mixture 資料夾")
    parser.add_argument('--workers', type=int, default=None, help="使用的 CPU 核心數 (預設為全部)")
    args = parser.parse_args()
    
    # 使用 if __name__ == '__main__' 保護，這是 multiprocessing 在 Windows/Linux 必要的
    evaluate_folder_parallel(args.ref, args.est, args.mix, args.workers)
"""
# 範例指令
python evaluate.py \
    --est test_results/wav \
    --ref test_results/gt_vocal_wav_high \
    --mix test_results/gt_mixture_wav_high
    
python evaluate.py \
    --est test_results/wav \
    --ref test_results/gt_vocal_wav_low \
    --mix test_results/gt_mixture_wav_low
---
svs_best_val: 2.2913

svs_400: 
====== 人聲 (Vocals) 評估結果 (平均值) ======
SDR: 4.3780  dB  (整體品質)
SIR: 15.3448 dB  (分離乾淨度/去伴奏能力)
SAR: 5.5292  dB  (無雜音程度/自然度)

svs_1206:
====== 人聲 (Vocals) 評估結果 (平均值) ======
SDR: 2.4139  dB  (整體品質)
SIR: 11.7054 dB  (分離乾淨度/去伴奏能力)
SAR: 3.7478  dB  (無雜音程度/自然度)

svs_500:
====== 人聲 (Vocals) 評估結果 (平均值) ======
SDR: 4.3701  dB  (整體品質)
SIR: 16.4115 dB  (分離乾淨度/去伴奏能力)
SAR: 5.2707  dB  (無雜音程度/自然度)

svs_1207:
====== 人聲 (Vocals) 評估結果 (平均值) ======
SDR: 2.5353  dB  (整體品質)
SIR: 12.5568 dB  (分離乾淨度/去伴奏能力)
SAR: 3.7224  dB  (無雜音程度/自然度)

svs_1208:
====== 人聲 (Vocals) 評估結果 (平均值) ======
SDR: 3.0661 dB  (整體品質)
SIR: 15.1832 dB  (分離乾淨度/去伴奏能力)
SAR: 3.8998 dB  (無雜音程度/自然度)
"""