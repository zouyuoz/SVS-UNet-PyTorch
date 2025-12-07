import os
import glob
import soundfile as sf
import mir_eval
import numpy as np
from tqdm import tqdm
import argparse

def evaluate_folder(ref_folder, est_folder, mix_folder):
    """
    ref_folder: Ground Truth 人聲 wav 資料夾
    est_folder: 模型預測的人聲 wav 資料夾
    mix_folder: 原始 Mixture wav 資料夾 (用來計算伴奏)
    """
    # 搜尋所有 wav 檔案 (假設檔名是對應的)
    est_files = sorted(glob.glob(os.path.join(est_folder, "*.wav")))
    
    if len(est_files) == 0:
        print("錯誤：找不到預測檔案。")
        return

    sdr_list, sir_list, sar_list = [], [], []
    
    print(f"開始評估 {len(est_files)} 首歌曲 (Vocal + Accompaniment)...")

    for est_path in tqdm(est_files):
        # 取得檔名 (basename)
        basename = os.path.basename(est_path)
        
        # 組合對應的 GT 和 Mixture 路徑
        # 注意：這裡假設檔名完全一致。如果不一致（例如多了前綴），請自行調整 replace 邏輯
        ref_path = os.path.join(ref_folder, basename)
        mix_path = os.path.join(mix_folder, basename)
        
        if not os.path.exists(ref_path) or not os.path.exists(mix_path):
            print(f"[跳過] 找不到對應檔案: {basename}")
            continue

        # 讀取音訊
        # y_est_voc: 預測的人聲
        # y_ref_voc: 真實的人聲
        # y_mix:     原始混音
        y_est_voc, sr = sf.read(est_path)
        y_ref_voc, _  = sf.read(ref_path)
        y_mix, _      = sf.read(mix_path)

        # 確保長度一致 (取最小值)
        min_len = min(len(y_est_voc), len(y_ref_voc), len(y_mix))
        y_est_voc = y_est_voc[:min_len]
        y_ref_voc = y_ref_voc[:min_len]
        y_mix     = y_mix[:min_len]

        # === 關鍵修改：建構雙軌訊號 (Vocal, Accompaniment) ===
        
        # 1. 計算 Ground Truth 的伴奏 (Ref Acc = Mix - Ref Voc)
        y_ref_acc = y_mix - y_ref_voc
        
        # 2. 計算 預測 的伴奏 (Est Acc = Mix - Est Voc)
        # 這是 SVS 評測的標準做法，確保 Mix = Voc + Acc 守恆
        y_est_acc = y_mix - y_est_voc
        
        # === [新增] 靜音檢查 (關鍵修正) ===
        # 檢查 Ground Truth 是否全為 0 (使用一個極小值判斷)
        is_vocal_silent = np.sum(np.abs(y_ref_voc)) < 1e-6
        is_acc_silent   = np.sum(np.abs(y_ref_acc)) < 1e-6

        if is_vocal_silent:
            print(f"\n[跳過] {basename}: 人聲軌道為靜音 (純音樂?)")
            continue
        
        if is_acc_silent:
            print(f"\n[跳過] {basename}: 伴奏軌道為靜音 (清唱?)")
            continue
        # =================================

        # 3. 堆疊成 (n_sources, n_samples)
        # Source 0: Vocal
        # Source 1: Accompaniment
        reference_sources = np.vstack([y_ref_voc, y_ref_acc])
        estimated_sources = np.vstack([y_est_voc, y_est_acc])

        # 計算 BSS Eval
        # compute_permutation=False: 因為我們很確定第0軌就是人聲，不需要讓演算法去猜配對
        (sdr, sir, sar, perm) = mir_eval.separation.bss_eval_sources(
            reference_sources, 
            estimated_sources, 
            compute_permutation=False
        )

        # 我們只取 index 0 (Vocal) 的成績
        sdr_list.append(sdr[0])
        sir_list.append(sir[0])
        sar_list.append(sar[0])

    print("\n====== 人聲 (Vocals) 評估結果 (平均值) ======")
    print(f"SDR: {np.mean(sdr_list):.4f} dB  (整體品質)")
    print(f"SIR: {np.mean(sir_list):.4f} dB  (分離乾淨度/去伴奏能力)")
    print(f"SAR: {np.mean(sar_list):.4f} dB  (無雜音程度/自然度)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--ref', type=str, required=True, help="Ground Truth 人聲資料夾 (WAV)")
    parser.add_argument('--est', type=str, required=True, help="預測結果資料夾 (WAV)")
    parser.add_argument('--mix', type=str, required=True, help="原始 Mixture 資料夾 (WAV)")
    args = parser.parse_args()
    
    evaluate_folder(args.ref, args.est, args.mix)
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
SDR: 4.3701 dB  (整體品質)
SIR: 16.4115 dB  (分離乾淨度/去伴奏能力)
SAR: 5.2707 dB  (無雜音程度/自然度)

svs_1207:
====== 人聲 (Vocals) 評估結果 (平均值) ======
SDR: 2.4451 dB  (整體品質)
SIR: 12.2559 dB  (分離乾淨度/去伴奏能力)
SAR: 3.8303 dB  (無雜音程度/自然度)
"""