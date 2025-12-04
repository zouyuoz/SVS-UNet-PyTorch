import os
import glob
import soundfile as sf
import mir_eval
import numpy as np
from tqdm import tqdm

def evaluate_folder(ref_folder, est_folder):
    """
    ref_folder: 包含 Ground Truth (純人聲) 的資料夾 (例如 unet_spectrograms/test/vocals 轉回來的 wav，或者原始 MUSDB18 的 wav)
    est_folder: 您模型預測出來的 wav 資料夾
    """
    # 搜尋所有 wav 檔案
    ref_files = sorted(glob.glob(os.path.join(ref_folder, "*.wav")))
    est_files = sorted(glob.glob(os.path.join(est_folder, "*.wav")))

    sdr_list, sir_list, sar_list = [], [], []

    print(f"開始評估 {len(est_files)} 首歌曲...")

    for ref_path, est_path in tqdm(zip(ref_files, est_files), total=len(ref_files)):
        # 讀取音訊
        ref_audio, sr = sf.read(ref_path)
        est_audio, sr2 = sf.read(est_path)
        
        # 確保採樣率一致
        assert sr == sr2, "採樣率不一致！"
        if ref_audio.ndim > 1:
                ref_audio = np.mean(ref_audio, axis=1)

        # 確保長度一致 (取最小值)
        min_len = min(len(ref_audio), len(est_audio))
        ref_audio = ref_audio[:min_len]
        est_audio = est_audio[:min_len]

        # mir_eval 需要 shape 為 (n_sources, n_samples)
        # 我們這裡是單一人聲評估，所以 n_sources = 1
        reference_sources = ref_audio[np.newaxis, :]
        estimated_sources = est_audio[np.newaxis, :]

        # 計算 BSS Eval 指標
        (sdr, sir, sar, perm) = mir_eval.separation.bss_eval_sources(
            reference_sources, 
            estimated_sources, 
            compute_permutation=False
        )

        sdr_list.append(sdr[0])
        sir_list.append(sir[0])
        sar_list.append(sar[0])

    print("\n====== 評估結果 (平均值) ======")
    print(f"SDR: {np.mean(sdr_list):.4f} dB (越高越好)")
    print(f"SIR: {np.mean(sir_list):.4f} dB (越高越好)")
    print(f"SAR: {np.mean(sar_list):.4f} dB (越高越好)")

if __name__ == "__main__":
    # 設定路徑 (請依您的實際路徑修改)
    # Ground Truth: 您必須先用 data.py 把 test set 的 vocals 轉回 wav 才能比較
    # 或者是直接指到 MUSDB18 的原始 wav 資料夾
    
    # 範例：假設您把測試集的正確答案轉回 wav 放在 'test_gt_wav'
    # 預測結果放在 'test_results_wav'
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--ref', type=str, required=True, help="Ground Truth 音訊資料夾")
    parser.add_argument('--est', type=str, required=True, help="預測結果音訊資料夾")
    args = parser.parse_args()
    
    evaluate_folder(args.ref, args.est)
    
"""
python evaluate.py \
    --ref test_results/gt_wav \
    --est test_results/wav
"""