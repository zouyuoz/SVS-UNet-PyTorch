#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import argparse
import glob
import numpy as np
import librosa
import mir_eval
import csv
import warnings

warnings.filterwarnings("ignore")

def load_mono_audio(path):
    """
    讀取 wav 轉為 mono 保持原始取樣率。
    回傳: waveform (np.ndarray, shape (T,)), sr
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
    y, sr = librosa.load(path, sr=None, mono=True)
    return y, sr


def compute_metrics_for_track(mix_path, vocal_ref_path, vocal_est_path):
    """
    對單一 track 計算:
        - SDR, SIR, SAR (for vocal)
        - NSDR (normalized SDR, 相對於 mixture)
    """

    # 讀取音訊
    mix, sr_mix = load_mono_audio(mix_path)
    vocal_ref, sr_ref = load_mono_audio(vocal_ref_path)
    vocal_est, sr_est = load_mono_audio(vocal_est_path)

    if not (sr_mix == sr_ref == sr_est):
        raise ValueError(
            f"Sample rate mismatch: mix={sr_mix}, ref={sr_ref}, est={sr_est}"
        )

    # 對齊長度
    min_len = min(len(mix), len(vocal_ref), len(vocal_est))
    mix = mix[:min_len]
    vocal_ref = vocal_ref[:min_len]
    vocal_est = vocal_est[:min_len]

    # === 1) 構造 2-source：vocal + accomp ===
    acc_ref = mix - vocal_ref      # 近似真實伴奏
    acc_est = mix - vocal_est      # 近似預測伴奏

    # shape: (n_sources, T) = (2, T)
    sources_ref = np.stack([vocal_ref, acc_ref], axis=0)
    sources_est = np.stack([vocal_est, acc_est], axis=0)

    # === 2) 用 BSS_eval 計算各種指標 ===
    sdr, sir, sar, perm = mir_eval.separation.bss_eval_sources(sources_ref, sources_est)

    # perm 會告訴你哪個 est 對應哪個 ref，但這裡我們假設順序一致（vocal 在 index 0）
    # 如果想要嚴謹一點，可以用 perm[0] 當 vocal 的 index
    vocal_idx = int(perm[0])  # est 中對應 vocal_ref 的 index

    sdr_vocal = float(sdr[vocal_idx])
    sir_vocal = float(sir[vocal_idx])
    sar_vocal = float(sar[vocal_idx])

    # === 3) 計算 NSDR (只針對 vocal) ===
    # NSDR = SDR(vocal_est, vocal_ref) - SDR(mix, vocal_ref)

    # step 1: mixture 當成「估計的 vocal」，單 source 評估
    s_ref_v = vocal_ref[None, :]   # (1, T)
    mix_as_est = mix[None, :]      # (1, T)
    sdr_mix, _, _, _ = mir_eval.separation.bss_eval_sources(s_ref_v, mix_as_est)
    sdr_mix_vocal = float(sdr_mix[0])

    nsdr_vocal = sdr_vocal - sdr_mix_vocal

    return {
        "SDR": sdr_vocal,
        "SIR": sir_vocal,
        "SAR": sar_vocal,
        "NSDR": nsdr_vocal,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate SVS results with SDR / SIR / SAR / NSDR (vocal only).")
    parser.add_argument("--est"    , type=str, required = True , help="預測人聲 wav 檔案資料夾（你的輸出）")
    parser.add_argument("--mix"    , type=str, required = True , help="真實 mixture wav 資料夾")
    parser.add_argument("--ref"    , type=str, required = True , help="真實 vocal wav 資料夾")
    parser.add_argument("--ext"    , type=str, default  = "wav", help="音檔副檔名(預設 wav)")
    parser.add_argument("--out_csv", type=str, default  = None , help="如果指定，會把每首歌的結果寫到這個 CSV 檔")

    args = parser.parse_args()

    pred_dir = args.est
    mix_dir = args.mix
    vocal_dir = args.ref
    ext = args.ext

    # 找出所有預測人聲檔案
    pred_files = sorted(glob.glob(os.path.join(pred_dir, f"*.{ext}")))
    if len(pred_files) == 0:
        print(f"[Error] No *.{ext} files found in {pred_dir}")
        return

    all_results = []
    sdr_list, sir_list, sar_list, nsdr_list = [], [], [], []

    print("=== Start Evaluation ===")
    print(f"#tracks = {len(pred_files)}\n")

    for pred_path in pred_files:
        basename = os.path.basename(pred_path)
        mix_path = os.path.join(mix_dir, basename)
        vocal_ref_path = os.path.join(vocal_dir, basename)

        # 檔案對不上就跳過
        if not os.path.exists(mix_path):
            print(f"[Warning] Mixture file not found, skip: {mix_path}")
            continue
        if not os.path.exists(vocal_ref_path):
            print(f"[Warning] Vocal ref file not found, skip: {vocal_ref_path}")
            continue

        try:
            metrics = compute_metrics_for_track(mix_path, vocal_ref_path, pred_path)
        except Exception as e:
            print(f"[Error] Failed on {basename}: {e}")
            continue

        track_name = os.path.splitext(basename)[0]

        print(
            f"{track_name[:20]}:\t"
            f"SDR={metrics['SDR']:.3f} dB,\t"
            f"SIR={metrics['SIR']:.3f} dB,\t"
            f"SAR={metrics['SAR']:.3f} dB,\t"
            f"NSDR={metrics['NSDR']:.3f} dB"
        )

        sdr_list.append(metrics["SDR"])
        sir_list.append(metrics["SIR"])
        sar_list.append(metrics["SAR"])
        nsdr_list.append(metrics["NSDR"])

        all_results.append(
            {
                "track": track_name,
                "SDR": metrics["SDR"],
                "SIR": metrics["SIR"],
                "SAR": metrics["SAR"],
                "NSDR": metrics["NSDR"],
            }
        )

    if len(all_results) == 0:
        print("\n[Error] No valid tracks evaluated.")
        return

    # ---- 整體平均 ----
    mean_sdr = float(np.mean(sdr_list))
    mean_sir = float(np.mean(sir_list))
    mean_sar = float(np.mean(sar_list))
    mean_nsdr = float(np.mean(nsdr_list))

    print("\n=== Overall Mean Metrics (vocal) ===")
    print(f"Mean SDR : {mean_sdr:.3f} dB")
    print(f"Mean SIR : {mean_sir:.3f} dB")
    print(f"Mean SAR : {mean_sar:.3f} dB")
    print(f"Mean NSDR: {mean_nsdr:.3f} dB")

    # ---- 輸出到 CSV（如果有指定）----
    if args.out_csv is not None:
        fieldnames = ["track", "SDR", "SIR", "SAR", "NSDR"]
        with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in all_results:
                writer.writerow(row)
        print(f"\n[Info] Results saved to {args.out_csv}")


if __name__ == "__main__":
    main()

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

svs_L1_ft16:
=== Overall Mean Metrics (vocal) ===
Mean SDR : 3.595 dB
Mean SIR : 15.544 dB
Mean SAR : 4.244 dB
Mean NSDR: 8.820 dB

"""