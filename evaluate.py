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

"""
這個程式會計算 test 資料夾下所有音樂 (20 首) 的預測人聲的 metrics (SDR, SAR, SIR)，並計算平均
會跑蠻久的
使用方式：終端機輸入
python evaluate.py \
    --est <存放 預測人聲的 "wav 檔" 的 folder> \
    --ref <存放 真實人聲的 "wav 檔" 的 folder> \
    --mix <存放 原始音樂的 "wav 檔" 的 folder>
注意三個資料夾內的 wav 檔的採樣率必須相同
"""

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
            f"{track_name[:19]}:\t"
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

svs_vanilla:

0000_AM Contra - He:    SDR=6.875 dB,   SIR=15.746 dB,  SAR=7.593 dB,   NSDR=12.098 dB
0001_Al James - Sch:    SDR=4.170 dB,   SIR=11.982 dB,  SAR=5.223 dB,   NSDR=5.193 dB
0002_Angels In Ampl:    SDR=6.753 dB,   SIR=13.445 dB,  SAR=7.993 dB,   NSDR=9.755 dB
0003_Arise - Run Ru:    SDR=1.568 dB,   SIR=7.659 dB,   SAR=3.482 dB,   NSDR=9.812 dB
0004_BKS - Bulldoze:    SDR=-6.464 dB,  SIR=1.212 dB,   SAR=-3.204 dB,  NSDR=7.519 dB
0005_BKS - Too Much:    SDR=7.297 dB,   SIR=14.711 dB,  SAR=8.311 dB,   NSDR=11.937 dB
0006_Ben Carrigan -:    SDR=1.562 dB,   SIR=14.267 dB,  SAR=1.961 dB,   NSDR=8.460 dB
0007_Bobby Nobody -:    SDR=2.428 dB,   SIR=9.984 dB,   SAR=3.681 dB,   NSDR=8.445 dB
0008_Buitraker - Re:    SDR=1.010 dB,   SIR=7.111 dB,   SAR=3.005 dB,   NSDR=10.685 dB
0009_Carlos Gonzale:    SDR=2.863 dB,   SIR=11.682 dB,  SAR=3.759 dB,   NSDR=4.351 dB
0010_Cristina Vane :    SDR=6.182 dB,   SIR=14.586 dB,  SAR=7.008 dB,   NSDR=10.782 dB
0011_Detsky Sad - W:    SDR=0.417 dB,   SIR=5.506 dB,   SAR=3.104 dB,   NSDR=8.188 dB
0012_Enda Reilly - :    SDR=6.142 dB,   SIR=14.396 dB,  SAR=7.001 dB,   NSDR=10.010 dB
0013_Forkupines - S:    SDR=-0.333 dB,  SIR=13.636 dB,  SAR=0.029 dB,   NSDR=9.435 dB
0014_Georgia Wonder:    SDR=0.881 dB,   SIR=8.146 dB,   SAR=2.403 dB,   NSDR=7.850 dB
0015_Girls Under Gl:    SDR=2.113 dB,   SIR=11.642 dB,  SAR=2.914 dB,   NSDR=12.669 dB
0016_Hollow Ground :    SDR=4.200 dB,   SIR=13.011 dB,  SAR=5.024 dB,   NSDR=15.113 dB
0017_James Elder & :    SDR=5.501 dB,   SIR=13.623 dB,  SAR=6.412 dB,   NSDR=8.674 dB
0018_Juliet's Rescu:    SDR=4.718 dB,   SIR=14.728 dB,  SAR=5.319 dB,   NSDR=13.657 dB
0019_Little Chicago:    SDR=5.411 dB,   SIR=9.628 dB,   SAR=7.926 dB,   NSDR=7.615 dB

=== Overall Mean Metrics (vocal) ===
Mean SDR : 3.165 dB
Mean SIR : 11.335 dB
Mean SAR : 4.447 dB
Mean NSDR: 9.612 dB

svs_vanilla_aug:

0000_AM Contra - He:    SDR=8.476 dB,   SIR=16.474 dB,  SAR=9.322 dB,   NSDR=13.699 dB
0001_Al James - Sch:    SDR=3.766 dB,   SIR=12.059 dB,  SAR=4.725 dB,   NSDR=4.789 dB
0002_Angels In Ampl:    SDR=6.973 dB,   SIR=15.016 dB,  SAR=7.849 dB,   NSDR=9.974 dB
0003_Arise - Run Ru:    SDR=3.124 dB,   SIR=10.182 dB,  SAR=4.474 dB,   NSDR=11.368 dB
0004_BKS - Bulldoze:    SDR=-7.817 dB,  SIR=6.416 dB,   SAR=-6.757 dB,  NSDR=6.166 dB
0005_BKS - Too Much:    SDR=7.740 dB,   SIR=16.498 dB,  SAR=8.457 dB,   NSDR=12.381 dB
0006_Ben Carrigan -:    SDR=0.155 dB,   SIR=16.696 dB,  SAR=0.344 dB,   NSDR=7.053 dB
0007_Bobby Nobody -:    SDR=3.431 dB,   SIR=11.902 dB,  SAR=4.369 dB,   NSDR=9.449 dB
0008_Buitraker - Re:    SDR=1.694 dB,   SIR=7.060 dB,   SAR=3.966 dB,   NSDR=11.369 dB
0009_Carlos Gonzale:    SDR=3.759 dB,   SIR=11.418 dB,  SAR=4.878 dB,   NSDR=5.247 dB
0010_Cristina Vane :    SDR=6.418 dB,   SIR=14.057 dB,  SAR=7.406 dB,   NSDR=11.017 dB
0011_Detsky Sad - W:    SDR=1.755 dB,   SIR=7.941 dB,   SAR=3.598 dB,   NSDR=9.527 dB
0012_Enda Reilly - :    SDR=6.337 dB,   SIR=14.505 dB,  SAR=7.207 dB,   NSDR=10.205 dB
0013_Forkupines - S:    SDR=4.013 dB,   SIR=15.891 dB,  SAR=4.415 dB,   NSDR=13.781 dB
0014_Georgia Wonder:    SDR=1.473 dB,   SIR=9.082 dB,   SAR=2.806 dB,   NSDR=8.442 dB
0015_Girls Under Gl:    SDR=3.371 dB,   SIR=14.055 dB,  SAR=3.926 dB,   NSDR=13.926 dB
0016_Hollow Ground :    SDR=4.076 dB,   SIR=11.720 dB,  SAR=5.179 dB,   NSDR=14.990 dB
0017_James Elder & :    SDR=4.983 dB,   SIR=12.810 dB,  SAR=5.988 dB,   NSDR=8.157 dB
0018_Juliet's Rescu:    SDR=5.643 dB,   SIR=15.954 dB,  SAR=6.176 dB,   NSDR=14.581 dB
0019_Little Chicago:    SDR=5.106 dB,   SIR=8.238 dB,   SAR=8.605 dB,   NSDR=7.310 dB

=== Overall Mean Metrics (vocal) ===
Mean SDR : 3.724 dB
Mean SIR : 12.399 dB
Mean SAR : 4.847 dB
Mean NSDR: 10.171 dB

svs_ag:

0000_AM Contra - He:    SDR=6.849 dB,   SIR=17.269 dB,  SAR=7.343 dB,   NSDR=12.073 dB
0001_Al James - Sch:    SDR=3.843 dB,   SIR=12.265 dB,  SAR=4.768 dB,   NSDR=4.866 dB
0002_Angels In Ampl:    SDR=6.889 dB,   SIR=14.347 dB,  SAR=7.906 dB,   NSDR=9.891 dB
0003_Arise - Run Ru:    SDR=2.364 dB,   SIR=9.668 dB,   SAR=3.702 dB,   NSDR=10.607 dB
0004_BKS - Bulldoze:    SDR=-7.901 dB,  SIR=0.807 dB,   SAR=-4.647 dB,  NSDR=6.082 dB
0005_BKS - Too Much:    SDR=7.701 dB,   SIR=16.392 dB,  SAR=8.430 dB,   NSDR=12.341 dB
0006_Ben Carrigan -:    SDR=0.936 dB,   SIR=14.018 dB,  SAR=1.324 dB,   NSDR=7.834 dB
0007_Bobby Nobody -:    SDR=2.886 dB,   SIR=9.460 dB,   SAR=4.431 dB,   NSDR=8.903 dB
0008_Buitraker - Re:    SDR=0.158 dB,   SIR=6.762 dB,   SAR=2.060 dB,   NSDR=9.833 dB
0009_Carlos Gonzale:    SDR=3.437 dB,   SIR=11.436 dB,  SAR=4.488 dB,   NSDR=4.925 dB
0010_Cristina Vane :    SDR=5.879 dB,   SIR=14.584 dB,  SAR=6.657 dB,   NSDR=10.479 dB
0011_Detsky Sad - W:    SDR=0.172 dB,   SIR=4.865 dB,   SAR=3.198 dB,   NSDR=7.943 dB
0012_Enda Reilly - :    SDR=6.310 dB,   SIR=14.996 dB,  SAR=7.076 dB,   NSDR=10.177 dB
0013_Forkupines - S:    SDR=0.442 dB,   SIR=12.318 dB,  SAR=0.981 dB,   NSDR=10.209 dB
0014_Georgia Wonder:    SDR=0.803 dB,   SIR=8.681 dB,   SAR=2.127 dB,   NSDR=7.772 dB
0015_Girls Under Gl:    SDR=2.907 dB,   SIR=11.688 dB,  SAR=3.808 dB,   NSDR=13.463 dB
0016_Hollow Ground :    SDR=3.364 dB,   SIR=12.263 dB,  SAR=4.213 dB,   NSDR=14.277 dB
0017_James Elder & :    SDR=5.652 dB,   SIR=14.048 dB,  SAR=6.498 dB,   NSDR=8.825 dB
0018_Juliet's Rescu:    SDR=4.871 dB,   SIR=15.353 dB,  SAR=5.402 dB,   NSDR=13.809 dB
0019_Little Chicago:    SDR=5.843 dB,   SIR=10.925 dB,  SAR=7.795 dB,   NSDR=8.048 dB

=== Overall Mean Metrics (vocal) ===
Mean SDR : 3.170 dB
Mean SIR : 11.607 dB
Mean SAR : 4.378 dB
Mean NSDR: 9.618 dB

svs_...

svs_ag_mid_aug_best:

0000_AM Contra - He:    SDR=8.787 dB,   SIR=17.305 dB,  SAR=9.526 dB,   NSDR=14.011 dB
0001_Al James - Sch:    SDR=4.091 dB,   SIR=11.036 dB,  SAR=5.400 dB,   NSDR=5.113 dB
0002_Angels In Ampl:    SDR=7.353 dB,   SIR=15.499 dB,  SAR=8.196 dB,   NSDR=10.354 dB
0003_Arise - Run Ru:    SDR=2.555 dB,   SIR=9.057 dB,   SAR=4.164 dB,   NSDR=10.799 dB
0004_BKS - Bulldoze:    SDR=-9.455 dB,  SIR=4.493 dB,   SAR=-7.955 dB,  NSDR=4.528 dB
0005_BKS - Too Much:    SDR=7.862 dB,   SIR=16.015 dB,  SAR=8.690 dB,   NSDR=12.502 dB
0006_Ben Carrigan -:    SDR=0.304 dB,   SIR=15.823 dB,  SAR=0.540 dB,   NSDR=7.202 dB
0007_Bobby Nobody -:    SDR=2.945 dB,   SIR=9.382 dB,   SAR=4.538 dB,   NSDR=8.962 dB
0008_Buitraker - Re:    SDR=1.967 dB,   SIR=8.276 dB,   SAR=3.727 dB,   NSDR=11.642 dB
0009_Carlos Gonzale:    SDR=3.877 dB,   SIR=10.020 dB,  SAR=5.498 dB,   NSDR=5.365 dB
0010_Cristina Vane :    SDR=6.399 dB,   SIR=14.464 dB,  SAR=7.289 dB,   NSDR=10.998 dB
0011_Detsky Sad - W:    SDR=1.372 dB,   SIR=6.685 dB,   SAR=3.729 dB,   NSDR=9.143 dB
0012_Enda Reilly - :    SDR=6.414 dB,   SIR=14.706 dB,  SAR=7.255 dB,   NSDR=10.282 dB
0013_Forkupines - S:    SDR=2.966 dB,   SIR=14.819 dB,  SAR=3.400 dB,   NSDR=12.733 dB
0014_Georgia Wonder:    SDR=1.501 dB,   SIR=9.029 dB,   SAR=2.857 dB,   NSDR=8.470 dB
0015_Girls Under Gl:    SDR=3.171 dB,   SIR=13.840 dB,  SAR=3.736 dB,   NSDR=13.727 dB
0016_Hollow Ground :    SDR=3.929 dB,   SIR=11.148 dB,  SAR=5.163 dB,   NSDR=14.842 dB
0017_James Elder & :    SDR=5.353 dB,   SIR=12.707 dB,  SAR=6.463 dB,   NSDR=8.527 dB
0018_Juliet's Rescu:    SDR=5.170 dB,   SIR=16.182 dB,  SAR=5.632 dB,   NSDR=14.109 dB
0019_Little Chicago:    SDR=4.757 dB,   SIR=7.718 dB,   SAR=8.495 dB,   NSDR=6.961 dB

=== Overall Mean Metrics (vocal) ===
Mean SDR : 3.566 dB (4.251)
Mean SIR : 11.910 dB (12.300)
Mean SAR : 4.817 dB (5.489)
Mean NSDR: 10.014 dB (10.302)

svs_ag_tail_aug_best (fine tune):

0000_AM Contra - He:    SDR=8.098 dB,   SIR=17.310 dB,  SAR=8.733 dB,   NSDR=13.322 dB
0001_Al James - Sch:    SDR=4.194 dB,   SIR=12.302 dB,  SAR=5.172 dB,   NSDR=5.217 dB
0002_Angels In Ampl:    SDR=7.287 dB,   SIR=15.414 dB,  SAR=8.136 dB,   NSDR=10.288 dB
0003_Arise - Run Ru:    SDR=3.030 dB,   SIR=11.013 dB,  SAR=4.114 dB,   NSDR=11.274 dB
0004_BKS - Bulldoze:    SDR=-7.952 dB,  SIR=3.803 dB,   SAR=-6.139 dB,  NSDR=6.031 dB
0005_BKS - Too Much:    SDR=7.882 dB,   SIR=16.480 dB,  SAR=8.624 dB,   NSDR=12.523 dB
0006_Ben Carrigan -:    SDR=0.670 dB,   SIR=15.960 dB,  SAR=0.909 dB,   NSDR=7.569 dB
0007_Bobby Nobody -:    SDR=2.914 dB,   SIR=10.023 dB,  SAR=4.265 dB,   NSDR=8.931 dB
0008_Buitraker - Re:    SDR=1.736 dB,   SIR=7.744 dB,   SAR=3.665 dB,   NSDR=11.411 dB
0009_Carlos Gonzale:    SDR=4.096 dB,   SIR=11.169 dB,  SAR=5.364 dB,   NSDR=5.584 dB
0010_Cristina Vane :    SDR=6.216 dB,   SIR=15.263 dB,  SAR=6.921 dB,   NSDR=10.816 dB
0011_Detsky Sad - W:    SDR=2.090 dB,   SIR=8.078 dB,   SAR=3.978 dB,   NSDR=9.861 dB
0012_Enda Reilly - :    SDR=6.671 dB,   SIR=16.181 dB,  SAR=7.290 dB,   NSDR=10.539 dB
0013_Forkupines - S:    SDR=3.854 dB,   SIR=14.875 dB,  SAR=4.350 dB,   NSDR=13.621 dB
0014_Georgia Wonder:    SDR=1.364 dB,   SIR=8.998 dB,   SAR=2.701 dB,   NSDR=8.333 dB
0015_Girls Under Gl:    SDR=2.766 dB,   SIR=12.728 dB,  SAR=3.454 dB,   NSDR=13.322 dB
0016_Hollow Ground :    SDR=3.035 dB,   SIR=11.477 dB,  SAR=4.005 dB,   NSDR=13.949 dB
0017_James Elder & :    SDR=5.479 dB,   SIR=13.617 dB,  SAR=6.388 dB,   NSDR=8.653 dB
0018_Juliet's Rescu:    SDR=6.014 dB,   SIR=17.153 dB,  SAR=6.444 dB,   NSDR=14.952 dB
0019_Little Chicago:    SDR=5.227 dB,   SIR=9.388 dB,   SAR=7.801 dB,   NSDR=7.431 dB

=== Overall Mean Metrics (vocal) ===
Mean SDR : 3.734 dB  (4.349)
Mean SIR : 12.449 dB (12.904)
Mean SAR : 4.809 dB  (5.385)
Mean NSDR: 10.181 dB (10.399)
"""