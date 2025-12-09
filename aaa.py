import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
import librosa
from model import UNet
from utils import *

def debug_inference(model_path, spec_path):
    # 1. 準備模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading model from {model_path} on {device}...")
    
    model = UNet().to(device)
    try:
        checkpoint = torch.load(model_path, map_location=device)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load(model_path)
    except Exception as e:
        print(f"Model load failed: {e}")
        return

    model.eval()

    # 2. 準備數據 (Mixture)
    print(f"Loading mixture from {spec_path}...")
    if not os.path.exists(spec_path):
        print("Error: Mixture file not found.")
        return
    mix_input = np.load(spec_path) # Shape: (513, T)

    # 2.1 準備對應的 Ground Truth (Vocal)
    vocal_path = spec_path.replace('mixture', 'vocal') 
    
    if os.path.exists(vocal_path):
        print(f"Loading GT vocal from {vocal_path}...")
        gt_vocal_np = np.load(vocal_path)
    else:
        print(f"Warning: GT Vocal file not found at {vocal_path}. Will use zeros.")
        gt_vocal_np = np.zeros_like(mix_input)

    # 3. 滑動視窗推論 (Sliding Window Inference)
    # [修正] 明確定義 segment length，避免依賴外部常數造成混淆
    seg_len = 128 
    length = mix_input.shape[1]
    
    predicted_specs = []
    mask_specs = []  # [修正] 新增 Mask 收集列表
    
    print("Running inference on full song...")
    with torch.no_grad():
        num_segments = (length // seg_len) + 1
        
        for i in range(num_segments):
            start = i * seg_len
            end = start + seg_len
            
            # 取出片段
            seg = mix_input[:, start:end]
            
            # Padding 處理
            current_seg_len = seg.shape[1]
            if current_seg_len == 0: continue
            
            if current_seg_len < seg_len:
                pad_width = seg_len - current_seg_len
                seg_pad = np.pad(seg, ((0, 0), (0, pad_width)), mode='constant')
                input_tensor = torch.from_numpy(seg_pad[np.newaxis, np.newaxis, :, :]).float().to(device)
            else:
                input_tensor = torch.from_numpy(seg[np.newaxis, np.newaxis, :, :]).float().to(device)

            # 預測
            mask = model(input_tensor)
            pred_seg = input_tensor * mask
            
            # 轉回 Numpy
            mask_np = mask.squeeze().cpu().numpy()
            pred_seg_np = pred_seg.squeeze().cpu().numpy()
            
            # 移除 Padding 並收集
            if current_seg_len < seg_len:
                mask_np = mask_np[:, :current_seg_len]
                pred_seg_np = pred_seg_np[:, :current_seg_len]
                
            predicted_specs.append(pred_seg_np)
            mask_specs.append(mask_np) # [修正] 收集 Mask

    # 4. 拼接結果 (確保所有資料都是整首歌的長度)
    full_pred_vocal = np.concatenate(predicted_specs, axis=1)
    full_mask = np.concatenate(mask_specs, axis=1) # [修正] 拼接 Mask
    
    # 確保長度一致
    min_len = min(mix_input.shape[1], full_pred_vocal.shape[1], gt_vocal_np.shape[1], full_mask.shape[1])
    mix_show = mix_input[:, :min_len]
    gt_show = gt_vocal_np[:, :min_len]
    pred_show = full_pred_vocal[:, :min_len]
    mask_show = full_mask[:, :min_len]

    # 5. 數據後處理 (修改為 dB 差異)
    # [修正] 統一使用 mix 的最大值做 ref，避免靜音段爆音
    ref_value = np.max(mix_show) + 1e-8

    gt_vocal_db = librosa.amplitude_to_db(gt_show, ref=ref_value, amin=1e-5)
    pred_vocal_db = librosa.amplitude_to_db(pred_show, ref=ref_value, amin=1e-5)
    mix_db = librosa.amplitude_to_db(mix_show, ref=ref_value, amin=1e-5)

    # 計算 dB 差異
    diff_db = pred_vocal_db - gt_vocal_db

    # 6. 畫圖
    fig = plt.figure(figsize=(15, 6))
    gs = fig.add_gridspec(3, 1)

    aspect_ratio = 'auto'
    origin_set = 'lower'
    db_vmin, db_vmax = -80, 0
    X_MAX = gt_vocal_db.shape[1]
    
    # # --- 1. Mixture ---
    # ax1 = fig.add_subplot(gs[0, 0])
    # ax1.set_title(f"1. Mixture (Input) - Full Song") # [修正] 標題改為 Full Song
    # im1 = ax1.imshow(mix_db, aspect=aspect_ratio, origin=origin_set, cmap='magma', vmin=db_vmin, vmax=db_vmax)
    # plt.colorbar(im1, ax=ax1, format='%+2.0f dB')

    # --- 2. GT Vocal ---
    ax2 = fig.add_subplot(gs[0, 0])
    ax2.set_title("2. True Vocal (Target)")
    im2 = ax2.imshow(gt_vocal_db, aspect=aspect_ratio, origin=origin_set, cmap='magma', vmin=db_vmin, vmax=db_vmax)
    plt.colorbar(im2, ax=ax2, format='%+2.0f dB')

    # --- 4. Predicted Vocal ---
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.set_title("4. Predicted Vocal (Result)")
    im3 = ax3.imshow(pred_vocal_db, aspect=aspect_ratio, origin=origin_set, cmap='magma', vmin=db_vmin, vmax=db_vmax)
    plt.colorbar(im3, ax=ax3, format='%+2.0f dB')

    # # --- 3. Mask ---
    # ax4 = fig.add_subplot(gs[1, 0])
    # ax4.set_title("3. Generated Mask (Concatenated)") # [修正] 標示為拼接後的 Mask
    # # [修正] 這裡改用 mask_show (整首歌) 而不是 mask_img (最後一段)
    # im4 = ax4.imshow(mask_show, aspect=aspect_ratio, origin=origin_set, cmap='gray', vmin=0, vmax=1)
    # plt.colorbar(im4, ax=ax4)
    # ax4.text(5, 50, f"Avg: {mask_show.mean():.3f}", color='yellow', fontweight='bold')

    # --- 5. Difference Map (dB) ---
    ax5 = fig.add_subplot(gs[2, 0])
    ax5.set_title("5. Difference in dB (Pred - True) | Red: Noise (Too Loud) | Blue: Loss (Too Quiet)")
    
    diff_range = 40 
    im5 = ax5.imshow(diff_db, aspect=aspect_ratio, origin=origin_set, cmap='berlin', vmin=-diff_range, vmax=diff_range)
    plt.colorbar(im5, ax=ax5, format='%+2.0f dB')
    
    tick_locations = np.arange(0, X_MAX, 60*SAMPLE_RATE/HOP_SIZE)
    tick_labels = (tick_locations / (60*SAMPLE_RATE/HOP_SIZE)).astype(int) 
    
    # 3. 應用到每個 subplot
    for ax in [ax2, ax3, ax5]:
        ax.set_xticks(tick_locations)
        if ax == ax5: ax.set_xticklabels(tick_labels)
        else: ax.set_xticklabels([])
    plt.tight_layout()
    
    # 存檔
    filename = os.path.basename(spec_path).replace('.npy', '')
    output_path = f'viz_{filename}.png'
    plt.savefig(output_path)
    print(f"Visualization saved to: {output_path}")
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='CKPT/svs_500.pth')
    parser.add_argument('--spec_path' , type=str, required=True, help="Path to the MIXTURE spectrogram")
    args = parser.parse_args()
    
    debug_inference(args.model_path, args.spec_path)

"""
python aaa.py --spec_path "unet_spectrograms/test/mixture/0007_Bobby Nobody - Stitch Up_spec.npy"
python aaa.py --spec_path "unet_spectrograms/train/mixture/0007_Aimee Norwich - Child_spec.npy"
python aaa.py --spec_path "unet_spectrograms_high/test/mixture/0007_Bobby Nobody - Stitch Up_spec.npy"
"""