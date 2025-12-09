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
    mix_np = np.load(spec_path) # Shape: (513, T)

    # 2.1 準備對應的 Ground Truth (Vocal)
    # 假設路徑結構為 .../mixture/xxxx_spec.npy -> .../vocal/xxxx_spec.npy
    # 如果您的資料夾名稱是 vocals (複數)，請將下面的 'vocal' 改為 'vocals'
    vocal_path = spec_path.replace('mixture', 'vocal') 
    
    if os.path.exists(vocal_path):
        print(f"Loading GT vocal from {vocal_path}...")
        gt_vocal_np = np.load(vocal_path)
    else:
        print(f"Warning: GT Vocal file not found at {vocal_path}. Will use zeros.")
        gt_vocal_np = np.zeros_like(mix_np)

    # 3. 數據裁切 (Preprocessing)
    total_len = mix_np.shape[1]
    target_len = INPUT_LEN
    
    # 決定裁切起始點 (取中間)
    if total_len > target_len:
        start = total_len // 2 
        # start = np.random.randint(0, total_len - target_len)
        
        # 同步裁切
        mix_crop = mix_np[1:, start : start + target_len]
        gt_vocal_crop = gt_vocal_np[1:, start : start + target_len]
    else:
        # 補零
        pad_width = target_len - total_len
        pad = np.zeros((513, pad_width))
        mix_np = np.concatenate([mix_np, pad], axis=1)
        gt_vocal_np = np.concatenate([gt_vocal_np, pad], axis=1)
        
        mix_crop = mix_np[1:, :target_len]
        gt_vocal_crop = gt_vocal_np[1:, :target_len]

    # 轉 Tensor
    input_tensor = torch.from_numpy(mix_crop[np.newaxis, np.newaxis, :, :]).float().to(device)

    # 4. 模型預測
    with torch.no_grad():
        mask = model(input_tensor)
        pred_vocal = input_tensor * mask

    # 5. 數據後處理 (修改為 dB 差異)
    mix_img = mix_crop
    gt_vocal_img = gt_vocal_crop
    mask_img = mask.squeeze().cpu().numpy()
    pred_vocal_img = pred_vocal.squeeze().cpu().numpy()
    
    # [關鍵修正 1] 統一使用 Mixture 的最大值作為 dB 轉換基準
    # 這樣可以確保 Pred 如果很小聲，轉出來的 dB 也會很小，不會被強行拉大
    ref_value = np.max(mix_img) + 1e-8

    # 轉 dB (設定 amin 防止 log(0) 變成 -inf)
    # 這裡我們用比較寬的 dynamic range (-80dB)
    gt_vocal_db = librosa.amplitude_to_db(gt_vocal_img, ref=ref_value, amin=1e-5)
    pred_vocal_db = librosa.amplitude_to_db(pred_vocal_img, ref=ref_value, amin=1e-5)
    mix_db = librosa.amplitude_to_db(mix_img, ref=ref_value, amin=1e-5)

    # [關鍵修正 2] 計算 dB 差異
    # Diff (dB) = Pred (dB) - GT (dB)
    # 紅色 = 預測太吵 (雜訊)
    # 藍色 = 預測太安靜 (人聲缺失)
    diff_db = pred_vocal_db - gt_vocal_db

    # 6. 畫圖
    fig = plt.figure(figsize=(18, 10))
    gs = fig.add_gridspec(2, 4)

    aspect_ratio = 'auto'
    origin_set = 'lower'
    db_vmin, db_vmax = -80, 0
    
    # --- 1. Mixture ---
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.set_title(f"1. Mixture (Input)\nTime: {start/44100*768:.1f}s")
    im1 = ax1.imshow(mix_db, aspect=aspect_ratio, origin=origin_set, cmap='magma', vmin=db_vmin, vmax=db_vmax)
    plt.colorbar(im1, ax=ax1, format='%+2.0f dB')

    # --- 2. GT Vocal ---
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.set_title("2. True Vocal (Target)")
    im2 = ax2.imshow(gt_vocal_db, aspect=aspect_ratio, origin=origin_set, cmap='magma', vmin=db_vmin, vmax=db_vmax)
    plt.colorbar(im2, ax=ax2, format='%+2.0f dB')

    # --- 4. Predicted Vocal ---
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.set_title("4. Predicted Vocal (Result)")
    im3 = ax3.imshow(pred_vocal_db, aspect=aspect_ratio, origin=origin_set, cmap='magma', vmin=db_vmin, vmax=db_vmax)
    plt.colorbar(im3, ax=ax3, format='%+2.0f dB')

    # --- 3. Mask ---
    ax4 = fig.add_subplot(gs[1, 0])
    ax4.set_title("3. Generated Mask")
    im4 = ax4.imshow(mask_img, aspect=aspect_ratio, origin=origin_set, cmap='gray', vmin=0, vmax=1)
    plt.colorbar(im4, ax=ax4)
    ax4.text(5, 50, f"Avg: {mask_img.mean():.3f}", color='yellow', fontweight='bold')

    # --- 5. Difference Map (dB) ---
    ax5 = fig.add_subplot(gs[0:, 2:])
    ax5.set_title("5. Difference in dB (Pred - True)\nRed: Noise (Too Loud) | Blue: Loss (Too Quiet)")
    
    # 設定差異範圍：+/- 40dB 已經是非常巨大的差異了，超過這個範圍就顯示全紅/全藍
    diff_range = 40 
    im5 = ax5.imshow(diff_db, aspect=aspect_ratio, origin=origin_set, cmap='berlin', vmin=-diff_range, vmax=diff_range)
    plt.colorbar(im5, ax=ax5, format='%+2.0f dB')

    plt.tight_layout()
    
    # 存檔
    filename = os.path.basename(spec_path).replace('.npy', '')
    output_path = f'viz_{filename}.png'
    plt.savefig(output_path)
    print(f"Visualization saved to: {output_path}")
    plt.close() # 關閉圖形釋放記憶體

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='CKPT/svs_best_1207.pth')
    parser.add_argument('--spec_path', type=str, required=True, help="Path to the MIXTURE spectrogram")
    args = parser.parse_args()
    
    debug_inference(args.model_path, args.spec_path)
"""
python aaa.py --spec_path "unet_spectrograms_high/test/mixture/0007_Bobby Nobody - Stitch Up_spec.npy"
python aaa.py --spec_path "test_results/spec/0007_Bobby Nobody - Stitch Up_spec.npy"
"""