import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
import librosa  # [新增] 用於 dB 轉換
from model import UNet

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

    # 2. 準備數據
    print(f"Loading spectrogram from {spec_path}...")
    if not os.path.exists(spec_path):
        print("Error: Spectrogram file not found.")
        return

    mix_np = np.load(spec_path) # Shape: (513, T)
    
    # [修正點 1] 改從歌曲 "中間" 取樣，避開開頭靜音
    total_len = mix_np.shape[1]
    target_len = 128
    
    if total_len > target_len:
        start = total_len // 2  # 取中間
        # start = np.random.randint(0, total_len - target_len) # 或是隨機
        input_crop = mix_np[1:, start : start + target_len]
    else:
        # 如果歌太短才補零
        pad = np.zeros((513, target_len - total_len))
        mix_np = np.concatenate([mix_np, pad], axis=1)
        input_crop = mix_np[1:, :target_len]

    # Shape check: (512, 128)
    
    # 轉 Tensor
    input_tensor = torch.from_numpy(input_crop[np.newaxis, np.newaxis, :, :]).float().to(device)

    # 3. 預測
    with torch.no_grad():
        mask = model(input_tensor)
        pred_vocal = input_tensor * mask

    # 轉回 Numpy
    input_img = input_crop
    mask_img = mask.squeeze().cpu().numpy()
    pred_img = pred_vocal.squeeze().cpu().numpy()

    # [修正點 2] 視覺化轉為 dB 單位
    # 加上一個極小值 1e-7 避免 log(0)
    input_img_db = librosa.amplitude_to_db(input_img, ref=np.max)
    pred_img_db = librosa.amplitude_to_db(pred_img, ref=np.max)

    # 4. 畫圖診斷
    plt.figure(figsize=(15, 6))

    # Input Spectrogram (dB)
    plt.subplot(1, 3, 1)
    plt.title(f"Input Mixture (dB)\nTime: {start/44100*768:.1f}s") # 顯示大概秒數
    # vmin=-80 代表顯示範圍到底部 -80dB，讓細節更明顯
    plt.imshow(input_img_db, aspect='auto', origin='lower', cmap='magma', vmin=-80, vmax=0)
    plt.colorbar(format='%+2.0f dB')

    # Generated Mask
    plt.subplot(1, 3, 2)
    plt.title("Generated Mask (0.0 ~ 1.0)")
    plt.imshow(mask_img, aspect='auto', origin='lower', cmap='gray', vmin=0, vmax=1)
    plt.colorbar()
    # 顯示 Mask 的統計值，確認不是全黑或全白
    plt.text(5, 5, f"Mean: {mask_img.mean():.4f}\nMax: {mask_img.max():.4f}", color='yellow', fontweight='bold')

    # Predicted Vocal (dB)
    plt.subplot(1, 3, 3)
    plt.title("Predicted Vocal (dB)")
    plt.imshow(pred_img_db, aspect='auto', origin='lower', cmap='magma', vmin=-80, vmax=0)
    plt.colorbar(format='%+2.0f dB')

    plt.tight_layout()
    plt.savefig('debug_mask_result_fixed.png')
    print("Diagnosis complete! Please check 'debug_mask_result_fixed.png'.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='svs_400.pth')
    parser.add_argument('--spec_path', type=str, required=True)
    args = parser.parse_args()
    
    debug_inference(args.model_path, args.spec_path)
"""
python aaa.py --spec_path "unet_spectrograms_high/test/mixture/0007_Bobby Nobody - Stitch Up_spec.npy"
"""