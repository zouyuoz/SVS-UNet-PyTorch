import numpy as np
import matplotlib.pyplot as plt
import os
import librosa
import librosa.display

def save_raw_image(npy_path: str, output_path: str):
    """將 .npy 矩陣儲存為沒有軸的圖片"""
    spec = np.load(npy_path)
    
    # 1. 轉換為 dB 尺度 (此步驟保持不變，可視化效果最好)
    S_dB = librosa.amplitude_to_db(spec, ref=np.max) if 'librosa' in globals() else spec
    
    plt.figure()
    plt.imshow(S_dB, aspect='auto', origin='lower', cmap='magma')
    
    # 關鍵步驟：移除所有軸、間距和邊框
    plt.axis('off')
    plt.gca().set_position([0, 0, 1, 1]) # 讓圖片佔滿整個 Figure 區域
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
    plt.close()

# 範例執行
example_npy_path = "unet_spectrograms_high/test/vocal/0007_Bobby Nobody - Stitch Up_spec.npy"
save_raw_image(example_npy_path, 'custom_result/image/gt_1207_0007.png')


example_npy_path = "test_results/spec/0007_Bobby Nobody - Stitch Up_spec.npy"
example_npy_path = "unet_spectrograms/test/vocal/0000_AM Contra - Heart Peripheral_spec.npy"
example_npy_path = "unet_spectrograms_high/test/mixture/0007_Bobby Nobody - Stitch Up_spec.npy"