from model import UNet
import numpy as np
import argparse
import torch
import os
from tqdm import tqdm
from utils import *

def num2str(n):
    return str(n).zfill(4)

"""
    This script defines the inference procedure of SVS-UNet

    @Author: SunnerLi
"""
# =========================================================================================
# 1. Parse the direction and related parameters
# =========================================================================================
"""
                                    Parameter Explain
    --------------------------------------------------------------------------------------------
        --model_path        The path of pre-trained model
        --mixture_folder    The root of the testing folder. You can generate via data.py 
        --tar               The folder where you want to save the splited magnitude in
    --------------------------------------------------------------------------------------------
"""
# 1. 參數設定
parser = argparse.ArgumentParser()
parser.add_argument('--model_path'    , type=str, required = True)
parser.add_argument('--tar'           , type=str, required = True)
parser.add_argument('--mixture_folder', type=str, required = True)
parser.add_argument('--vocal_solo'    , type=int,  default = 1, help="輸出頻譜圖將只有人聲")
args = parser.parse_args()

if not os.path.exists(args.tar):
    os.makedirs(args.tar, exist_ok=True)

# 2. 準備模型與設備
device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
print(f"Inference using device: {device}")

model = UNet()
model.to(device)
try:
    model.load(args.model_path)
    print(f"成功載入模型: {args.model_path}")
except Exception as e:
    print(f"載入模型失敗: {e}")
    exit(1)

model.eval()

# 3. 開始分離
with torch.no_grad():
    # 掃描 mixture 資料夾中的 spec 檔案
    files = sorted([f for f in os.listdir(args.mixture_folder) if f.endswith('_spec.npy')])
    # files = files[:10]
    print(f"找到 {len(files)} 個檔案，開始處理...")

    bar = tqdm(files)
    for name in bar:
        filepath = os.path.join(args.mixture_folder, name)
        mix = np.load(filepath) # Shape: (513, Time)
        
        # 移除 DC component 以符合模型輸入 (513 -> 512)
        mix_crop = mix[1:, :] 
        
        spec_sum = None
        
        # 滑動視窗推論 (Sliding Window Inference)
        # 每次切 INPUT_LEN 的長度丟進去 <--- 應該要跟 train.py 中的 target_len 一樣!!!!!
        seg_len = INPUT_LEN
        num_segments = (mix_crop.shape[-1] // seg_len) + 1
        
        generated_spec = []

        for i in range(num_segments):
            start = i * seg_len
            end = start + seg_len
            
            # 取出片段
            seg = mix_crop[:, start:end]
            
            # 如果是最後一段長度不夠，需要 Padding
            current_seg_len = seg.shape[1]
            if current_seg_len == 0: continue
            
            if current_seg_len < seg_len:
                pad_width = seg_len - current_seg_len
                seg_input = np.pad(seg, ((0, 0), (0, pad_width)), mode='constant')
            else:
                seg_input = seg

            # 轉 Tensor (1, 1, 512, 128)
            seg_tensor = torch.from_numpy(seg_input[np.newaxis, np.newaxis, :, :]).float().to(device)

            # 生成 人聲Mask
            msk = model(seg_tensor)
            # 如果要去除人聲，則改成 1 - msk
            if (not args.vocal_solo): msk = 1 - msk

            # 應用 Mask (Vocal = Mix * Mask)
            # 根據 train.py 的 loss 計算方式：loss = crit(msk * mix, voc)
            # 所以這裡是直接相乘
            pred_vocal_tensor = seg_tensor * msk

            # 轉回 Numpy
            pred_vocal_np = pred_vocal_tensor.squeeze().cpu().numpy() # (512, 128)

            # 如果原本有 Padding，要把多餘的部分切掉
            if current_seg_len < seg_len:
                pred_vocal_np = pred_vocal_np[:, :current_seg_len]

            generated_spec.append(pred_vocal_np)

        # 合併所有片段
        if generated_spec:
            vocal_full = np.concatenate(generated_spec, axis=1)
            
            # 把切掉的第 0 個頻率補回 0 (變成 513)
            vocal_full = np.vstack((np.zeros((1, vocal_full.shape[1]), dtype=np.float32), vocal_full))
            
            # 存檔
            save_name = name # 保持原檔名，方便 data.py 對應
            np.save(os.path.join(args.tar, save_name), vocal_full)

print("分離完成！")

"""
python inference.py \
    --model_path svs_best_1206.pth \
    --mixture_folder unet_spectrograms/test/mixture \
    --tar test_results/spec \
    --vocal_solo 1

python data.py \
    --direction to_wave \
    --src test_results/spec \
    --phase unet_spectrograms/test/mixture  \
    --tar test_results/wav \
	--hop_size 768 \
	--sr 8192

---

python data.py \
    --src custom_song \
    --tar custom_result/spec \
    --direction to_spec

python inference.py \
    --model_path svs_best_val.pth \
    --mixture_folder custom_result/spec/mixture \
    --tar custom_result/spec/rm_vocal_pred \
    --vocal_solo 0

python data.py \
    --direction to_wave \
    --src custom_result/spec/rm_vocal_pred \
    --phase custom_result/spec/mixture \
    --tar custom_result/wav

---


"""