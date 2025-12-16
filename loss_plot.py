import pandas as pd
import matplotlib.pyplot as plt
import os
import sys
import torch

# --- 設定輸入檔案名稱 ---
label = 'MAAG'
# CKPT_FILE_NAME = f'cKPT/loss_{label}.pth'
CKPT_FILE_NAME = f'cKPT/_{label}.pth'
checkpoint = torch.load(CKPT_FILE_NAME)
train_loss_history = checkpoint.get('train_loss_history')
valid_loss_history = checkpoint.get('valid_loss_history')
print(len(train_loss_history))
print(len(valid_loss_history))

# --- 模擬數據生成結束 ---
import matplotlib.pyplot as plt
import torch
import numpy as np
import os

def plot_losses(train_loss_history, valid_loss_history):
    # 1. 確保數據是列表或 numpy array
    if isinstance(train_loss_history, torch.Tensor):
        train_loss_history = train_loss_history.cpu().numpy()
    if isinstance(valid_loss_history, torch.Tensor):
        valid_loss_history = valid_loss_history.cpu().numpy()
        
    # 避免除以零錯誤
    if len(valid_loss_history) == 0:
        print("Error: valid_loss_history is empty.")
        return

    # 2. 計算間隔 (Interval)
    interval = len(train_loss_history) // len(valid_loss_history)
    print(f"Info: Detected alignment interval: every {interval} steps.")

    # 3. 建立 X 軸座標
    train_x = range(1, len(train_loss_history) + 1)
    # Val X 軸公式: (i + 1) * interval
    val_x = [(i + 1) * interval for i in range(len(valid_loss_history))]

    # --- 新增部分：計算最小 Validation Loss 及其位置 ---
    min_val_loss = np.min(valid_loss_history)          # 找出最小值
    min_val_idx = np.argmin(valid_loss_history)        # 找出最小值的索引 (0-based)
    min_step = (min_val_idx + 1) * interval            # 換算成對應的 step/epoch
    
    print(f"✅ Min Val Loss: {min_val_loss:.6f} achieved at step {min_step}")
    # ----------------------------------------------------

    # 4. 繪圖
    plt.figure(figsize=(10, 6))
    
    # 畫 Train Loss
    plt.plot(train_x, train_loss_history, label='Train Loss', color='blue', linewidth=1, alpha=0.7)
    
    # 畫 Val Loss
    plt.plot(val_x, valid_loss_history, label='Valid Loss', color='red', marker='o', linestyle='--', markersize=3)

    # --- 新增部分：在圖上標註最小值 ---
    # 畫一個特別的點 (星號) 標示最低處
    plt.scatter(min_step, min_val_loss, color='gold', s=100, zorder=5, marker='*', edgecolors='black', label='Min Val Loss')
    
    # 添加文字註釋 (Annotate)
    # plt.annotate(f'Min: {min_val_loss:.4f}\nStep: {min_step}',
    #              xy=(min_step, min_val_loss),           # 箭頭指向的座標
    #              xytext=(min_step, min_val_loss + (max(valid_loss_history) - min_val_loss)*0.1), # 文字顯示的位置 (稍微往上偏移)
    #              arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=8),
    #              fontsize=10, ha='center',
    #              bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black", alpha=0.8))
    # ------------------------------------

    plt.title(f'{CKPT_FILE_NAME} Loss History')
    plt.xlabel('Steps / Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    
    # 確保資料夾存在
    os.makedirs('LOSS_CURVE', exist_ok=True)
    
    # 存檔
    save_path = f'LOSS_CURVE/{os.path.basename(CKPT_FILE_NAME)[:-4]}.png'
    plt.savefig(save_path)
    print(f"Plot saved to {save_path}")


# --- 執行分析 ---
if __name__ == "__main__":
    plot_losses(train_loss_history, valid_loss_history)