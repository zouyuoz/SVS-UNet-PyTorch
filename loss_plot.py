import pandas as pd
import matplotlib.pyplot as plt
import os
import sys
import torch

# --- 設定輸入檔案名稱 ---
CKPT_FILE_NAME = 'CKPT/svs_attn_grad_clip.pth'
checkpoint = torch.load(CKPT_FILE_NAME)
train_loss_history = checkpoint.get('train_loss_history')
valid_loss_history = checkpoint.get('valid_loss_history')
# print(len(train_loss_history))
# print(len(valid_loss_history))

# --- 模擬數據生成結束 ---
def plot_losses(train_loss_history, valid_loss_history):
    # 1. 確保數據是列表或 numpy array (如果 checkpoint 存的是 Tensor，需要轉換)
    if isinstance(train_loss_history, torch.Tensor):
        train_loss_history = train_loss_history.cpu().numpy()
    if isinstance(valid_loss_history, torch.Tensor):
        valid_loss_history = valid_loss_history.cpu().numpy()
        
    # 避免除以零錯誤
    if len(valid_loss_history) == 0:
        print("Error: valid_loss_history is empty.")
        return

    # 2. 計算間隔 (Interval)
    # 例如: Train=73, Val=7 -> interval = 10
    interval = len(train_loss_history) // len(valid_loss_history)
    
    print(f"Info: Detected alignment interval: every {interval} steps.")

    # 3. 建立 X 軸座標
    # Train X 軸: [1, 2, 3, ..., 73]
    train_x = range(1, len(train_loss_history) + 1)
    
    # Val X 軸: [10, 20, 30, ..., 70]
    # 公式: (i + 1) * interval
    val_x = [(i + 1) * interval for i in range(len(valid_loss_history))]

    # 4. 繪圖
    plt.figure(figsize=(10, 6))
    
    # 畫 Train Loss
    plt.plot(train_x, train_loss_history, label='Train Loss', color='blue', linewidth=1)
    
    # 畫 Val Loss (加上 marker 以便觀察確切落點)
    plt.plot(val_x, valid_loss_history, label='Valid Loss', color='red', marker='o', linestyle='--', markersize=3)

    plt.title(f'Loss History (Val Interval: {interval})')
    plt.xlabel('Steps / Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    
    # 如果你在 VS Code 或支援 GUI 的環境
    # plt.show()
    
    # 如果你在 WSL 無法顯示視窗，請存成圖片
    plt.savefig('output.png')
    print("Plot saved to output.png")


# --- 執行分析 ---
if __name__ == "__main__":
    plot_losses(train_loss_history, valid_loss_history)