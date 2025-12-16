import matplotlib.pyplot as plt
import torch
import numpy as np
import os
import sys

# --- 設定輸入檔案清單 (請在此處填入您的四個檔案路徑與標籤) ---
# 格式: ('圖例顯示名稱', '檔案路徑')
CKPT_LIST = [
    ('Vanilla', 'cKPT/svs_VNL.pth'),      # 改成您的檔案路徑
    ('Resize', 'cKPT/svs_VNL_aug.pth'),      # 改成您的檔案路徑
    ('Mix', 'CKPT/ML_M.pth'),      # 改成您的檔案路徑
    ('Resize + Mix', 'CKPT/ML_MA.pth'),   # 改成您的檔案路徑
]

# 核心修改：新增 y_min 和 y_max 參數
def plot_multi_training_losses(ckpt_list, y_min=None, y_max=None):
    plt.figure(figsize=(12, 7))
    
    # 預設顏色庫，確保線條顏色不同
    colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown']
    
    # 紀錄所有數據的 Loss 範圍，以便在沒有設定 y_min/y_max 時計算
    all_losses = []
    
    print(f"--- 開始讀取 {len(ckpt_list)} 個 Checkpoints ---")

    for i, (label, file_path) in enumerate(ckpt_list):
        if not os.path.exists(file_path):
            print(f"⚠️ Warning: 找不到檔案 {file_path}，跳過此項目。")
            continue
            
        try:
            # 載入 Checkpoint
            print(f"Loading: {label} ({file_path})...")
            checkpoint = torch.load(file_path, map_location='cpu')
            train_loss_history = checkpoint.get('valid_loss_history')

            if train_loss_history is None:
                print(f"⚠️ Warning: {label} 中沒有 'train_loss_history'，跳過。")
                continue

            # 1. 確保數據格式正確 (轉為 numpy)
            if isinstance(train_loss_history, torch.Tensor):
                train_loss_history = train_loss_history.cpu().numpy()
            elif isinstance(train_loss_history, list):
                train_loss_history = np.array(train_loss_history)
                
            # 將當前數據加入總體列表，用於計算自動範圍
            all_losses.extend(train_loss_history.tolist())
                
            # 2. 建立 X 軸座標
            # 假設 X 軸從 1 開始
            steps = range(1, len(train_loss_history) + 1)
            
            # 3. 繪製曲線
            color = colors[i % len(colors)] # 循環使用顏色
            plt.plot(steps, train_loss_history, label=label, color=color, linewidth=1.5, alpha=0.8)
            
            # (可選) 標註最後一個 Loss 值
            final_loss = train_loss_history[-1]
            print(f"   -> Final Loss: {final_loss:.5f}")

        except Exception as e:
            print(f"❌ Error loading {file_path}: {e}")

    # --- 圖表設定 ---
    plt.title('Training Loss Comparison', fontsize=16)
    plt.xlabel('Steps / Epochs', fontsize=12)
    plt.ylabel('Training Loss', fontsize=12)
    plt.legend(fontsize=16, loc='upper right') # 顯示圖例
    plt.grid(True, linestyle='--', alpha=0.5)

    # 核心修改：使用 plt.ylim() 設定自訂 Y 軸範圍
    if y_min is not None or y_max is not None:
        # 如果只設定了其中一個，另一個則使用數據的極值
        if y_min is None and all_losses:
            y_min = np.min(all_losses) * 0.95 # 底部給一點緩衝
        if y_max is None and all_losses:
            y_max = np.max(all_losses) * 1.05 # 頂部給一點緩衝
            
        plt.ylim(y_min, y_max)
    # ------------------------------------
    
    # 確保資料夾存在
    os.makedirs('LOSS_CURVE', exist_ok=True)
    
    # 存檔
    save_path = 'LOSS_CURVE/training_loss_comparison.png'
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"\n✅ Comparison plot saved to {save_path}")
    
    # 如果環境允許顯示視窗
    # plt.show()

# --- 執行分析 ---
if __name__ == "__main__":
    MIN_Y = 0.00175
    MAX_Y = 0.00250
    
    print(f"\n--- 設定繪圖範圍: Y 軸 [{MIN_Y}, {MAX_Y}] ---")
    plot_multi_training_losses(CKPT_LIST, y_min=MIN_Y, y_max=MAX_Y)
    
    # 範例：如果不設定參數，則使用預設的 None，由 Matplotlib 自動決定範圍
    # plot_multi_training_losses(CKPT_LIST)