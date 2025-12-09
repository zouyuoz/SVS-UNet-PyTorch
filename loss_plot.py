import pandas as pd
import matplotlib.pyplot as plt
import os
import sys

# --- 設定輸入檔案名稱 ---
# 假設您的檔案名為 losses.txt
TXT_FILE_NAME = 'log_L1+SL_test.txt'

# --- 模擬數據生成結束 ---
def process_and_plot_losses(file_path):
    """
    讀取文字檔案，提取 Loss 和 Val Loss，並繪製折線圖。
    """
    if not os.path.exists(file_path):
        print(f"ERROR: File not found: {file_path}")
        sys.exit(1)
    
    train_losses = []
    val_losses = []
    val_x_indices = []
    current_x_index = 1

    # --- 數據解析與對齊 ---
    try:
        with open(file_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                if line.startswith('Val'):
                    # 提取 Val Loss
                    try:
                        val_value = float(line.replace('Val', '').strip())
                        val_losses.append(val_value)
                        
                        # Val Loss 對齊它上一個 Loss 的 X 座標
                        # current_x_index 已經在前面 Loss 處理時遞增過
                        if current_x_index > 0:
                            val_x_indices.append(current_x_index - 1)
                        else:
                            # 避免檔案開頭就是 Val Loss
                            print("WARNING: Skipped Val Loss at start of file due to missing preceding Loss.")
                            val_losses.pop() # 移除這個無效的 Val Loss
                            
                    except ValueError:
                        print(f"WARNING: Skipping invalid Val Loss entry: {line}")
                
                else:
                    # 提取 Loss (純數字)
                    try:
                        train_value = float(line)
                        train_losses.append(train_value)
                        # 每個 Loss 數據點都遞增 X 座標
                        current_x_index += 1 
                        
                    except ValueError:
                        print(f"WARNING: Skipping invalid Loss entry: {line}")

    except Exception as e:
        print(f"ERROR: An error occurred during file reading: {e}")
        sys.exit(1)
        
    if not train_losses:
        print("INFO: No valid Loss data found for plotting.")
        return

    # 確保 Val Loss 有數據點
    if not val_losses:
        print("INFO: No valid Val Loss data found, plotting Loss only.")


    # --- 繪圖設定與執行 ---
    plt.figure(figsize=(10, 6))
    
    # 創建 Loss 的 X 軸
    train_x_indices = range(len(train_losses))

    # 繪製 Loss 數據
    plt.plot(train_x_indices, train_losses, linestyle='-', color='blue', label='Train Loss', linewidth=1)
    
    # 繪製 Val Loss 數據
    if val_losses:
        # 使用 val_x_indices 確保 Val Loss 與對應的 Loss 點對齊
        plt.plot(val_x_indices, val_losses, linestyle='--', color='red', label='Val Loss', marker='o', markersize=3)
        
    # 設定圖表屬性
    plt.title('Training and Validation Loss Over Steps', fontsize=14)
    plt.xlabel('Training Step Index', fontsize=12)
    plt.ylabel('Loss Value', fontsize=12)
    plt.legend(loc='upper right')
    plt.grid(True, linestyle='--', alpha=0.6)
    
    # 設置 X 軸刻度為整數，確保清晰
    if len(train_losses) < 50:
         plt.xticks(train_x_indices)

    # 輸出結果
    plt.savefig('output.png') 

    # 打印原始數據摘要 (無中文)
    print("\n--- Data Summary ---")
    print(f"Total Train Loss points: {len(train_losses)}")
    print(f"Total Val Loss points: {len(val_losses)}")
    print("--------------------")


# --- 執行分析 ---
if __name__ == "__main__":
    Path = 'LOG/' + TXT_FILE_NAME
    process_and_plot_losses(Path)