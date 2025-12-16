import librosa
import soundfile as sf

def clip_audio(input_file, output_file, start_sec, end_sec):
    """
    載入音訊檔案，擷取特定秒數區段，並儲存為新的 WAV 檔案。

    Args:
        input_file (str): 原始音訊檔案路徑。
        output_file (str): 輸出音訊檔案路徑。
        start_sec (float): 擷取起始時間（秒）。
        end_sec (float): 擷取結束時間（秒）。
    """
    print(f"載入檔案: {input_file}...")
    
    # 載入整個音訊檔案，y 是音訊數據，sr 是取樣率
    y, sr = librosa.load(input_file, sr=None) 
    
    # 計算起始和結束時間點的取樣索引 (Index)
    start_frame = int(start_sec * sr)
    end_frame = int(end_sec * sr)
    
    # 擷取音訊數據 (NumPy array 切片)
    y_clip = y[start_frame:end_frame]
    
    # 儲存擷取後的音訊數據
    sf.write(output_file, y_clip, sr, subtype='PCM_16')
    print(f"成功擷取 {start_sec:.2f}s 到 {end_sec:.2f}s 的音訊，儲存為 {output_file}")


# --- 執行範例 ---
INPUT_FILE = [
    "不如別人太",
    "AT",
    "LastChristmas",
    "LastChristmasTS",
    "Leader",
    "Maple",
    "TC",
    "Shouldnt"
]
START_TIME = [46, 120+35, 60+25, 13, 60+27, 120+39, 29, 115]
END_TIME   = [64, 180,    60+48, 35, 120+6, 180+50, 49, 157]

# 執行函式
for index in range(len(START_TIME)):
    clip_audio(
        "custom_result/wav_vocal/" + INPUT_FILE[index] + ".wav",
        "RESULTS/vocal/" + INPUT_FILE[index] + ".wav",
        START_TIME[index],
        END_TIME[index])

for index in range(len(START_TIME)):
    clip_audio(
        "custom_result/wav_accomp/" + INPUT_FILE[index] + ".wav",
        "RESULTS/accomp/" + INPUT_FILE[index] + ".wav",
        START_TIME[index],
        END_TIME[index])