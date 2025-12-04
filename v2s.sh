# (1) 處理 TRAIN 資料集
# 請根據專案的 data.py 腳本接受的參數格式進行調整
python3 data.py \
    --src /home/m314834001/.cache/kagglehub/datasets/quanglvitlm/musdb18-hq/versions/3/train \
    --tar ./unet_spectrograms/train

# (2) 處理 VALID/TEST 資料集
python3 data.py \
    --src /home/m314834001/.cache/kagglehub/datasets/quanglvitlm/musdb18-hq/versions/3/valid \
    --tar ./unet_spectrograms/valid

# (3) 處理 TEST 資料集 (用於最終評估)
python3 data.py \
    --src /home/m314834001/.cache/kagglehub/datasets/quanglvitlm/musdb18-hq/versions/3/test \
    --tar ./unet_spectrograms/test