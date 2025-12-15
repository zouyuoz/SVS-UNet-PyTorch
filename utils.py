def num2str(n):
    if n < 10:
        return '000' + str(n)
    elif n < 100:
        return '00' + str(n)
    elif n < 1000:
        return '0' + str(n)
    else:
        return str(n)

# --- 44100 Params ---
# WINDOW_SIZE = 2048
# HOP_SIZE = 512
# SAMPLE_RATE = 44100
# INPUT_LEN = 1024
# SAMPLES_PER_SONG = 64

# --- 1209 Params ---
WINDOW_SIZE = 1024
HOP_SIZE = 768
SAMPLE_RATE = 8192
INPUT_LEN = 128
SAMPLES_PER_SONG = 64
# batch_size = 32 --- IGNORE ---
# lr = 1e-4
# epoch = 500
# self.crit = DBLoss()