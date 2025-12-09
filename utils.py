def num2str(n):
    if n < 10:
        return '000' + str(n)
    elif n < 100:
        return '00' + str(n)
    elif n < 1000:
        return '0' + str(n)
    else:
        return str(n)

# --- Low Res Train Params ---
# WINDOW_SIZE = 1024
# HOP_SIZE = 768
# SAMPLE_RATE = 8192
# INPUT_LEN = 128
# SAMPLES_PER_SONG = 8

# --- 44100 Params ---
# WINDOW_SIZE = 1024
# HOP_SIZE = 256
# SAMPLE_RATE = 44100
# INPUT_LEN = 512
# SAMPLES_PER_SONG = 64

# --- Fine Tune Params ---
# WINDOW_SIZE = 1024
# HOP_SIZE = 256
# SAMPLE_RATE = 44100
# INPUT_LEN = 1536
# SAMPLES_PER_SONG = 16
# batch_size = 16
# lr = 5e-4
# self.crit = nn.L1Loss()

# --- 1207 Params ---
# WINDOW_SIZE = 1024
# HOP_SIZE = 768
# SAMPLE_RATE = 44100
# INPUT_LEN = 512
# SAMPLES_PER_SONG = 64
# batch_size = 32
# lr = 1e-4
# epoch = 500
# self.crit = nn.L1Loss()

# --- 1207 Params ---
WINDOW_SIZE = 2048
HOP_SIZE = 1024
SAMPLE_RATE = 44100
INPUT_LEN = 360
SAMPLES_PER_SONG = 64
# batch_size = 32
# lr = 1e-4
# epoch = 500
# self.crit = nn.L1Loss()