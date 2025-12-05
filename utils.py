def num2str(n):
    if n < 10:
        return '000' + str(n)
    elif n < 100:
        return '00' + str(n)
    elif n < 1000:
        return '0' + str(n)
    else:
        return str(n)
    
INPUT_LEN = 128
SAMPLES_PER_SONG = 8