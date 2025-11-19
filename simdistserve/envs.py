import os
SKIP_PREFILL = bool(int(os.getenv('SKIP_PREFILL', 0)))
SKIP_DECODE = bool(int(os.getenv('SKIP_DECODE', 0)))
GPU_FREQ = int(os.getenv('GPU_FREQ', '1830'))

def get_gpu_freq():
    global GPU_FREQ
    return GPU_FREQ

def set_gpu_freq(value):
    global GPU_FREQ
    GPU_FREQ = value

def get_skip_prefill():
    global SKIP_PREFILL
    return SKIP_PREFILL

def set_skip_prefill(value):
    global SKIP_PREFILL
    SKIP_PREFILL = value

def get_skip_decode():
    global SKIP_DECODE
    return SKIP_DECODE

def set_skip_decode(value):
    global SKIP_DECODE
    SKIP_DECODE = value

OVERWRITE_PREFILL_LEN = os.getenv('OVERWRITE_PREFILL_LEN')
if OVERWRITE_PREFILL_LEN:
    OVERWRITE_PREFILL_LEN = int(OVERWRITE_PREFILL_LEN)

OVERWRITE_DECODE_LEN = os.getenv('OVERWRITE_DECODE_LEN')
if OVERWRITE_DECODE_LEN:
    OVERWRITE_DECODE_LEN = int(OVERWRITE_DECODE_LEN)

PRINT_EXCEPT_MSG = bool(int(os.getenv('PRINT_EXCEPT_MSG', '0')))

SCALE_ARRIVAL_TIME = bool(int(os.getenv('SCALE_ARRIVAL_TIME', '0')))
LIMIT_NUM_REQ = bool(int(os.getenv('LIMIT_NUM_REQ', '0')))

OPTIMIZE_ENERGY = bool(int(os.getenv('OPTIMIZE_POWER', '0')))

IGNORE_FIRST_AND_LAST_QUARTER = bool(int(os.getenv('IGNORE_FIRST_AND_LAST_QUARTER', '0')))