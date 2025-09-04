import os
SKIP_PREFILL = bool(int(os.getenv('SKIP_PREFILL', 0)))
SKIP_DECODE = bool(int(os.getenv('SKIP_DECODE', 0)))
GPU_FREQ = int(os.getenv('GPU_FREQ', '0'))
