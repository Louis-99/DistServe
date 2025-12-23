import os
import time
from multiprocessing import Process, Manager, Lock
from time import sleep
import math

import pandas as pd
from tqdm import tqdm
import numpy as np

from simdistserve.benchmarks.search_configs import get_distserve_configs, get_vllm_config
from simdistserve.constants import ModelTypes

from simdistserve.envs import set_gpu_freq, set_skip_prefill, set_skip_decode
from simdistserve.benchmarks.simulate_dist import run_experiment, parse_args


# Restrict runtime to <= 32 CPU core.
# RunPod encounters problem when using `os.cpu_count()`
# to query the number of CPUs
MAX_CPU_COUNT = min(os.cpu_count() - 2, int(os.getenv('MAX_CPU_COUNT', 32)))

GOODPUT_DICT_UPDATE_ALPHA = 0.5
# CONFIG_N_SCALE = 2
CONFIG_N_SCALE = 1


def run_binary_search(
    model_type: ModelTypes,
    containment_targets: tuple, # (prefill_target, decode_target, prefill_containment, decode_containment)
    high: int = 50,
    low: int = 0,
    pid=0,
    esp=0.5,
    N=1000,
    seed: int = 0,
    workload: str = 'sharegpt',
    freq: int = 1830,
    tp_prefill: int = 2,
    tp_decode: int = 4,
    num_prefill: int = 4,
    num_decode: int = 2,
):
    
    N = str(N)

    tp_prefill = max(tp_prefill, 2)
    tp_decode = max(tp_decode, 2)    

    config_args = [
        '--tp-prefill', f'{tp_prefill}',
        '--pp-prefill', f'1',
        '--n-prefill', f'{num_prefill}',
        '--tp-decode', f'{tp_decode}',
        '--pp-decode', f'1',
        '--n-decode', f'{num_decode}',
    ]

    num_gpu = tp_prefill * num_prefill + tp_decode * num_decode

    prefill_target, decode_target, prefill_containment, decode_containment = containment_targets

    fixed_args = [
        '--arrival', 'poisson',
        '--seed', str(int(seed)),
        '--N', N,
        '--prefill-containment', prefill_containment,  # P90
        '--prefill-target', prefill_target,  # ms
        '--decode-containment', decode_containment,  # P90
        '--decode-target', decode_target,  # ms
        '--model', ModelTypes.formalize_model_name(model_type),
        '--workload', workload,
        '--slas', '[]',
        '--slo-scales', '[1]',
        '--backend', 'distserve',
        '--prefill-freq', freq,
        '--decode-freq', freq,
    ]

    first_run = True
    prefill_energy = None
    decode_energy = None
    while (high - low) > esp or first_run:
        first_run = False
        # Run simulation
        this_rate = (low + high) / 2
        print(f'start: {this_rate=}, {low=}, {high=}', flush=True)
        rate = this_rate * num_gpu
        args = [*fixed_args, *config_args, '--rate', rate, ]
        args = [str(i) for i in args]
        args = parse_args(args)
        try:
            is_prefill_contained, is_decode_contained, prefill_energy, decode_energy, df = run_experiment(args)
        except Exception as e:
            import traceback
            print(
                f"({pid=}) Error when computing f{tp_prefill=} f{tp_decode=} f{freq=}. This may not be a real error "
                f"(e.g. bad parallelism strategy). Exception detail: {traceback.format_exc()}."
            )
            return None

        # Update the range
        success = is_prefill_contained and is_decode_contained
        if not success:
            # The experiment not passing the attainment threshold
            high = this_rate
            best_per_gpu_rate = (high + low) / 2
        else:
            # Experiment passed the attainment threshold
            low = this_rate
            best_per_gpu_rate = (high + low) / 2
        print(f'{this_rate=}, {low=}, {high=}', flush=True)
        
        
    total_energy = prefill_energy + decode_energy
    return best_per_gpu_rate, total_energy / num_gpu


    

if __name__ == '__main__':
    n_init = 200_000

    workload_list = [
        # "/export1/liu3882/llm_energy/vllm_script/trace/azure_code_arrival/azure_2024_code_sharegpt-ctx-len_qps40_req-cnt144000.csv",
        # "/export1/liu3882/llm_energy/vllm_script/trace/azure_code_arrival/azure_2024_code_sharegpt-ctx-len_qps107_req-cnt200000.csv",
        # "/export1/liu3882/llm_energy/vllm_script/trace/azure_code_arrival_scale_correctly/azure_2024_code_sharegpt-ctx-len_qps93.46_5min-0.csv",
        # "/export1/liu3882/llm_energy/vllm_script/trace/azure_code_arrival_scale_correctly/azure_2024_code_sharegpt-ctx-len_qps93.46_5min-1.csv",
        # "/export1/liu3882/llm_energy/vllm_script/trace/azure_code_arrival_scale_correctly/azure_2024_code_sharegpt-ctx-len_qps93.46_5min-2.csv",
        # "/export1/liu3882/llm_energy/vllm_script/trace/azure_code_arrival_scale_correctly/azure_2024_code_sharegpt-ctx-len_qps93.46_5min-3.csv",
        # "/export1/liu3882/llm_energy/vllm_script/trace/azure_code_arrival_scale_correctly/azure_2024_code_sharegpt-ctx-len_qps93.46_5min-4.csv",
        # "/export1/liu3882/llm_energy/vllm_script/trace/azure_code_arrival_scale_correctly/azure_2024_code_sharegpt-ctx-len_qps93.46_5min-5.csv",
        # "/export1/liu3882/llm_energy/vllm_script/trace/azure_code_arrival_scale_correctly/azure_2024_code_sharegpt-ctx-len_qps93.46_5min-6.csv",

        "/export1/liu3882/llm_energy/vllm_script/trace/azure_code_arrival_scale_correctly/azure_2024_code_sharegpt-ctx-len_qps93.46_req-cnt20000_chunk-0.csv",
        # "/export1/liu3882/llm_energy/vllm_script/trace/azure_code_arrival_scale_correctly/azure_2024_code_sharegpt-ctx-len_qps93.46_req-cnt20000_chunk-1.csv",
        # "/export1/liu3882/llm_energy/vllm_script/trace/azure_code_arrival_scale_correctly/azure_2024_code_sharegpt-ctx-len_qps93.46_req-cnt20000_chunk-2.csv",
        # "/export1/liu3882/llm_energy/vllm_script/trace/azure_code_arrival_scale_correctly/azure_2024_code_sharegpt-ctx-len_qps93.46_req-cnt20000_chunk-3.csv",
        # "/export1/liu3882/llm_energy/vllm_script/trace/azure_code_arrival_scale_correctly/azure_2024_code_sharegpt-ctx-len_qps93.46_req-cnt20000_chunk-4.csv",
        # "/export1/liu3882/llm_energy/vllm_script/trace/azure_code_arrival_scale_correctly/azure_2024_code_sharegpt-ctx-len_qps93.46_req-cnt20000_chunk-5.csv",
        # "/export1/liu3882/llm_energy/vllm_script/trace/azure_code_arrival_scale_correctly/azure_2024_code_sharegpt-ctx-len_qps93.46_req-cnt20000_chunk-6.csv",
        # "/export1/liu3882/llm_energy/vllm_script/trace/azure_code_arrival_scale_correctly/azure_2024_code_sharegpt-ctx-len_qps93.46_req-cnt20000_chunk-7.csv",
        # "/export1/liu3882/llm_energy/vllm_script/trace/azure_code_arrival_scale_correctly/azure_2024_code_sharegpt-ctx-len_qps93.46_req-cnt20000_chunk-8.csv",
        # "/export1/liu3882/llm_energy/vllm_script/trace/azure_code_arrival_scale_correctly/azure_2024_code_sharegpt-ctx-len_qps93.46_req-cnt20000_chunk-9.csv",
    ]

    # TP_PREFILL, NUM_PREFILL, TP_DECODE, NUM_DECODE
    placements = [
        (2, 4, 4, 2),
        (4, 2, 4, 2),
    ]

    for i, workload in enumerate(workload_list):
        print(f"{workload=}")
        for placement in placements:
            tp_prefill, num_prefill, tp_decode, num_decode = placement
            print(f"  {placement=}")
            result = run_binary_search(
                model_type=ModelTypes.llama3_70b, 
                containment_targets=(600, 100, 99, 99), 
                high=30,
                low=0,
                esp=0.05, 
                N=n_init, 
                workload=workload,
                freq=1830,
                tp_prefill=tp_prefill,
                num_prefill=num_prefill,
                tp_decode=tp_decode,
                num_decode=num_decode,
            )
            print(result)
    
