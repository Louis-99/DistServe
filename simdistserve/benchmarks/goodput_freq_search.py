import os
import time
from multiprocessing import Process, Manager
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
    config: tuple, # (tp_prefill, tp_decode, freq)
    containment_targets: tuple, # (prefill_target, decode_target, prefill_containment, decode_containment)
    high: int = 50,
    low: int = 0,
    pid=0,
    esp=0.5,
    N=1000,
    result: None|dict =None,
    low_dict: None|dict = None,
    high_dict: None|dict = None,
    seed: int = 0,
):
    N = str(N)

    tp_prefill, tp_decode, freq = config

    skip_decode = (tp_decode <= 0)
    skip_prefill = (tp_prefill <= 0)
    set_gpu_freq(freq)
    set_skip_decode(skip_decode)
    set_skip_prefill(skip_prefill)

    tp_prefill = max(tp_prefill, 2)
    tp_decode = max(tp_decode, 2)    

    config_args = [
        '--tp-prefill', f'{tp_prefill}',
        '--pp-prefill', f'1',
        '--tp-decode', f'{tp_decode}',
        '--pp-decode', f'1',
    ]

    assert not (skip_decode and skip_prefill)
    if skip_decode: # prefill only
        num_gpu = tp_prefill
    elif skip_prefill: # decode only
        num_gpu = tp_decode
    else:
        num_gpu = tp_prefill + tp_decode

    prefill_target, decode_target, prefill_containment, decode_containment = containment_targets

    if low_dict is not None:
        for (tp_p, tp_d, f), prev_rate in low_dict.copy().items():
            if tp_p == config[0] and tp_d == config[1]:
                if f < freq and prev_rate > low:
                    low = prev_rate

    if high_dict is not None:
        for (tp_p, tp_d, f), prev_rate in high_dict.copy().items():
            if tp_p == config[0] and tp_d == config[1]:
                if f > freq and prev_rate < high:
                    high = prev_rate

    # print(f'{config=} {low=:.2f} {high=:.2f}')


    best_per_gpu_rate = 0

    fixed_args = [
        '--arrival', 'poisson',
        '--seed', str(int(seed)),
        '--N', N,
        '--prefill-containment', prefill_containment,  # P90
        '--prefill-target', prefill_target,  # ms
        '--decode-containment', decode_containment,  # P90
        '--decode-target', decode_target,  # ms
        '--model', ModelTypes.formalize_model_name(model_type),
        '--workload', 'sharegpt',
        '--slas', '[]',
        '--slo-scales', '[1]',
        '--backend', 'distserve',
        '--freq', freq,
    ]

    first_run = True
    prefill_energy = None
    decode_energy = None
    while (high - low) > esp or first_run:
        first_run = False
        # Run simulation
        this_rate = (low + high) / 2
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
            high_dict[config] = this_rate
            continue

        # Experiment passed the attainment threshold
        low = this_rate
        low_dict[config] = this_rate
        best_per_gpu_rate = this_rate
        pass
    if result is not None:
        assert prefill_energy is not None
        assert decode_energy is not None
        if skip_decode:
            total_energy = prefill_energy
        elif skip_prefill:
            total_energy = decode_energy
        else:
            total_energy = prefill_energy + decode_energy
        result[config] = (best_per_gpu_rate, total_energy) 
    return best_per_gpu_rate

def generate_configs_dict():
    freq_list = np.arange(360, 1830+1, 30).tolist()
    # freq_list = np.arange(780, 1830+1, 75).tolist()
    # freq_list = np.arange(780, 1830+1, 150).tolist()
    # freq_list = np.arange(780, 1830+1, 300).tolist()
    freq_idx = [0, len(freq_list)-1]

    while len(freq_idx) < len(freq_list):
        freq_idx_copy = freq_idx.copy()
        freq_idx_copy.sort()
        for low, high in zip(freq_idx_copy[:-1], freq_idx_copy[1:]):
            if high - low == 1:
                continue
            freq_idx.append((low + high) // 2)
    freq_list = [freq_list[i] for i in freq_idx]

    configs_dict = {f'config{i}': [] for i in range(4)}
    for freq in freq_list:
        configs_dict['config0'].append((2, 0, freq))
        configs_dict['config1'].append((4, 0, freq))
        configs_dict['config2'].append((0, 2, freq))
        configs_dict['config3'].append((0, 4, freq))
    return configs_dict


def main(
    model_type: ModelTypes,
    backend: str = "distserve", 
    attainment=(200, 100, 90, 90),
    max_per_gpu_rate=5, esp=0.25, N=1000,
    max_cpu_count=MAX_CPU_COUNT,
    goodput_dict: None|dict[tuple, float] = None,
    target_goodput: None|float = None,
    seed: int = 0,
):
    """
    :return result: dict that maps config to the best_per_gpu_rate (int)
    """
    assert backend == "distserve"

    assert target_goodput is not None or goodput_dict is None

    configs_dict = generate_configs_dict()
    processes_dict: dict[str, list[Process]] = {key: [] for key in configs_dict.keys()}

    # Add a multiproc shared dict
    with Manager() as manager:
        result = manager.dict()
        low_dict = manager.dict()
        high_dict = manager.dict()
        with tqdm(total=sum(map(len, configs_dict.values()))) as pbar:
            while len(configs_dict) > 0:
                for key, configs in configs_dict.copy().items():
                    if len(configs) == 0:
                        del configs_dict[key]
                        continue

                    while True:
                        for processes in processes_dict.values():
                            for p in processes.copy():
                                if not p.is_alive():
                                    p.join()
                                    processes.remove(p)
                                    pbar.update(1)

                        if sum(map(len, processes_dict.values())) < max_cpu_count:
                            break
                        else:
                            sleep(0.2)

                    if len(processes_dict[key]) >= math.ceil(max_cpu_count / len(configs_dict)):
                        continue

                    config = configs.pop(0)
                    # number of requests needed for this simulation is dependent on its pass goodput
                    config_N = N
                    if goodput_dict is not None and config in goodput_dict.keys():
                        old_goodput = goodput_dict[config]
                        config_N = int(min(1, CONFIG_N_SCALE * old_goodput / target_goodput) * N)
                        config_N = max(500, config_N)
                    # print(f'{config=} {config_N=}')

                    proc = Process(
                        target=run_binary_search,
                        args=(
                            model_type, config,
                            attainment,
                        ),
                        kwargs=dict(
                            high=max_per_gpu_rate,
                            esp=esp,
                            N=config_N, result=result,
                            low_dict=low_dict,
                            high_dict=high_dict,
                            seed=seed,
                        )
                    )
                    proc.start()
                    processes_dict[key].append(proc)    
             
            for processes in processes_dict.values():
                for p in processes:
                    pbar.update(1)      
                    p.join()
        result = dict(result)

        # update goodput dict
        if goodput_dict is not None:
            for config, (goodput, _) in result.items():
                goodput *= config[0] + config[1]
                if config in goodput_dict.keys():
                    goodput_dict[config] += GOODPUT_DICT_UPDATE_ALPHA * (goodput - goodput_dict[config])
                else:
                    goodput_dict[config] = goodput
        
        return result
    

if __name__ == '__main__':
    target_goodput = 22
    # n_init = 4000
    n_init = 10000
    # input 4k
    # print('Begin test with N=4k')
    # my_goodput_dict = {}
    # for i in range(5):
    #     start_time = time.perf_counter()
    #     result = main(
    #         model_type=ModelTypes.llama3_70b, 
    #         attainment=(600, 100, 95, 95), 
    #         max_per_gpu_rate=20, 
    #         esp=0.05, 
    #         N=n_init if i == 0 else 4000, 
    #         max_cpu_count=28,
    #         goodput_dict=my_goodput_dict,
    #         target_goodput=target_goodput,
    #         seed=i,
    #     )
    #     end_time = time.perf_counter()
    #     print(f'total time for {i}-th run is {end_time-start_time:.3f}s')
    #     print(f'{my_goodput_dict=}')
    # print('End test with N=4k')

    print('Begin test with N=5x60xtarget_goodput')
    my_goodput_dict = {}
    for i in range(5):
        start_time = time.perf_counter()
        result = main(
            model_type=ModelTypes.llama3_70b, 
            attainment=(600, 100, 95, 95), 
            max_per_gpu_rate=20, 
            esp=0.05, 
            N=n_init if i == 0 else 5 * 60 * target_goodput, 
            max_cpu_count=28,
            # goodput_dict=my_goodput_dict,
            # target_goodput=target_goodput,
            seed=i,
        )
        end_time = time.perf_counter()
        print(result)
        print(f'total time for {i}-th run is {end_time-start_time:.3f}s')
        break
    print('End test with N=5x60xtarget_goodput')
    
    # print(result)

