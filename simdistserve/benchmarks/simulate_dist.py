"""
Simulate DistServe

Output a JSON (list) where each item is the lifecycle for a request.
"""
import argparse
import json
import os
import random
from pathlib import Path
from typing import Literal, Union

import numpy as np
import pandas as pd
import simpy

from simdistserve.base.organize_data import organize_request_df, organize_request_event_df, \
    calculate_per_request_latency, organize_worker_event_df
from simdistserve.base.scheduler import put_requests_with_interarrivals
from simdistserve.base.worker import WorkerConfig
from simdistserve.base.workload import (
    get_gamma_interarrival,
    get_fixed_interarrival,
    convert_absolutearrival_to_interarrival, convert_pd_pair_to_request, sample_requests
)
from simdistserve.clusters.disagg import DisaggCluster
from simdistserve.clusters.vllm import VLLMCluster
from simdistserve.constants import ModelTypes
from simdistserve.estimators.memory_estimator import get_max_num_tokens, is_model_runnable

from simdistserve.estimators.power_estimator import get_prefill_idle_power_interp
from simdistserve.envs import get_skip_prefill, get_skip_decode, SCALE_ARRIVAL_TIME, IGNORE_FIRST_AND_LAST_QUARTER, LIMIT_NUM_REQ
from simdistserve.envs import set_gpu_freq, set_skip_decode, set_skip_prefill

def freq_list_type_func(arg: str):
    freq_list = list(map(int, arg.split(',')))
    if len(freq_list) == 0:
        return None
    elif len(freq_list) == 1:
        return freq_list[0]
    else:
        return freq_list
    
def weight_list_type_func(arg:str):
    weight_list = list(map(float, arg.split(',')))
    if len(weight_list) <= 1:
        return None
    else:
        return weight_list
    

def parse_args(args_=None):
    parser = argparse.ArgumentParser(description='Simulation: vLLM, DistServe')
    parser.add_argument('--backend', type=str, default='distserve',
                        help='Backend to simulate (distserve, vllm)')
    parser.add_argument('--model', type=str, default='facebook/opt-13b',
                        help='Model type (opt_13b, opt_66b, opt_175b,'
                             'or facebook/opt-13b, facebook/opt-66b, facebook/opt-175b)')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--rate', type=float, default=float("inf"),
                        help='Rate of requests per second')
    parser.add_argument('--N', type=int, default=64, help='Number of requests')
    parser.add_argument(
        '--arrival', type=str, default='poisson',
        help=('Arrival distribution (gamma, poisson, fixed, custom). '
              'If custom, then require the JSON file workload to specify '
              'the "start_time" field for each incoming request.'))
    parser.add_argument(
        '--workload', type=str, default='sharegpt',
        help=(
            'Workload type, or a JSON file that contains the workload. '
            'The workload file should be a list of pairs with (prompt_len, decode_len) length. '
            '(e.g.: "sharegpt", "longbench", "humaneval", or specify your own path like "./workload/workload.json")')
    )
    parser.add_argument('--cv', type=float, default=1.0)
    parser.add_argument('--tp-prefill', type=int, default=1, help='Number of TP per prefill worker (used in DistServe)')
    parser.add_argument('--pp-prefill', type=int, default=1, help='Number of PP per prefill worker (used in DistServe)')
    parser.add_argument('--tp-decode', type=int, default=1, help='Number of TP per decode worker (used in DistServe)')
    parser.add_argument('--pp-decode', type=int, default=1, help='Number of PP per decode worker (used in DistServe)')
    parser.add_argument('--name', type=str, default=None)  # Experiment name
    parser.add_argument('--output', type=str, default=None, help='Output SLA (csv)')
    parser.add_argument('--output-request-info', type=str, default=None, help='Output request info (csv)')
    parser.add_argument('--output-request-event', type=str, default=None,
                        help='Output per-request event dataframe (csv)')
    parser.add_argument('--output-request-latency', type=str, default=None, help='Output per-request latency (csv)')
    parser.add_argument('--output-worker', type=str, default=None,
                        help='Output per-worker per-iteration time (csv)')
    parser.add_argument('--prefill-containment', type=int, default=None,
                        help='Containment target for prefill')
    parser.add_argument('--prefill-target', type=int, default=200,
                        help='Target latency for prefill')
    parser.add_argument('--decode-containment', type=int, default=None,
                        help='Containment target for decode')
    parser.add_argument('--slas', type=str, default='[85, 90, 95, 98, 99]',
                        help='Fix attainment and get the target.'),
    parser.add_argument('--slo-scales', type=str, default='[1.0, 0.4, 0.6, 0.8, 1.2]',
                        help='SLO scales in a python list.'),
    parser.add_argument('--decode-target', type=int, default=100,
                        help='Target latency for decode')
    parser.add_argument('--verbose', action='store_true', default=False,
                        help='Print verbose output')
    parser.add_argument('--n-decode', type=int, default=1)
    parser.add_argument('--n-prefill', type=int, default=1)
    parser.add_argument('--freq', type=int)
    parser.add_argument('--skip-prefill', type=int)
    parser.add_argument('--skip-decode', type=int)
    parser.add_argument('--prefill-freq', type=freq_list_type_func)
    parser.add_argument('--decode-freq', type=freq_list_type_func)
    parser.add_argument('--prefill-weights', type=weight_list_type_func)
    parser.add_argument('--decode-weights', type=weight_list_type_func)
    parser.add_argument('--rps-adjustment-method', type=str, choices=['stretch', 'sample', 'sample-max'], default='stretch')



    args = parser.parse_args(args=args_)

    assert args.backend in ['distserve', 'vllm'], f'Unknown backend: {args.backend}'
    assert args.arrival in ['poisson', 'gamma', 'fixed', 'custom'], f'Unknown arrival process: {args.arrival}'
    args.slo_scales = eval(args.slo_scales)
    args.slas = eval(args.slas)
    assert isinstance(args.slo_scales, list)
    assert isinstance(args.slas, list)
    return args


def check_dataset_existence(x):
    if not Path(x).exists():
        raise FileNotFoundError(f"Dataset {x} does not exist.")
    return


def load_workload(workload: str|pd.DataFrame, N, rate, cv, seed, process: Literal["fixed", "gamma"], rps_adjustment_method: str):
    random.seed(seed)
    np.random.seed(seed)
    if workload in ['sharegpt', 'longbench', 'humaneval']:
        dataset_root = os.environ.get('DATASET', '/app/dataset')
        dataset_root = Path(dataset_root)
        assert dataset_root.exists(), (
            f"Dataset root {dataset_root} does not exist. "
            f"Please set the env var `DATASET` to the correct path."
        )
        dataset_file = dataset_root / f"{workload}.ds"
        check_dataset_existence(dataset_file)
        requests = sample_requests(dataset_file, N)

        if process == 'fixed':
            delay = 1 / rate * 1000  # ms
            arrival = get_fixed_interarrival(N, delay)
        else:
            arrival = get_gamma_interarrival(N, rate, cv, seed=seed)

    else:
        # Open the file to get the JSON data
        # [ { "start_time": int, "prompt_len": int, "output_len":int,  } ]

        if isinstance(workload, str|os.PathLike) and workload.endswith('.json'):
            with open(workload, 'r') as f:
                data = json.load(f)
            if not LIMIT_NUM_REQ:
                N = len(data)
            input_len_array = np.array([d['prompt_len'] for i, d in enumerate(data) if i < N])
            output_len_array = np.array([d['output_len'] for i, d in enumerate(data) if i < N])
            arrival_time_array = np.array([d['start_time'] for i, d in enumerate(data) if i < N])
        else:
            if isinstance(workload, str|os.PathLike): 
                assert workload.endswith('.csv')
                df = pd.read_csv(workload)
            else:
                assert isinstance(workload, pd.DataFrame)
                df = workload
            if not LIMIT_NUM_REQ:
                N = len(df)
            if 'input_len' in df.keys():
                input_len_array = df['input_len'][:N].to_numpy()
            else:
                input_len_array = df['num_prefill_tokens'][:N].to_numpy()

            if 'output_len' in df.keys():
                output_len_array = df['output_len'][:N].to_numpy()
            else:
                output_len_array = df['num_decode_tokens'][:N].to_numpy()
            
            if 'time' in df.keys():
                arrival_time_array = df['time'][:N].to_numpy()
            else:
                arrival_time_array = df['arrived_at'][:N].to_numpy()
            assert len(input_len_array) == len(arrival_time_array)

        absolute_arrival = np.array(arrival_time_array)

        SAMPLE_START_IDX=0

        if SCALE_ARRIVAL_TIME:
            if rps_adjustment_method == 'stretch':
                cur_rate = len(absolute_arrival) / absolute_arrival[-1]
                absolute_arrival *= cur_rate / rate 
            elif rps_adjustment_method == 'sample':
                # assert not LIMIT_NUM_REQ, "Do not set LIMIT_NUM_REQ if sample method is used"
                total_num_req = len(absolute_arrival)
                num_sampled_req = np.round(rate * (absolute_arrival[-1] - absolute_arrival[0])).astype(int)
                sampled_req_idx = np.round(np.linspace(SAMPLE_START_IDX, total_num_req-1, num=num_sampled_req)).astype(int)
                absolute_arrival = absolute_arrival[sampled_req_idx]
                input_len_array = input_len_array[sampled_req_idx]
                output_len_array = output_len_array[sampled_req_idx]
            elif rps_adjustment_method == 'sample-max':
                # assert not LIMIT_NUM_REQ, "Do not set LIMIT_NUM_REQ if sample-max method is used"
                total_num_req = len(absolute_arrival)
                num_sampled_req = np.round(rate * (absolute_arrival[-1] - absolute_arrival[0])).astype(int)
                sampled_req_idx = np.round(np.linspace(SAMPLE_START_IDX, total_num_req-1, num=num_sampled_req)).astype(int)
                absolute_arrival = absolute_arrival[sampled_req_idx]
                new_input_len_list = [input_len_array[0]]
                new_output_len_list = [output_len_array[0]]
                for prev_idx, cur_idx in zip(sampled_req_idx[:-1], sampled_req_idx[1:]):
                    if prev_idx == cur_idx:
                        new_input_len_list.append(input_len_array[cur_idx])
                        new_output_len_list.append(output_len_array[cur_idx])
                    new_input_len_list.append(input_len_array[prev_idx+1:cur_idx+1].max())
                    new_output_len_list.append(output_len_array[prev_idx+1:cur_idx+1].max())
                input_len_array = np.array(new_input_len_list)
                output_len_array = np.array(new_output_len_list)
        else:
            assert 'sample' not in rps_adjustment_method, 'If you want to sample the trace, you must set SCALE_ARRIVAL_TIME=1'
        arrival = convert_absolutearrival_to_interarrival(absolute_arrival)

        assert len(input_len_array) == len(output_len_array)
        request_pairs = list(zip(input_len_array, output_len_array))
        requests = convert_pd_pair_to_request(request_pairs)

        assert len(requests) == len(absolute_arrival)
        assert len(requests) == len(arrival)
    
    if get_skip_decode():
        for req in requests:
            req.output_lens = 1
    
    return requests, arrival


def main(args, outputs=None, workload_df: None|pd.DataFrame = None):
    outputs = outputs if outputs is not None else {}

    cv = args.cv
    N = args.N
    rate = args.rate
    seed = args.seed
    workload: pd.DataFrame|str = workload_df if workload_df is not None else args.workload
    model_type = ModelTypes.model_str_to_object(args.model)
    process = args.arrival

    TP_Prefill = args.tp_prefill
    PP_prefill = args.pp_prefill
    TP_Decode = args.tp_decode
    PP_decode = args.pp_decode
    N_Prefill = args.n_prefill
    N_Decode = args.n_decode
    rps_adjustment_method = args.rps_adjustment_method

    prefill_freq = args.prefill_freq
    decode_freq = args.decode_freq

    prefill_weights = args.prefill_weights
    decode_weights = args.decode_weights

    if args.freq is not None:
        set_gpu_freq(args.freq)
    if args.skip_prefill is not None:
        set_skip_prefill(bool(args.skip_prefill))
    if args.skip_decode is not None:
        set_skip_decode(bool(args.skip_decode))

    #
    # Handle vllm in data processing
    #
    if not get_skip_prefill() and not is_model_runnable(model_type, TP_Prefill, PP_prefill):
        raise ValueError(
            f"Model {model_type} is not runnable with TP={TP_Prefill}, PP={PP_prefill}. "
            f"Skipping by throwing exception..."
        )
    
    if not get_skip_decode() and not is_model_runnable(model_type, TP_Decode, PP_decode):
        raise ValueError(
            f"Model {model_type} is not runnable with TP={TP_Prefill}, PP={PP_prefill}. "
            f"Skipping by throwing exception..."
        )
    
    if get_skip_prefill():
        prefill_max_tokens = int(1e9)
    else:
        prefill_max_tokens = get_max_num_tokens(model_type, TP_Prefill, PP_prefill)
    
    if get_skip_decode():
        decode_max_tokens = int(1e9)
    elif args.backend == 'vllm':
        TP_Decode = PP_decode = 0
        decode_max_tokens = prefill_max_tokens
        pass
    else:
        decode_max_tokens = get_max_num_tokens(model_type, TP_Decode, PP_decode)

    # Setting the seed to sample request / process
    requests, arrival = load_workload(workload, N, rate, cv, seed, process, rps_adjustment_method)

    # After sample, N may change
    N = len(arrival)

    # Run simulation
    env = simpy.Environment()
    if args.backend == 'vllm':
        worker_config = WorkerConfig(
            model_type=model_type,
            TP=TP_Prefill, TP_Prefill=TP_Prefill, TP_Decode=TP_Prefill,
            prefill_max_batch_size=10 ** 7,  # inf
            decode_max_batch_size=10 ** 7,  # inf
            prefill_max_tokens=prefill_max_tokens,
            decode_max_tokens=decode_max_tokens,
            enable_chunked_prefill=False,
            engine_type=args.backend,
        )

        cluster = VLLMCluster(
            env=env, PP=PP_prefill, worker_configs=worker_config,
        )
    elif args.backend == 'distserve':
        worker_config = WorkerConfig(
            model_type=model_type,
            TP=TP_Prefill, TP_Prefill=TP_Prefill, TP_Decode=TP_Decode,
            # prefill_max_batch_size=10 ** 7,  # inf
            # decode_max_batch_size=10 ** 7,  # inf
            # TODO: (Yunzhao) expose below two parameter to CLI
            prefill_max_batch_size=2048, 
            decode_max_batch_size=2048,
            # prefill_max_tokens=prefill_max_tokens,
            # prefill_max_tokens=1024*8,
            prefill_max_tokens=1024*2,
            decode_max_tokens=decode_max_tokens,
            # enable_chunked_prefill=False,
            enable_chunked_prefill=True,
            engine_type=args.backend,
        )

        cluster = DisaggCluster(
            env=env, PP_prefill=PP_prefill, PP_decode=PP_decode,
            N_prefill_instance=N_Prefill, N_decode_instance=N_Decode,
            worker_configs=worker_config,
            freq_prefill=prefill_freq,
            freq_decode=decode_freq,
            weights_prefill=prefill_weights,
            weights_decode=decode_weights,
        )
    else:
        raise ValueError(f"Unknown backend: {args.backend}")
    cluster.run()
    put_requests_with_interarrivals(env, cluster.scheduler, arrival, requests)
    env.run()

    #
    # Collect request-level data and containment
    #
    request_df = organize_request_df(requests)
    request_event_df = organize_request_event_df(requests)
    per_request_latency_df = calculate_per_request_latency(
        request_event_df, request_df.output_lens
    )
    outputs['request_df'] = request_df
    outputs['request_event_df'] = request_event_df
    outputs['per_request_latency_df'] = per_request_latency_df
    if args.output_request_info:
        with open(args.output_request_info, 'w') as f:
            request_df.to_csv(f, index=False)
    if args.output_request_event:
        with open(args.output_request_event, 'w') as f:
            request_event_df.to_csv(f, index=False)
    if args.output_request_latency:
        with open(args.output_request_latency, 'w') as f:
            per_request_latency_df.to_csv(f, index=False)

    columns = [
        "backend", "model_type", "pd", "rate", "target", "attainment",
        "tp_prefill", "pp_prefill", "tp_decode", "pp_decode",
    ]
    output_results = []
    # Fix the prefill & decode target (SLO & scale),
    # then find the attainment (percentage of requests that meet the SLO)
    for scale in args.slo_scales:
        prefill_target = args.prefill_target * scale
        if IGNORE_FIRST_AND_LAST_QUARTER:
            first = N // 4
            last = N * 3 // 4
            prefill_attainment = (per_request_latency_df['first_token_latency'][first:last] <= prefill_target).sum() / (last - first)
        else:
            prefill_attainment = (per_request_latency_df['first_token_latency'] <= prefill_target).sum() / N
        prefill_attainment *= 100
        item = [args.backend, model_type, 'prefill', rate, prefill_target, prefill_attainment,
                TP_Prefill, PP_prefill, TP_Decode, PP_decode]
        output_results.append(item)

        decode_target = args.decode_target * scale
        if IGNORE_FIRST_AND_LAST_QUARTER:
            first = N // 4
            last = N * 3 // 4
            decode_attainment = (per_request_latency_df['tpot'][first:last] <= decode_target).sum() / (last - first)
        else:
            decode_attainment = (per_request_latency_df['tpot'] <= decode_target).sum() / N
        decode_attainment *= 100
        item = [args.backend, model_type, 'decode', rate, decode_target, decode_attainment,
                TP_Prefill, PP_prefill, TP_Decode, PP_decode]
        output_results.append(item)

        both_attainment = (
                              (per_request_latency_df['first_token_latency'] <= prefill_target) &
                              (per_request_latency_df['tpot'] <= decode_target)
                          ).sum() / N
        both_attainment *= 100
        item = [args.backend, model_type, 'both', rate, (prefill_target, decode_target), both_attainment,
                TP_Prefill, PP_prefill, TP_Decode, PP_decode]
        output_results.append(item)
        pass

    # Fix the attainment (percentage of requests that meet the SLO),
    # then find the prefill /  decode SLO target that it can meet.
    slas = args.slas
    for sla in slas:
        prefill_attainment = decode_attainment = sla
        prefill_target = per_request_latency_df['first_token_latency'].quantile(prefill_attainment / 100)
        decode_target = per_request_latency_df['tpot'].quantile(decode_attainment / 100)
        item = [args.backend, model_type, 'prefill', rate, prefill_target, prefill_attainment,
                TP_Prefill, PP_prefill, TP_Decode, PP_decode]
        output_results.append(item)
        item = [args.backend, model_type, 'decode', rate, decode_target, decode_attainment,
                TP_Prefill, PP_prefill, TP_Decode, PP_decode]
        output_results.append(item)
        pass

    df = pd.DataFrame(output_results, columns=columns)
    outputs['latency_df'] = df

    if args.output:
        with open(args.output, 'w') as f:
            df.to_csv(f, index=False)

    if args.verbose:
        print(df.to_markdown())

    #
    # Collect worker-level data
    #
    worker_df = organize_worker_event_df(cluster)
    if args.output_worker:
        worker_df.to_csv(args.output_worker, index=False)

        outputs['worker_df'] = worker_df

    # if __name__ == '__main__':
    #     print('run simulate_dist as main')
    #     return
    
    # assert N_Prefill == 1
    # assert N_Decode == 1
    assert N_Prefill <= worker_df['worker_id'].max() < N_Prefill + N_Decode, f"{worker_df['worker_id'].max()=} {N_Prefill=} {N_Decode=}"
    prefill_worker_df = worker_df[worker_df['worker_id'] < N_Prefill]
    decode_worker_df = worker_df[worker_df['worker_id'] >= N_Prefill]

    if not get_skip_prefill():
        for prefill_id in prefill_worker_df['worker_id'].unique():
            if isinstance(prefill_freq, list):
                cur_freq = prefill_freq[prefill_id]
            else:
                cur_freq = prefill_freq
            prefill_idle_power = get_prefill_idle_power_interp(TP_Prefill, model_type, freq=cur_freq)
            prefill_worker_df.loc[prefill_worker_df['worker_id'] == prefill_id, 'power'] = \
                prefill_worker_df.loc[prefill_worker_df['worker_id'] == prefill_id, 'power'].clip(lower=prefill_idle_power) 
        if IGNORE_FIRST_AND_LAST_QUARTER:
            prefill_total_energy = 0
            for worker_id in prefill_worker_df['worker_id'].unique():
                prefill_worker_part_df = prefill_worker_df[prefill_worker_df['worker_id'] == worker_id]
                total_len = len(prefill_worker_part_df)
                power_array = prefill_worker_part_df['power'].to_numpy()[total_len // 4 : total_len * 3 // 4]
                duration_array = prefill_worker_part_df['duration'].to_numpy()[total_len // 4 : total_len * 3 // 4] * 1e-3
                prefill_total_energy += np.sum(power_array * duration_array)
        else:
            prefill_total_energy = np.sum(prefill_worker_df['power'].to_numpy() * prefill_worker_df['duration'].to_numpy() * 1e-3)
    else:
        prefill_total_energy = 0
    
    if not get_skip_decode():
        decode_idle_power = 76 * TP_Decode
        decode_worker_df.loc[:, 'power'] = decode_worker_df['power'].clip(lower=decode_idle_power) 
        if IGNORE_FIRST_AND_LAST_QUARTER:
            for worker_id in decode_worker_df['worker_id'].unique():
                decode_worker_part_df = decode_worker_df[decode_worker_df['worker_id'] == worker_id]
                total_len = len(decode_worker_part_df)
                power_array = decode_worker_part_df['power'].to_numpy()[total_len // 4 : total_len * 3 // 4]
                duration_array = decode_worker_part_df['duration'].to_numpy()[total_len // 4 : total_len * 3 // 4] * 1e-3
                decode_total_energy = np.sum(power_array * duration_array)
        else:
            decode_total_energy = np.sum(decode_worker_df['power'].to_numpy() * decode_worker_df['duration'].to_numpy() * 1e-3)
    else:
        decode_total_energy = 0
    
    #
    # Return if the agreement of prefill/decode is met
    #
    is_prefill_contained = None
    is_decode_contained = None

    prefill_containment = args.prefill_containment
    prefill_target = args.prefill_target
    if prefill_containment:
        # See if the P{prefill_containment} is less than prefill_target
        t = per_request_latency_df['first_token_latency'].quantile(prefill_containment / 100)
        is_prefill_contained = t < prefill_target
        pass

    decode_containment = args.decode_containment
    decode_target = args.decode_target
    if decode_containment:
        t = per_request_latency_df['tpot'].quantile(decode_containment / 100)
        is_decode_contained = t < decode_target
        pass

    prefill_energy_per_req = prefill_total_energy / N
    decode_energy_per_req = decode_total_energy / N

    return is_prefill_contained, is_decode_contained, prefill_energy_per_req, decode_energy_per_req, df


run_experiment = main


def test_opt_13b_grid_search_serial():
    arg_lists = [
        [
            '--arrival', 'poisson',
            '--seed', '0',
            '--N', '100',
            '--prefill-containment', '90',  # P90
            '--prefill-target', '200',  # ms
            '--decode-containment', '90',  # P90
            '--decode-target', '100',  # ms
            '--model', 'opt_13b',
            '--workload', 'sharegpt',
        ]
    ]

    config_list = [
        [
            '--rate', f'{rate}',
            # '--output',
            # f'raw_results/request.opt-13b-p{tp_prefill}{pp_prefill}{tp_decode}{pp_decode}-rate{rate}.csv',
            # '--output-worker',
            # f'raw_results/worker.opt-13b-p{tp_prefill}{pp_prefill}{tp_decode}{pp_decode}-rate{rate}.csv',
            '--pp-prefill', f'{pp_prefill}',
            '--pp-decode', f'{pp_decode}',
            '--tp-prefill', f'{tp_prefill}',
            '--tp-decode', f'{tp_decode}',
        ]
        # for rate in range(1, 8)
        for rate in range(1, 9)
        for pp_prefill in [1, 2, 4, 8]
        for pp_decode in [1, 2, 4, 8]
        for tp_prefill in [1, 2, 4, 8]
        for tp_decode in [1, 2, 4, 8]
    ]

    best_config = None
    best_goodput = 0
    # pbar = tqdm(total=len(arg_lists) * len(config_list))

    print("tp_prefill,pp_prefill,tp_decode,pp_decode,rate,goodput")
    for machine_config in config_list:
        best_config_this_iter = None
        for task_config in arg_lists:
            # pbar.update(1)

            args = parse_args(args_=task_config + machine_config)
            key = (
                args.tp_prefill, args.pp_prefill,
                args.tp_decode, args.pp_decode,
            )
            num_gpu = args.pp_prefill * args.tp_prefill + args.pp_decode * args.tp_decode
            rate = args.rate * num_gpu
            if num_gpu > 32:
                continue
            goodput = args.rate / num_gpu
            if goodput < best_goodput:
                continue

            # print(args.rate, args.pp_prefill, args.tp_prefill, args.pp_decode, args.tp_decode)
            is_prefill_contained, is_decode_contained, containment_df = main(args)

            if not is_prefill_contained or not is_decode_contained:
                break

            if goodput > best_goodput:
                best_config = args
                best_goodput = goodput

                print(
                    f"{best_config.tp_prefill},{best_config.pp_prefill},"
                    f"{best_config.tp_decode},{best_config.pp_decode},"
                    f"{rate},{best_goodput}")

    best_config_str = (f"(Prefill TP = {best_config.tp_prefill}, Prefill PP = {best_config.pp_prefill},"
                       f" Decode TP = {best_config.tp_decode}, Decode PP = {best_config.pp_decode})")
    print(f"Best Config: {best_config_str} with goodput {best_goodput}")


def test_opt_13b_one_case(
    rate=1, tp_prefill=1, pp_prefill=1, tp_decode=1, pp_decode=1,
    backend='vllm',
):
    suffix = f"opt-13b-p{tp_prefill}{pp_prefill}{tp_decode}{pp_decode}-rate{rate}.csv"
    args = [
        '--arrival', 'poisson',
        '--seed', '0',
        '--N', '1000',
        '--backend', backend,
        '--prefill-containment', '90',  # P90
        '--prefill-target', '200',  # ms
        '--decode-containment', '90',  # P90
        '--decode-target', '100',  # ms
        '--model', 'opt_13b',
        '--workload', 'sharegpt',
        '--rate', f'{rate}',
        '--output', f'logs/request.{suffix}',
        '--output-request-event', f'logs/request-event.{suffix}',
        '--output-request-latency', f'logs/request-latency.{suffix}',
        '--output-worker', f'logs/worker.{suffix}',
        '--pp-prefill', f'{pp_prefill}',
        '--pp-decode', f'{pp_decode}',
        '--tp-prefill', f'{tp_prefill}',
        '--tp-decode', f'{tp_decode}',
        '--verbose',
    ]
    args = parse_args(args_=args)
    main(args)
    return


if __name__ == '__main__':
    args = parse_args()
    print(args)
    main(args)
    # test_opt_13b_grid_search_serial()
    pass
