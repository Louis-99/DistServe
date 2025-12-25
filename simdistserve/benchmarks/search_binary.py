import time

from simdistserve.benchmarks.simulate_dist import run_experiment, parse_args
from simdistserve.constants import ModelTypes

from simdistserve.envs import get_skip_decode, get_skip_prefill, OPTIMIZE_ENERGY, get_gpu_freq


def run_binary_search(
    model_type: ModelTypes,
    # config
    # - distserve: (pp_cross, tp_prefill, pp_prefill, tp_decode, pp_decode)
    # - vllm: (tp, pp)
    config,
    backend: str,
    containment_targets: '(prefill_target, decode_target, prefill_containment, decode_containment)',
    max_per_gpu_rate: int = 16,
    min_per_gpu_rate: int = 0,
    freq: None|int = None,
    pid=0,
    esp=0.5,
    N=1000,
    debug=False,
    result=None,
):
    N = str(N)

    if freq is None:
        freq = get_gpu_freq()

    #
    # make config args
    #
    if backend == 'distserve':
        assert len(config) == 5 or len(config) == 7
        if len(config) == 5:
            (pp_cross, tp_prefill, pp_prefill, tp_decode, pp_decode) = config
            num_gpu = pp_cross * (tp_prefill * pp_prefill + tp_decode * pp_decode)
            config_args = [
                '--tp-prefill', f'{tp_prefill}',
                '--pp-prefill', f'{pp_cross * pp_prefill}',
                '--tp-decode', f'{tp_decode}',
                '--pp-decode', f'{pp_cross * pp_decode}',
            ]
        elif len(config) == 7:
            (pp_cross, tp_prefill, pp_prefill, tp_decode, pp_decode, n_prefill, n_decode) = config
            num_gpu = pp_cross * (tp_prefill * pp_prefill * n_prefill + tp_decode * pp_decode * n_decode)
            config_args = [
                '--tp-prefill', f'{tp_prefill}',
                '--pp-prefill', f'{pp_cross * pp_prefill}',
                '--tp-decode', f'{tp_decode}',
                '--pp-decode', f'{pp_cross * pp_decode}',
                '--n-prefill', f'{n_prefill}',
                '--n-decode', f'{n_decode}',
            ]

        assert not (get_skip_decode() and get_skip_prefill())
        if get_skip_decode(): # prefill only
            num_gpu = pp_cross * pp_prefill * tp_prefill
        elif get_skip_prefill(): # decode only
            num_gpu = pp_cross * pp_decode * tp_decode
    else:
        (tp, pp) = config
        num_gpu = tp * pp
        config_args = [
            '--tp-prefill', f'{tp}',
            '--pp-prefill', f'{pp}',
            '--tp-decode', 0,
            '--pp-decode', 0,
        ]

    prefill_target, decode_target, prefill_containment, decode_containment = containment_targets

    #
    # bisect the integer range between 1 <= per_gpu_rate <= max_per_gpu_rate
    #
    low = min_per_gpu_rate
    high = max_per_gpu_rate
    best_per_gpu_rate = 0

    fixed_args = [
        '--arrival', 'poisson',
        '--seed', '0',
        '--N', N,
        '--prefill-containment', prefill_containment,  # P90
        '--prefill-target', prefill_target,  # ms
        '--decode-containment', decode_containment,  # P90
        '--decode-target', decode_target,  # ms
        '--model', ModelTypes.formalize_model_name(model_type),
        '--workload', 'sharegpt',
        '--slas', '[]',
        '--slo-scales', '[1]',
        '--backend', backend,
        '--freq', freq,
    ]

    time_durations = []
    prefill_energy_per_req = None
    decode_energy_per_req = None
    while (high - low) > esp:
        # print(f"pid={pid}, config={config}, low={low}, high={high}")
        # Run simulation
        this_rate = (low + high) / 2
        # print(f'{this_rate=}')
        rate = this_rate * num_gpu
        args = [*fixed_args, *config_args, '--rate', rate, ]
        args = [str(i) for i in args]
        args = parse_args(args)
        try:
            start_time = time.time()
            is_prefill_contained, is_decode_contained, prefill_energy_per_req, decode_energy_per_req, df = run_experiment(args)
            end_time = time.time()
            time_durations.append((config, this_rate, end_time - start_time))
        except Exception as e:
            from simdistserve.envs import PRINT_EXCEPT_MSG
            if PRINT_EXCEPT_MSG:
                import traceback
                print(
                    f"({pid=}) Error when computing {config=}. This may not be a real error "
                    f"(e.g. bad parallelism strategy). Exception detail: {traceback.format_exc()}."
                )
            return None

        # Update the range
        success = is_prefill_contained and is_decode_contained
        if not success:
            # The experiment not passing the attainment threshold
            high = this_rate
            continue

        # Experiment passed the attainment threshold
        low = this_rate
        best_per_gpu_rate = this_rate
        pass
    if result is not None:
        assert prefill_energy_per_req is not None
        assert decode_energy_per_req is not None
        assert not get_skip_decode() or not get_skip_prefill()
        if get_skip_decode():
            total_energy_per_req = prefill_energy_per_req
        elif get_skip_prefill():
            total_energy_per_req = decode_energy_per_req
        else:
            total_energy_per_req = prefill_energy_per_req + decode_energy_per_req
        if OPTIMIZE_ENERGY:
            result[config] = 1 / total_energy_per_req
        else:
            result[config] = (best_per_gpu_rate, total_energy_per_req) 
    if OPTIMIZE_ENERGY:
        return 1 / total_energy_per_req
    else:
        return best_per_gpu_rate


if __name__ == '__main__':
    args = parse_args()
    run_binary_search(
        ModelTypes.opt_13b,
        (1, 1, 1, 1, 1),
        "distserve",
        (200, 100, 90, 90),
        max_per_gpu_rate=16,
        is_debug=args.debug,
    )
