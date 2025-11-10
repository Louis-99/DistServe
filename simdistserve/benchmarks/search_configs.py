from itertools import product

from simdistserve.constants import ModelTypes
from simdistserve.estimators.memory_estimator import get_model_possible_tp, get_model_possible_pp

from simdistserve.envs import get_skip_decode, get_skip_prefill

def get_distserve_configs(
    model_type: ModelTypes,
    num_node: int,
    num_gpu_per_node: int,
    is_high_affinity: bool,
) -> '(pp_cross, tp_prefill, pp_prefill, tp_decode, pp_decode)':
    total_num_gpus = num_node * num_gpu_per_node

    model_name: str = ModelTypes.formalize_model_name(model_type)
    possible_tps = get_model_possible_tp(model_name)
    possible_pps = get_model_possible_pp(model_name)

    # tp
    if is_high_affinity:
        tps = [i for i in possible_tps if i <= total_num_gpus]
    else:
        # within-node tp (if low-affinity)
        tps = [i for i in possible_tps if i <= num_gpu_per_node]

    # pp
    if is_high_affinity:
        pps = [i for i in possible_pps if i <= total_num_gpus]
    else:
        # within-node pp (if low-affinity)
        pps = [i for i in possible_pps if i <= num_gpu_per_node]

    # Cross-node pp
    if is_high_affinity:
        cpps = [1]  # ignore cross-node restriction
    else:
        # cross-node pp (if low-affinity)
        cpps = [i for i in possible_pps if i <= num_node]
        pass

    exist_configs_for_skip = set()

    # Get all possible configs
    possible_configs = []
    for pp_cross, tp_prefill, pp_prefill, tp_decode, pp_decode in product(cpps, tps, pps, tps, pps):
        gpu_per_segment = (tp_prefill * pp_prefill + tp_decode * pp_decode)
        if not is_high_affinity:
            if gpu_per_segment > num_gpu_per_node:
                continue

        if pp_cross * gpu_per_segment > total_num_gpus:
            continue
        if pp_cross * pp_prefill not in possible_pps:
            continue
        if pp_cross * pp_decode not in possible_pps:
            continue

        if is_high_affinity:
            assert pp_cross == 1
            gpu_per_prefill = tp_prefill * pp_prefill
            gpu_per_decode = tp_decode * pp_decode
            max_instance_cnt = total_num_gpus // min(gpu_per_decode, gpu_per_prefill)
            if get_skip_prefill() or get_skip_decode():
                max_instance_cnt = 2 # one for decode and one for prefill (cannot be 1 because of code below)
            for n_prefill in range(1, max_instance_cnt): # = max_instance_cnt is not possible
                for n_decode in range(1, max_instance_cnt): # = max_instance_cnt is not possible
                    if n_prefill * gpu_per_prefill + n_decode * gpu_per_decode > total_num_gpus:
                        break
                    if get_skip_decode() or get_skip_prefill():
                        if get_skip_decode():
                            cur_config = (tp_prefill, pp_prefill)
                        else: # SKIP_PREFIL
                            cur_config = (tp_decode, pp_decode)
                        if cur_config in exist_configs_for_skip:
                            continue
                        else:
                            exist_configs_for_skip.add(cur_config)
                    possible_configs.append((pp_cross, tp_prefill, pp_prefill, tp_decode, pp_decode, n_prefill, n_decode))

        else:
            possible_configs.append((pp_cross, tp_prefill, pp_prefill, tp_decode, pp_decode))
        pass
    return possible_configs


def get_vllm_config(
    model_type: ModelTypes,
    num_gpus: int,
) -> '(tp, pp)':
    model_name: str = ModelTypes.formalize_model_name(model_type)
    possible_tps = get_model_possible_tp(model_name)
    possible_pps = get_model_possible_pp(model_name)
    possible_configs = []
    for tp, pp in product(possible_tps, possible_pps):
        if tp * pp > num_gpus:
            continue
        possible_configs.append((tp, pp))
    return possible_configs
