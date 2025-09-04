# Fit a model where prefill does not have an intercept, and decode does have one.
import json
from pathlib import Path

from simdistserve.constants import ModelTypes
from simdistserve.envs import GPU_FREQ

# TODO: (Yunzhao) add new data to json
def load_distserve_profile_data():
    profile_data_path = Path(__file__).parent / "profile_data" / "profiler-a100-80g.distserve.json"
    with open(profile_data_path) as f:
        profile_data = json.load(f)
        return profile_data

def load_our_profile_data():
    profile_data_path = Path(__file__).parent / "profile_data" / "ours_a10040g_profile_data.json"
    with open(profile_data_path) as f:
        profile_data = json.load(f)
        return profile_data
    
def load_our_freq_profile_data():
    profile_data_path = Path(__file__).parent / "profile_data" / "ours_a10040g_freq_profile_data.json"
    with open(profile_data_path) as f:
        profile_data = json.load(f)
        return profile_data


def load_vllm_profile_data():
    profile_data_path = Path(__file__).parent / "profile_data" / "profiler-a100-80g.vllm.json"
    with open(profile_data_path) as f:
        profile_data = json.load(f)
        return profile_data


distserve_profile_data: dict = load_distserve_profile_data()
ours_profile_data: dict = load_our_profile_data()
ours_freq_profile_data: dict = load_our_freq_profile_data()
vllm_profile_data: dict = load_vllm_profile_data()

def get_coefs_from_param_with_thres(param, value_to_check):
    thres_list = param['thres']
    coefs_list = param['coefs']
    assert len(coefs_list) - len(thres_list) == 1
    for thres, coefs in zip(thres_list, coefs_list):
        if value_to_check <= thres:
            return coefs
    return coefs_list[-1]


def get_prefill_time(num_tokens=None, pp=1, bs=1, decode_bs=0, model_type=ModelTypes.opt_13b, TP=1,
                     prefill_len_list=None, engine_type="distserve", **kw):
    model_name = ModelTypes.formalize_model_name(model_type)

    if GPU_FREQ > 0:
        assert engine_type == "distserve", "only distserve support different GPU frequence right now"
        assert model_name in ours_freq_profile_data.keys()
        params_with_freq = ours_freq_profile_data[model_name][str(TP)]
        freq_list = list(map(int, params_with_freq.keys()))
        freq_list.sort()
        assert GPU_FREQ >= min(freq_list) and GPU_FREQ <= max(freq_list), "GPU frequency out of range"
        if GPU_FREQ in freq_list:
            params = params_with_freq[str(GPU_FREQ)]["prefill"]
            a, b, c = get_coefs_from_param_with_thres(params, num_tokens)
        else: # linear interpolation
            for freq_low, freq_high in zip(freq_list[:-1], freq_list[1:]):
                if GPU_FREQ > freq_low and GPU_FREQ < freq_high:
                    params1 = params_with_freq[str(freq_low)]["prefill"]
                    params2 = params_with_freq[str(freq_high)]["prefill"]
                    a1, b1, c1 = get_coefs_from_param_with_thres(params1, num_tokens)
                    a2, b2, c2 = get_coefs_from_param_with_thres(params2, num_tokens)
                    k1 = (freq_high - GPU_FREQ) / (freq_high - freq_low)
                    k2 = (GPU_FREQ - freq_low) / (freq_high - freq_low)
                    a = k1 * a1 + k2 * a2
                    b = k1 * b1 + k2 * b2
                    c = k1 * c1 + k2 * c2
                    break
    elif engine_type == "distserve":
        if model_name in distserve_profile_data.keys():
            params = distserve_profile_data[model_name][str(TP)]
            a, b, c = params["prefill"]
        else:
            assert model_name in ours_profile_data.keys()
            params = ours_profile_data[model_name][str(TP)]
            a, b, c = params["prefill"]
    else:
        params = vllm_profile_data[ModelTypes.formalize_model_name(model_type)][str(TP)]
        a, b, c = params["prefill"]

    f = 1
    a, b, c = (a * f, b * f, c * f)
    pp_factor = 1 / pp
    pp_const = 1 * pp  # TODO: Modulate the PP overhead
    num_total_tokens = sum(prefill_len_list)
    sum_num_tokens_sqr = sum([x ** 2 for x in prefill_len_list])
    delay = a + b * num_total_tokens + c * sum_num_tokens_sqr
    delay = delay * pp_factor + pp_const
    return delay


def get_decode_time(num_requests, pp=1, model_type=ModelTypes.opt_13b, TP=1, token_generated_list=None,
                    engine_type="distserve", **kw):
    batch_size = num_requests
    model_name = ModelTypes.formalize_model_name(model_type)

    if GPU_FREQ > 0:
        assert engine_type == "distserve", "only distserve support different GPU frequence right now"
        assert model_name in ours_freq_profile_data.keys()
        params_with_freq = ours_freq_profile_data[model_name][str(TP)]
        freq_list = list(map(int, params_with_freq.keys()))
        freq_list.sort()
        assert GPU_FREQ >= min(freq_list) and GPU_FREQ <= max(freq_list), "GPU frequency out of range"
        if GPU_FREQ in freq_list:
            params = params_with_freq[str(GPU_FREQ)]["decode"]
            a, b, c = get_coefs_from_param_with_thres(params, batch_size)
        else: # linear interpolation
            for freq_low, freq_high in zip(freq_list[:-1], freq_list[1:]):
                if GPU_FREQ > freq_low and GPU_FREQ < freq_high:
                    params1 = params_with_freq[str(freq_low)]["decode"]
                    params2 = params_with_freq[str(freq_high)]["decode"]
                    a1, b1, c1 = get_coefs_from_param_with_thres(params1, batch_size)
                    a2, b2, c2 = get_coefs_from_param_with_thres(params2, batch_size)
                    k1 = (freq_high - GPU_FREQ) / (freq_high - freq_low)
                    k2 = (GPU_FREQ - freq_low) / (freq_high - freq_low)
                    a = k1 * a1 + k2 * a2
                    b = k1 * b1 + k2 * b2
                    c = k1 * c1 + k2 * c2
                    break
    elif engine_type == "distserve":
        if model_name in distserve_profile_data.keys():
            params = distserve_profile_data[model_name][str(TP)]
            threshold = params[
                "decoding_large_small_bs_threshold"]
            if batch_size < threshold:
                a, b, c = params["decoding_smallbs"]
            else:
                a, b, c = params["decoding_largebs"]
        else:
            assert model_name in ours_profile_data.keys()
            params = ours_profile_data[model_name][str(TP)]
            threshold_list = params["decode_thres"]
            decode_param_idx = 0
            for thres in threshold_list:
                if batch_size < thres:
                    decode_param_idx += 1
                else:
                    break
            a, b, c = params["decode"][decode_param_idx]
    else:
        params = vllm_profile_data[ModelTypes.formalize_model_name(model_type)][str(TP)]
        threshold = params[
            "decoding_large_small_bs_threshold"]
        if batch_size < threshold:
            a, b, c = params["decoding_smallbs"]
        else:
            a, b, c = params["decoding_largebs"]
        pass
    f = 1
    pp_factor = 1 / pp
    # pp_const = 1 * pp  # TODO: Modulate the PP overhead
    pp_const = 0
    num_total_tokens = sum(token_generated_list)

    delay = a + b * num_total_tokens + c * batch_size
    delay = delay * pp_factor + pp_const
    delay *= f
    return delay
