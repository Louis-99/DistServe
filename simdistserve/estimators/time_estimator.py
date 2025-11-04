# Fit a model where prefill does not have an intercept, and decode does have one.
import json
import os
import numpy as np
import pandas as pd
from pathlib import Path

from simdistserve.constants import ModelTypes
from simdistserve.envs import GPU_FREQ

from scipy.interpolate import interpn

from lightgbm import LGBMRegressor
from joblib import load
import skl2onnx
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
import onnxruntime as ort

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
    
def load_tree_models():
    dec = None
    pre = None
    MODEL_DIR = Path(__file__).parent / "tree_models"
    dec_path = MODEL_DIR / "decode_model_latency.onnx"
    pre_path = MODEL_DIR / "prefill_model_latency.onnx"
    if dec_path.exists():
        dec = ort.InferenceSession(dec_path)
    if pre_path.exists():
        pre = ort.InferenceSession(pre_path)
    return dec, pre


distserve_profile_data: dict = load_distserve_profile_data()
ours_profile_data: dict = load_our_profile_data()
ours_freq_profile_data: dict = load_our_freq_profile_data()
vllm_profile_data: dict = load_vllm_profile_data()

dec_model, pre_model = load_tree_models()

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
    # input_df = pd.DataFrame([[bs, sum(prefill_len_list), TP, GPU_FREQ]],
    #                         columns=["batch_size", "total_tokens", "tp_degree", "freq"])
    # delay = prefill_lgbm_model.predict(input_df)[0]
    return delay * 1000


def get_decode_time(num_requests, pp=1, model_type=ModelTypes.opt_13b, TP=1, token_generated_list=None,
                    engine_type="distserve", freq: int = GPU_FREQ, **kw):
    batch_size = num_requests
    model_name = ModelTypes.formalize_model_name(model_type)

    if freq > 0:
        assert engine_type == "distserve", "only distserve support different GPU frequence right now"
        assert model_name in ours_freq_profile_data.keys()
        params_with_freq = ours_freq_profile_data[model_name][str(TP)]
        freq_list = list(map(int, params_with_freq.keys()))
        freq_list.sort()
        assert freq >= min(freq_list) and freq <= max(freq_list), "GPU frequency out of range"
        if freq in freq_list:
            params = params_with_freq[str(freq)]["decode"]
            a, b, c = get_coefs_from_param_with_thres(params, batch_size)
        else: # linear interpolation
            for freq_low, freq_high in zip(freq_list[:-1], freq_list[1:]):
                if freq > freq_low and freq < freq_high:
                    params1 = params_with_freq[str(freq_low)]["decode"]
                    params2 = params_with_freq[str(freq_high)]["decode"]
                    a1, b1, c1 = get_coefs_from_param_with_thres(params1, batch_size)
                    a2, b2, c2 = get_coefs_from_param_with_thres(params2, batch_size)
                    k1 = (freq_high - freq) / (freq_high - freq_low)
                    k2 = (freq - freq_low) / (freq_high - freq_low)
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

    # input_df = pd.DataFrame([[batch_size, sum(token_generated_list), TP, GPU_FREQ]],
    #                         columns=["batch_size", "total_tokens", "tp_degree", "freq"])
    # delay = decode_lgbm_model.predict(input_df)[0]
    return delay * 1000


def get_prefill_time_tree(num_tokens=None, pp=1, bs=1, decode_bs=0, model_type=ModelTypes.opt_13b, TP=1,
                     prefill_len_list=None, time_since_last_batch=0, engine_type="distserve", freq: int = GPU_FREQ, **kw):
    if bs == 0: # for when no work being done
        return 1
    sample_freq_list = [780, 1080, 1380, 1680, 1830]
    assert min(sample_freq_list) <= freq <= max(sample_freq_list)
    if freq in sample_freq_list:
        query_freq_list = [freq]
    else:
        for i, sample_freq in enumerate(sample_freq_list):
            if freq <= sample_freq:
                query_freq_list = [sample_freq_list[0]] if i == 0 else [sample_freq_list[i-1], sample_freq_list[i]]
                break

    delay_list = []
    model_name = ModelTypes.formalize_model_name(model_type)
    num_total_tokens = sum(prefill_len_list)
    for sample_freq in query_freq_list:
        input_feed = {
            "model": np.array([[model_name]], dtype=str),
            "batch_size": np.array([[bs]], dtype=np.float32),
            "input_len_sum": np.array([[num_total_tokens]], dtype=np.float32),
            "input_len_mean": np.array([[num_total_tokens / bs]], dtype=np.float32),
            "input_len_std": np.array([[np.std(prefill_len_list)]], dtype=np.float32),
            "tp_degree": np.array([[TP]], dtype=np.float32),
            "freq_mhz": np.array([[sample_freq]], dtype=np.float32),
        }
        delay_list.append(pre_model.run(None, input_feed)[0][0][0])

    if len(delay_list) == 1:
        return 1000 * delay_list[0]
    else:
        return 1000 * float(interpn(points=(query_freq_list,), values=delay_list, xi=[freq]))

def get_decode_time_tree(num_requests, pp=1, model_type=ModelTypes.opt_13b, TP=1, token_generated_list=None,
                    engine_type="distserve", freq: int = GPU_FREQ, **kw):
    batch_size = num_requests
    if batch_size == 0: # for when no work being done
        return 1

    sample_freq_list = [780, 1080, 1380, 1680, 1830]
    assert min(sample_freq_list) <= freq <= max(sample_freq_list)
    if freq in sample_freq_list:
        query_freq_list = [freq]
    else:
        for i, sample_freq in enumerate(sample_freq_list):
            if freq <= sample_freq:
                query_freq_list = [sample_freq_list[0]] if i == 0 else [sample_freq_list[i-1], sample_freq_list[i]]
                break

    delay_list = []
    model_name = ModelTypes.formalize_model_name(model_type)
    num_total_tokens = sum(token_generated_list)
    for sample_freq in query_freq_list:
        input_feed = {
            "model": np.array([[model_name]], dtype=str),
            "batch_size": np.array([[batch_size]], dtype=np.float32),
            "input_len_sum": np.array([[num_total_tokens]], dtype=np.float32),
            "input_len_mean": np.array([[num_total_tokens / batch_size]], dtype=np.float32),
            "input_len_std": np.array([[np.std(token_generated_list)]], dtype=np.float32),
            "tp_degree": np.array([[TP]], dtype=np.float32),
            "freq_mhz": np.array([[sample_freq]], dtype=np.float32),
        }
        delay_list.append(dec_model.run(None, input_feed)[0][0][0])

    if len(delay_list) == 1:
        return 1000 * delay_list[0]
    else:
        return 1000 * float(interpn(points=(query_freq_list,), values=delay_list, xi=[freq]))

if __name__ == "__main__":
    print(get_prefill_time_tree(bs=4, decode_bs=4, model_type=ModelTypes.gemma2_27b, TP=2, prefill_len_list=[512]*4))
    print(get_prefill_time_tree(bs=3, decode_bs=3, model_type=ModelTypes.gemma2_27b, TP=2, prefill_len_list=[512]*3))

    print(get_decode_time_tree(num_requests=4, decode_bs=4, model_type=ModelTypes.gemma2_27b, TP=2, token_generated_list=[512]*4))
    print(get_decode_time_tree(num_requests=3, decode_bs=3, model_type=ModelTypes.gemma2_27b, TP=2, token_generated_list=[512]*3))