import json
import os
import numpy as np
import pandas as pd
from pathlib import Path

from simdistserve.constants import ModelTypes
from simdistserve.envs import GPU_FREQ

from lightgbm import LGBMRegressor
from joblib import load
import skl2onnx
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
import onnxruntime as ort

def load_tree_models():
    dec = None
    pre = None
    MODEL_DIR = Path(__file__).parent / "tree_models"
    dec_path = MODEL_DIR / "decode_model_power.onnx"
    pre_path = MODEL_DIR / "prefill_model_power.onnx"
    if dec_path.exists():
        dec = ort.InferenceSession(dec_path)
    if pre_path.exists():
        pre = ort.InferenceSession(pre_path)
    return dec, pre

dec_model, pre_model = load_tree_models()


def get_prefill_power_tree(num_tokens=None, pp=1, bs=1, decode_bs=0, model_type=ModelTypes.opt_13b, TP=1,
                     prefill_len_list=None, time_since_last_batch=0, engine_type="distserve", **kw):
    if bs == 0: # for when no work being done
        return 0
    model_name = ModelTypes.formalize_model_name(model_type)
    num_total_tokens = sum(prefill_len_list)
    input_feed = {
        "model": np.array([[model_name]], dtype=str),
        "batch_size": np.array([[bs]], dtype=np.float32),
        "input_len_sum": np.array([[num_total_tokens]], dtype=np.float32),
        "input_len_mean": np.array([[num_total_tokens / bs]], dtype=np.float32),
        "input_len_std": np.array([[np.std(prefill_len_list)]], dtype=np.float32),
        "tp_degree": np.array([[TP]], dtype=np.float32),
        "freq_mhz": np.array([[GPU_FREQ]], dtype=np.float32),
    }
    power = pre_model.run(None, input_feed)[0][0][0]
    return power

def get_decode_power_tree(num_requests, pp=1, model_type=ModelTypes.opt_13b, TP=1, token_generated_list=None,
                    engine_type="distserve", **kw):
    batch_size = num_requests
    if batch_size == 0: # for when no work being done
        return 0
    model_name = ModelTypes.formalize_model_name(model_type)
    num_total_tokens = sum(token_generated_list)
    input_feed = {
        "model": np.array([[model_name]], dtype=str),
        "batch_size": np.array([[batch_size]], dtype=np.float32),
        "input_len_sum": np.array([[num_total_tokens]], dtype=np.float32),
        "input_len_mean": np.array([[num_total_tokens / batch_size]], dtype=np.float32),
        "input_len_std": np.array([[np.std(token_generated_list)]], dtype=np.float32),
        "tp_degree": np.array([[TP]], dtype=np.float32),
        "freq_mhz": np.array([[GPU_FREQ]], dtype=np.float32),
    }
    power = dec_model.run(None, input_feed)[0][0][0]
    return power