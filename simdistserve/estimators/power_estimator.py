import json
import os
import numpy as np
import pandas as pd
from pathlib import Path

from simdistserve.constants import ModelTypes
from simdistserve.envs import get_gpu_freq

from lightgbm import LGBMRegressor
from joblib import load
import skl2onnx
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
import onnxruntime as ort

from scipy.interpolate import interpn

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
                     prefill_len_list=None, time_since_last_batch=0, engine_type="distserve", freq: None|int = None, **kw):
    if freq is None:
        freq = get_gpu_freq()
    if bs == 0: # for when no work being done
        return 0
    assert False, "don't use tree model for prefill power, use interp instead"
    model_name = ModelTypes.formalize_model_name(model_type)
    num_total_tokens = sum(prefill_len_list)
    input_feed = {
        "model": np.array([[model_name]], dtype=str),
        "batch_size": np.array([[bs]], dtype=np.float32),
        "input_len_sum": np.array([[num_total_tokens]], dtype=np.float32),
        "input_len_mean": np.array([[num_total_tokens / bs]], dtype=np.float32),
        "input_len_std": np.array([[np.std(prefill_len_list)]], dtype=np.float32),
        "tp_degree": np.array([[TP]], dtype=np.float32),
        "freq_mhz": np.array([[freq]], dtype=np.float32),
    }
    power = pre_model.run(None, input_feed)[0][0][0]
    return power

def get_decode_power_tree(num_requests, pp=1, model_type=ModelTypes.opt_13b, TP=1, token_generated_list=None,
                    engine_type="distserve", freq: None|int = None, **kw):
    if freq is None:
        freq = get_gpu_freq()

    batch_size = num_requests
    if batch_size == 0: # for when no work being done
        return 0
    
    sample_freq_list = [360, 570, 780, 1080, 1380, 1680, 1830]
    assert min(sample_freq_list) <= freq <= max(sample_freq_list)
    if freq in sample_freq_list:
        query_freq_list = [freq]
    else:
        for i, sample_freq in enumerate(sample_freq_list):
            if freq <= sample_freq:
                query_freq_list = [sample_freq_list[0]] if i == 0 else [sample_freq_list[i-1], sample_freq_list[i]]
                break
    
    power_list = []
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
        power_list.append(dec_model.run(None, input_feed)[0][0][0])
    if len(power_list) == 1:
        return power_list[0]
    else:
        return float(interpn(points=(query_freq_list,), values=power_list, xi=[freq]))

# New code for interp prefill power

possible_freq = np.array([ 360,  570,  780, 1080, 1380, 1680, 1830])
possible_input_len = np.array(
    [32,   64,   96,  128,  160,  192,  256,  384,  512,  768, 1024, 1280, 1536, 1792, 2048])

busy_power_values_dict = {
    4: np.array([
        [ 766.79782832,  818.96870802,  865.25887505,  984.72127171, 1140.23307683, 1396.33445554, 1385.88617877],
        [ 814.60153684,  876.97261754,  924.91678831, 1082.53452469, 1286.32423938, 1549.65300951, 1563.94855435],
        [ 833.32759634,  902.68569682,  954.73581584, 1140.63229992, 1390.3780871 , 1680.76153648, 1734.70052576],
        [ 860.67616663,  960.86158035, 1018.43250191, 1221.78283476, 1543.07507994, 1880.22343364, 1942.6190668 ],
        [ 831.1066616 ,  962.64835685, 1021.0607414 , 1224.78029167, 1607.07943643, 1927.45478727, 2127.5052573 ],
        [ 847.49345326, 1002.65630459, 1084.28640911, 1291.95411139, 1742.84315608, 2198.67549647, 2404.429592  ],
        [ 868.87835047, 1036.20243426, 1143.74980904, 1367.73381494, 1882.58995999, 2417.12518785, 2634.21850238],
        [ 864.77862948, 1043.89451998, 1181.99973971, 1429.2342348 , 1956.98430597, 2582.52452   , 2703.46718597],
        [ 779.6524155 ,  956.80138795, 1106.30663166, 1350.266084  , 1895.1023916 , 2612.01642318, 2745.32649567],
        [ 814.26393778, 1005.21900045, 1180.96507466, 1481.40437675, 2063.47833033, 2676.77416513, 2702.09631725],
        [ 828.05580063, 1032.6387077 , 1217.45493742, 1520.95248079, 2177.22006411, 2759.28575851, 2759.6566107 ],
        [ 836.89135666, 1034.72175891, 1193.05040754, 1509.91219388, 2172.56049632, 2745.16476285, 2749.93814766],
        [ 809.83990546, 1005.36396346, 1180.79333461, 1488.51255534, 2203.16329111, 2766.10493464, 2769.15640533],
        [ 820.17031014, 1017.70414403, 1261.75169216, 1606.47416798, 2322.08879116, 2761.67362614, 2765.07185752],
        [ 827.67340895, 1029.70857166, 1235.87224067, 1559.00235443, 2283.79591686, 2776.05462126, 2775.78978486],
    ]),
    2: np.array([
        [ 467.95701759,  496.55788652,  504.8714655 ,  644.05910082,  788.11071287,  964.64911905,  996.81165047],
        [ 488.153836  ,  522.63585906,  538.43388779,  679.8367838 ,  848.52178567, 1058.02993614, 1115.76423859],
        [ 501.14422157,  541.09192822,  558.62907545,  706.88651643,  891.60417424, 1123.97111969, 1218.24403351],
        [ 528.47890244,  592.96161024,  610.07134923,  787.30547876, 1020.72949344, 1280.57615433, 1376.44100852],
        [ 485.13935138,  590.1152666 ,  612.87938317,  786.45673293, 1023.93927892, 1311.40308726, 1392.21389419],
        [ 517.1179014 ,  629.16407227,  657.19657127,  834.17186602, 1141.80856109, 1383.5541608 , 1385.80419169],
        [ 478.86860659,  598.14209491,  664.47926456,  828.20318831, 1160.58184678, 1384.09975325, 1386.41520802],
        [ 486.30697887,  615.57433168,  684.86457528,  862.61612102, 1236.03113618, 1384.74473118, 1385.64871904],
        [ 464.11729818,  589.60919617,  676.96543248,  856.05726866, 1240.74766892, 1379.47374536, 1383.89076739],
        [ 443.4393147 ,  562.81395845,  646.56773068,  847.77476953, 1195.56088575, 1386.44341108, 1388.97561606],
        [ 451.4055709 ,  575.64271474,  665.42824697,  880.82823165, 1249.25048523, 1389.56332899, 1386.52722456],
        [ 456.53185123,  580.18164108,  657.98246341,  869.99551703, 1257.0508306 , 1392.58820345, 1391.76599695],
        [ 455.30556431,  580.04494506,  631.10018873,  826.83978314, 1203.5316831 , 1393.6138576 , 1387.72965146],
        [ 446.82092724,  568.29904263,  696.76195988,  926.3252101 , 1336.88877153, 1393.38139544, 1391.86912304],
        [ 450.95068799,  574.6552041 ,  682.11812234,  906.90752387, 1317.28533085, 1392.48822538, 1391.85159044],
    ]),
}

idle_power_values_dict = {
    4: np.array([316.850818757042, 341.6161000279436, 351.3068106255578, 385.13777390054054, 436.57400559323213, 555.108677099649, 618.3753703198889]),
    2: np.array([160.7856843934641, 167.50415613042543, 166.3517919412156, 183.0507797413239, 210.4289325309798, 300.1692712489896, 330.3604991503286]),
}

def get_prefill_power_interp(num_tokens=None, pp=1, bs=1, decode_bs=0, model_type=ModelTypes.opt_13b, TP=1,
                     prefill_len_list=None, time_since_last_batch=0, engine_type="distserve", freq: None|int = None, **kw):
    assert model_type == ModelTypes.llama3_70b, "Currently only support Llama3-70B"
    assert TP == 2 or TP == 4, f"{TP=} is not in {{2, 4}}"
    if freq is None:
        freq = get_gpu_freq()
    input_len = min(2048, max(32, num_tokens))
    return float(interpn(points=(possible_input_len, possible_freq), values=busy_power_values_dict[TP], xi=[input_len, freq]))

def get_prefill_idle_power_interp(TP: int, model_type: ModelTypes, freq: None|int = None):
    if freq is None:
        freq = get_gpu_freq()
    assert model_type == ModelTypes.llama3_70b, "Currently only support Llama3-70B"
    assert TP == 2 or TP == 4, f"{TP=} is not in {{2, 4}}"
    return float(interpn(points=(possible_freq,), values=idle_power_values_dict[TP], xi=[freq]))