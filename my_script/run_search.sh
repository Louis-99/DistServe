export DATASET=/export1/liu3882/llm_energy/dataset

export SCALE_ARRIVAL_TIME=1 
export LIMIT_NUM_REQ=1 
python -m simdistserve.benchmarks.goodput_freq_search > search_goodput_diff_workload.txt