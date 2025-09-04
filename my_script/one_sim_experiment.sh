#!/bin/bash
export DATASET=/export1/liu3882/llm_energy/dataset

export GPU_FREQ=1410

output_folder=sim_one_outputs

mkdir -p $output_folder

python -m simdistserve.benchmarks.simulate_dist \
    --arrival 'poisson' \
    --seed 0 \
    --N 10000 \
    --prefill-containment 90 \
    --prefill-target 600 \
    --decode-containment 90 \
    --decode-target 100 \
    --model 'google/gemma-2-27b'\
    --workload sharegpt \
    --slas '[]' \
    --slo-scales '[1]' \
    --backend 'distserve' \
    --tp-prefill 4 \
    --pp-prefill 1 \
    --tp-decode 4 \
    --pp-decode 1 \
    --n-prefill 1 \
    --n-decode 1 \
    --rate 14.697265625 \
    --output-request-latency $output_folder/req_latency.csv \
    --output-worker $output_folder/worker.csv 
    # --output-request-event $output_folder/req_event.csv
