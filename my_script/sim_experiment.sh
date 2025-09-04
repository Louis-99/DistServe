export DATASET=/export1/liu3882/llm_energy/dataset


prefill_target=600
decode_target=100

result_folder=pd_results_new_full_mem
mkdir -p $result_folder
export GPU_FREQ=1410


for prefill_target in 200 400 600 800 1000 1200; do 
    export SKIP_PREFILL=0
    export SKIP_DECODE=1

    python -m simdistserve.simulate \
        --num-node 1 --ngpu-per-node 16 \
        --is-high-affinity \
        --model-type "gemma2_27b" \
        --workload sharegpt --backend distserve \
        --prefill-target $prefill_target --decode-target $decode_target \
        --prefill-percentage 90 --decode-percentage 90 \
        --max-per-gpu-rate 100 \
        --esp 0.01 \
        --N 8000 > ${result_folder}/sim_gemma3_27b_prefill_p${prefill_target}_d${decode_target}_out.txt &

    export SKIP_PREFILL=1
    export SKIP_DECODE=0

    python -m simdistserve.simulate \
        --num-node 1 --ngpu-per-node 16 \
        --is-high-affinity \
        --model-type "gemma2_27b" \
        --workload sharegpt --backend distserve \
        --prefill-target $prefill_target --decode-target $decode_target \
        --prefill-percentage 90 --decode-percentage 90 \
        --max-per-gpu-rate 100 \
        --esp 0.01 \
        --N 8000 > ${result_folder}/sim_gemma3_27b_decode_p${prefill_target}_d${decode_target}_out.txt &
done

wait

# distserve 13b Best config: pp_cross=1, tp_prefill=1, pp_prefill=1, tp_decode=1, pp_decode=1, n_prefill=11, n_decode=5
# Best config: pp_cross=1, tp_prefill=1, pp_prefill=1, tp_decode=1, pp_decode=1, n_prefill=11, n_decode=5
# distserve 66b Best config:
#  Best config: pp_cross=1, tp_prefill=2, pp_prefill=2, tp_decode=2, pp_decode=1, n_prefill=2, n_decode=4

