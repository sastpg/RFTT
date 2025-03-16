CUDA_VISIBLE_DEVICES=0 python src/do_search.py \
    --dataset_name MATH \
    --model_ckpt /path/to/Qwen2.5-7B-Instruct \
    --note default \
    --num_rollouts 16 \
    --verbose \
    --save_tree \
    --max_depth_allowed 15

# --start_idx 0 \
# --end_idx 1000