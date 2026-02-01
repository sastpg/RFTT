CUDA_VISIBLE_DEVICES=1 python src/do_search.py \
    --dataset_name MATH \
    --test_json_filename test_all \
    --model_ckpt /path/to/Qwen3-4B-Instruct-2507 \
    --model_port 8001 \
    --note default \
    --num_rollouts 16 \
    --verbose \
    --save_tree \
    --max_depth_allowed 15

# --start_idx 0 \
# --end_idx 1000