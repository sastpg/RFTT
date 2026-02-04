python src/do_search.py \
    --dataset_name MATH \
    --test_json_filename test_all \
    --model_ckpt /pubshare/LLM/Qwen/Qwen2.5-7B-Instruct \
    --model_port 8001 \
    --prm_url http://localhost:8008/get_reward \
    --note default \
    --num_rollouts 16 \
    --verbose \
    --save_tree \
    --max_depth_allowed 15

# --start_idx 0 \
# --end_idx 1000