set -x 
# export CUDA_VISIBLE_DEVICES=0,1,2,3
ray job submit --address="http://127.0.0.1:8265" \
   --runtime-env-json='{"working_dir": "/openrlhf/OpenRLHF_MCTS","excludes":["run_outputs/"]}' \
   -- python3 -m openrlhf.cli.train_ppo_ray \
   --ref_num_nodes 1 \
   --ref_num_gpus_per_node 2 \
   --actor_num_nodes 1 \
   --actor_num_gpus_per_node 2 \
   --vllm_num_engines 2 \
   --vllm_tensor_parallel_size 1 \
   --colocate_actor_ref \
   --pretrain /openrlhf/models/Qwen2.5-7B-SFT \
   --remote_rm_url http://localhost:5000/get_reward \
   --reward_pretrain /do_not_care \
   --save_path /openrlhf/models/checkpoints \
   --advantage_estimator reinforce \
   --n_samples_per_prompt 16 \
   --micro_train_batch_size 8 \
   --train_batch_size 16 \
   --micro_rollout_batch_size 8 \
   --rollout_batch_size 16 \
   --max_samples 100000 \
   --max_epochs 1 \
   --bf16 \
   --flash_attn \
   --prompt_max_len 8192 \
   --generate_max_len 1024 \
   --zero_stage 3 \
   --actor_learning_rate 5e-7 \
   --critic_learning_rate 9e-6 \
   --init_kl_coef 0.01 \
   --prompt_data /openrlhf/OpenRLHF_MCTS/datasets/train.jsonl \
   --input_key problem \
   --normalize_reward \
   --packing_samples \
   --adam_offload \
   --gradient_checkpointing \
   --load_checkpoint \
   --save_tree /openrlhf/OpenRLHF_MCTS/run_outputs \
   --sampling_method mcts \
   --use_prm \
   --search_depth 18 \
   --save_steps 100 \
   --logging_path /openrlhf/OpenRLHF_MCTS/mcts_prm.jsonl \
   --apply_chat_template

# --runtime-env-json='{"setup_commands": ["pip install openrlhf[vllm]"]}' [Install deps]
# --ref_reward_offload [Offload to CPU]
# --remote_rm_url http://localhost:5000/get_reward

# --vllm_sync_backend nccl (Only for multi-nodes with vLLM 0.6.4+ or vLLM 0.4.2)