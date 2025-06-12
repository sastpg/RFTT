# Two phase training
## Supervised fine-tuning
We use [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) to supervise fine-tuning initial models on reasoning pathe with learnable functional tokens. The configuration files in this stage are in `./sft`.

## Reinforcement Learning
We modify [OpenRLHF](https://github.com/OpenRLHF/OpenRLHF) to reinforce high-value reasoning paths through functional token-guided tree search in online RL.

> [!IMPORTANT]
> When using PRM as the reward model, we recommend deploying the reward model as an API server using provided code `nohup python prm_api.py &`. After that, copy the address of the API server to `--remote_rm_url` in training  scripts, e.g., `--remote_rm_url http://127.0.0.1:5000/get_reward`.



We provide three scripts of different training configurations:
- `train_ppo_ray_mcts_orm.sh`: Reinforcement Learning with MCTS sampling and only outcome-based reward.
- `train_ppo_ray_mcts_prm.sh`: Reinforcement Learning with MCTS sampling and process reward. Each process is divided by the functional token, *i.e*, tree nodes in the MCTS tree.
- `train_ppo_ray_normal_orm.sh`: Reinforcement Learning with random sampling and only outcome-based reward.

