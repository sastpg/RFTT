<h1 align="center">Reasoning with Reinforced Functional Token Tuning</h1>

<div align="center">
<a href='https://arxiv.org/abs/2502.13389'><img src='https://img.shields.io/badge/arXiv-2502.13389-%23B31B1B?logo=arxiv'></a> 
<a href='https://huggingface.co/sastpg/Qwen2.5-7B-Instruct-RFTT'><img src='https://img.shields.io/badge/Hugging_Face-Models-%23FFD21E?style=flat&logo=huggingface&logoColor=%23FFD21E'></a>
<a href='https://huggingface.co/datasets/sastpg/RFTT_Dataset'><img src='https://img.shields.io/badge/Hugging_Face-Datasets-blue?style=flat&logo=huggingface&logoColor=%23FFD21E'></a>
</div>

Official codebase for the paper **"[Reasoning with Reinforced Functional Token Tuning](https://arxiv.org/abs/2502.13389)"**.

<div align="center">
<img src="./images/framework.png" width="100%">
</div>

## Overview 

**Abstract:** In this work, we propose ***R*einforced *F*unctional *T*oken *T*uning** (RFTT), a novel reinforced fine-tuning framework that empowers Large Language Models (LLMs) with self-play learn-to-reason capabilities. Unlike prior prompt-driven reasoning efforts, RFTT embeds a rich set of learnable functional tokens (e.g., `<analyze>`, `<verify>`, `<refine>`) directly into the model vocabulary, enabling chain-of-thought construction with diverse human-like reasoning behaviors. Specifically, RFTT comprises two phases: 

- (1) Supervised fine-tuning performs prompt-driven tree search to obtain *self-generated* training data annotated with functional tokens, which warms up the model to learn these tokens for reasoning;
- (2) Online reinforcement learning further allows the model to explore different reasoning pathways through functional token sampling without relying on prompts, thereby facilitating effective *self-improvement* for functional reasoning.

Extensive experiments demonstrate the superiority of the proposed RFTT on mathematical benchmarks, significantly boosting Qwen-2.5-7B-Instruct (70.6% to 79.8%) and LLaMA-3.1-8B-Instruct (32.2% to 60.2%) on the MATH dataset. Moreover, the performance of RFTT consistently improves with more search rollouts at inference time. Our code will be made available.

## News

- **`[Feb 20, 2025]`** 🔥 Our [paper](https://arxiv.org/abs/2502.13389) for **RFTT** has been released!

## Get Started
### Requirements
We recommend using conda environment and run the following command to install the required packages:
```
conda create -n rftt python=3.10
conda activate rftt
cd train/OpenRLHF_MCTS
pip install -r requirements.txt
```

### SFT Data Generation
To conduct functional tree search that generates SFT data, please run the following command:
```
cd RFTT/gen_data
sh scripts/tree_search.sh
```
Key parameters in `tree_search.sh`:
- `dataset_name`: the dataset to generate SFT data, *e.g.*, MATH;
- `--model_ckpt`: the model to conduct tree search, *e.g.*, Qwen-2.5-7B-Instruct;
- `--num_rollouts`: the number of rollouts for MCTS;
- `--max_depth_allowed`: the maximum depth allowed for MCTS.

For more details about data construction and visualization, please refer to directory `RFTT/gen_data`.

### Two Phase Training
For **SFT Warmup**, we use [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) to fine-tune models. You can refer to the LLaMA-Factory repo for more details. We suggest using the following command to conduct SFT:
```
cd RFTT/train
git clone --depth 1 https://github.com/hiyouga/LLaMA-Factory.git
llamafactory-cli train SFT/full_sft.yaml
```

For **Online RL**, we use [OpenRLHF](https://github.com/OpenRLHF/OpenRLHF) to train the policy model. We provide example scripts to start training:
```
cd RFTT/train/OpenRLHF_MCTS
ray start --head --node-ip-address 0.0.0.0  # start the ray cluster
sh examples/scripts/train_ppo_ray_mcts_orm.sh  # training script
```
We list the description of key parameters in the RL training script:

| Parameters                 | Description                                                  |
| -------------------------- | ------------------------------------------------------------ |
| `--pretrain`               | Absolute path to the initial policy model after SFT Warmup.  |
| `--save_path`              | Absolute path to save the trained model.                     |
| `--prompt_data`            | Absolute path to instructions used for training.             |
| `--use_prm`                | Whether to use PRM for training. Only supported when `--sampling_method` is `mcts`. |
| `--remote_rm_url`          | The URL of the remote reward model. Only supported when `--use_prm` is enabled. |
| `--search_depth`           | The maximum depth allowed in MCTS.                           |
| `--sampling_method`        | Can only be `mcts` or `noraml` representing MCTS and random sampling, respectively. |

## Results Preview

<div align="center">
<img src="./images/training_curve.png" width="100%">
</div>

> The training curve of RFTT with and w/o PRM on Qwen-2.5-7B-Instruct during RL.


## Citation
If you find this work useful for your research, please cite our paper:

```
@article{rftt2025,
      title={Reasoning with Reinforced Functional Token Tuning}, 
      author={Kongcheng Zhang and Qi Yao and Baisheng Lai and Jiaxing Huang and Wenkai Fang and Dacheng Tao and Mingli Song and Shunyu Liu},
      journal={arXiv preprint arXiv:2502.13389},
      year={2025}
}
```

## Acknowledgements
- [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory)
- [OpenRLHF](https://github.com/OpenRLHF/OpenRLHF)

## Contact
Please feel free to contact me via email (zhangkc@zju.edu.cn) if you are interested in my research :)