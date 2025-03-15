# Two phase training
## Supervised fine-tuning
We use [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) to supervise fine-tuning initial models on reasoning pathe with learnable functional tokens. The configuration files in this stage are in `./sft`.

## Reinforcement Learning
We modify [OpenRLHF](https://github.com/OpenRLHF/OpenRLHF) to reinforce high-value reasoning paths through functional token-guided tree search.
### Environment Setup
Using conda:
```bash
conda create -n rftt python=3.10
conda activate rftt
cd OpenRLHF_MCTS
pip install -r requirements.txt
```
