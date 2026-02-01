# Reasoning Paths Construction
This is the code base for reasoning paths construction in SFT Warmup stage using functional tree search. The searching tree will be saved in directory `run_outputs`.

## Usage
First deploy the model through vllm:
```bash
CUDA_VISIBLE_DEVICES=0 nohup python -m vllm.entrypoints.openai.api_server --model /path/to/Qwen3-4B-Instruct-2507 --port 8001
```

Next depoly the PRM:
```bash
cd RFTT/gen_data
python prm_api.py
```

Finally run the script:
```bash
sh scripts/tree_search.sh
```

## Visualize
Use the following command:
```
python src/mcts.py
```

Visualization of the searching tree:

![tree](../images/tree.png)