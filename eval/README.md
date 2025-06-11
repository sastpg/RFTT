# Evaluation
## Code base
This is the code for evaluating the performance of the model.

For fast evaluation, we use the code snippet in `eval.py` to verify correctness of generated answers:
```python
is_equiv(remove_boxed(last_boxed_only_string(model_ans)), remove_boxed(last_boxed_only_string(golden_ans)))
```

For rigorous and comprehensive assessment in evaluation, we suggest using [Math-Verify](https://github.com/huggingface/Math-Verify) library to verify:
```python
from math_verify import verify, parse
verify(parse(golden_ans), parse(model_ans))
```

## Model deployment
We use [vLLM](https://github.com/vllm-project/vllm) to deploy a server that implements the OpenAI API protocol. Notably, the temperature is set to 0.0, and the max_tokens is set to 8192:

```bash
CUDA_VISIBLE_DEVICES=0 nohup python -m vllm.entrypoints.openai.api_server --model /path/to/Qwen2.5-7B-Instruct --port 8090 > /dev/null 2>&1 &
```

## Usage
Run `python inference.py` and get the sample output of the model. The results is saved in `results.jsonl`.

After that, you can use `python eval.py` to evaluate the performance of the model.