# Licensed under the MIT license.

import sys

sys.path.append(".")

from openai import OpenAI

def generate_with_vLLM_model(
    model,
    port,
    input,
    temperature=0.7, #0..7
    top_p=0.6, #0.8
    top_k=50, #50
    repetition_penalty=1.05,
    n=1,
    max_tokens=512,
    logprobs=1,
    stop=[],
):
    client = OpenAI(
        base_url=f"http://localhost:{port}/v1",
        api_key="0",
    )
    parameters = {
        "temperature": temperature,
        "max_tokens": max_tokens,
        # "top_p": top_p,
        "stop": stop,
        # "logprobs": logprobs,
    }
    completion = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "user", "content": input}
        ],
        # extra_body={
        #     "repetition_penalty": repetition_penalty,
        #     "top_k": top_k,
        # },
        **parameters
    )

    return completion.choices[0].message.content

def load_vLLM_model(model_ckpt, seed, tensor_parallel_size=1, half_precision=False, max_num_seqs=256):
    return None, model_ckpt

class IO_System:
    """Input/Output system"""

    def __init__(self, args, tokenizer, model) -> None:
        self.api = args.api
        self.port = args.model_port
        self.model_ckpt = args.model_ckpt
        self.temperature = args.temperature
        self.top_k = args.top_k
        self.top_p = args.top_p
        self.tokenizer = tokenizer
        self.model = model

        self.call_counter = 0
        self.token_counter = 0

    def generate(self, model_input, max_tokens: int, num_return: int, stop_tokens):
        if isinstance(model_input, str):
            if self.api == "vllm":
                io_output_list = []
                for _ in range(num_return):
                    vllm_response = generate_with_vLLM_model(
                        self.model,
                        self.port,
                        input=model_input,
                        # temperature=self.temperature,
                        # top_p=self.top_p,
                        # top_k=self.top_k,
                        n=num_return,
                        max_tokens=max_tokens,
                        stop=stop_tokens,
                    )
                    io_output_list.append(vllm_response)
                    
                    self.call_counter += 1
                    self.token_counter += 10
            else:
                raise NotImplementedError(f"API {self.api} is not implemented.")
        elif isinstance(model_input, list):
            if self.api == "vllm":
                io_output_list = []
                for _ in range(num_return):
                    vllm_response = generate_with_vLLM_model(
                        self.model,
                        self.port,
                        input=model_input,
                        # temperature=self.temperature,
                        # top_p=self.top_p,
                        # top_k=self.top_k,
                        n=num_return,
                        max_tokens=max_tokens,
                        stop=stop_tokens,
                    )
                    io_output_list.append(vllm_response)
                    
                    self.call_counter += 1
                    self.token_counter += 10
            else:
                raise NotImplementedError(f"API {self.api} is not implemented.")

        return io_output_list
