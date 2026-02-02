# Licensed under the MIT license.

import os, json, torch, math
from argparse import ArgumentParser
from datetime import datetime


def get_parser():
    parser = ArgumentParser()

    parser.add_argument("--note", type=str, default="debug")

    allowed_apis = ["together", "huggingface", "llama", "vllm", "debug", "gpt3.5-turbo"]
    parser.add_argument(
        "--api", type=str, choices=allowed_apis, default="vllm", help=f"API to use: Choose from {allowed_apis}."
    )

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--verbose", action="store_true")

    #! WandB settings
    parser.add_argument("--wandb_mode", type=str, default="disabled", choices=["disabled", "online"])

    #! LLM settings
    parser.add_argument("--model_ckpt", required=True)
    parser.add_argument("--model_port", type=int, default=8001, help="port of vLLM server")

    parser.add_argument("--model_parallel", action="store_true")
    parser.add_argument("--half_precision", action="store_true")

    parser.add_argument("--max_tokens", type=int, default=1024, help="max_tokens")
    parser.add_argument("--temperature", type=float, default=0.8, help="temperature")
    parser.add_argument("--top_k", type=int, default=40, help="top_k")
    parser.add_argument("--top_p", type=float, default=0.95, help="top_p")
    parser.add_argument("--num_beams", type=int, default=1, help="num_beams")
    # parser.add_argument('--repetition_penalty', type=float, default=1.1, help='repetition_penalty')
    parser.add_argument("--max_num_worker", type=int, default=3, help="maximum number of workers for dataloader")
    parser.add_argument("--test_batch_size", type=int, default=1)  # batch_size
    parser.add_argument("--tensor_parallel_size", type=int, default=1)  # tensor_parallel_size

    #! prompt settings
    parser.add_argument("--prompts_root", default="prompts")

    #! dataset settings
    parser.add_argument("--data_root", default="data")
    allowed_dataset_names = ["MATH", "GSM8K", "GSM8KHARD", "STG", "SVAMP", "MULTIARITH"]
    parser.add_argument(
        "--dataset_name",
        required=True,
        choices=allowed_dataset_names,
        help=f"Test dataset name: Choose from {allowed_dataset_names}.",
    )
    parser.add_argument("--test_json_filename", type=str, default="test_all")
    parser.add_argument("--start_idx", type=int, default=0, help="Start index of test questions (inclusive)")
    parser.add_argument("--end_idx", type=int, default=math.inf, help="End index of test questions (inclusive))")

    #! outputs settings
    parser.add_argument("--run_outputs_root", type=str, default="run_outputs")
    parser.add_argument("--eval_outputs_root", type=str, default="eval_outputs")

    return parser


def post_process_args(args):
    # Set up logging
    suffix = "---[" + args.note + "]" if args.note is not None else ""
    model_name = args.model_ckpt.split("/")[-1]
    args.run_outputs_dir = os.path.join(
        args.run_outputs_root,
        args.dataset_name,
        model_name,
        f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}" + suffix,
    )
    os.makedirs(args.run_outputs_dir, exist_ok=True)

    args.answer_sheets_dir = os.path.join(args.run_outputs_dir, "answer_sheets")
    os.makedirs(args.answer_sheets_dir, exist_ok=True)

    return args
