# Licensed under the MIT license.

import sys
import os, json, time
from tqdm import tqdm

sys.path.append(".")

from common.utils import fix_seeds, setup_model_parallel, read_json
from common.arguments import get_parser, post_process_args, save_args
from mcts_utils import GeneratorError
from mcts import Generator, search_for_answers
from common.eval import *

def main(args):
    fix_seeds(args.seed)
    if args.model_parallel:
        args.local_rank, args.world_size = setup_model_parallel()
    else:
        args.local_rank, args.world_size = 0, 1

    test_file = os.path.join(args.data_root, args.dataset_name, args.test_json_filename + ".json")
    assert os.path.exists(test_file), f"Test file {test_file} does not exist."
    data_item_list = read_json(test_file)

    evaluator = eval(f"{args.dataset_name}Evaluator()")

    tokenizer, model = None, None
    if args.api == "huggingface":
        from models.HuggingFace_API import load_HF_model

        tokenizer, model = load_HF_model(args.model_ckpt)
    elif args.api == "vllm":
        from models.vLLM_API import load_vLLM_model

        tokenizer, model = load_vLLM_model(args.model_ckpt, args.seed, args.tensor_parallel_size, args.half_precision)
    elif args.api == "gpt3.5-turbo":
        from models.OpenAI_API import load_OpenAI_model

        tokenizer, model = load_OpenAI_model(args.model_ckpt)
    generator = Generator(args, tokenizer, model, evaluator)

    total_correct = 0
    total_correct_limit = 0
    num_tested = 0
    start_time = time.time()

    offset = 0
    for i, data_item in enumerate(
        (pbar := tqdm(data_item_list, disable=args.local_rank > 0 or args.verbose, position=1))
    ):
        if i < args.start_idx or i >= args.end_idx:
            continue

        _, problem, gt_solution = None, data_item["problem"], data_item["output"]
        gt_answer = evaluator.extract_answer_from_gold_solution(gt_solution)

        js = {
            # "id": problem_id,
            "problem": problem,
            "model_completion": None,
            "model_answer": None,
            "all_model_completions": {},
            "gold_solution": gt_solution,
            "gold_answer": gt_answer,
        }

        model_solutions, stopping_id, model_all_solutions = [], -1, []

        # try:
        model_solutions, stopping_id, model_all_solutions = search_for_answers(
            args=args, user_question=problem, question_id=i+offset, gt_answer=gt_solution, generator=generator
        )

        num_tested += 1
        qid = i + offset
        with open(os.path.join(args.answer_sheets_dir, f"Question {qid:04d} - Answer.json"), "w") as f:
            json.dump(js, f)



if __name__ == "__main__":
    #! -------------------------------- Arguments --------------------------------
    parser = get_parser()

    parser.add_argument("--num_rollouts", type=int, default=15)
    parser.add_argument(
        "--num_subquestions", type=int, default=3, help="Number of trials for proposing the next subquestion"
    )
    parser.add_argument("--num_votes", type=int, default=10)
    parser.add_argument("--max_depth_allowed", type=int, default=5)

    # MCTS
    parser.add_argument("--mcts_discount_factor", type=float, default=1.0)
    parser.add_argument("--mcts_exploration_weight", type=float, default=2.0)
    parser.add_argument("--mcts_weight_scheduler", choices=["exp", "lin", "const"], default="const")
    parser.add_argument("--mcts_num_last_votes", type=int, default=None)
    parser.add_argument("--save_tree", action="store_true")

    # Action1: Propose an one-step thought.
    parser.add_argument("--num_a1_steps", type=int, default=None)
    parser.add_argument("--disable_a1", action="store_true")

    # Paraphrasing
    parser.add_argument("--modify_prompts_for_rephrasing", action="store_true")
    parser.add_argument("--disable_a5", action="store_true")

    #! -------------------------- Used for selecting answer --------------------------
    parser.add_argument("--enable_potential_score", action="store_true")

    #! -------------------------------------------------------------------------------

    args = parser.parse_args()

    if args.mcts_num_last_votes is None:
        args.mcts_num_last_votes = 32

    if not args.disable_a1:
        if args.num_a1_steps is None:
            args.num_a1_steps = 3

    #! ----------------------------------------------------------------------------

    prompts_dir = os.path.join(args.prompts_root, args.dataset_name)

    args.fewshot_cot_prompt_path = os.path.join(prompts_dir, "fewshot_cot", "fewshot_cot_prompt.txt")
    args.fewshot_cot_config_path = os.path.join(prompts_dir, "fewshot_cot", "fewshot_cot_config.json")

    args.fewshot_ost_prompt_path = os.path.join(prompts_dir, "fewshot_ost", "fewshot_ost_prompt.txt")
    args.fewshot_ost_config_path = os.path.join(prompts_dir, "fewshot_ost", "fewshot_ost_config.json")

    args.decompose_template_path = os.path.join(prompts_dir, "decompose", "decompose_template.json")
    args.decompose_prompt_path = os.path.join(prompts_dir, "decompose", "decompose_prompt.txt")

    args.direct_answers_prompt_path = os.path.join(prompts_dir, "direct_prompt_template.txt")
    args.ost_prompt_path = os.path.join(prompts_dir, "ost_prompt_template.txt")
    args.clarify_prompt_path = os.path.join(prompts_dir, "clarify_prompt_template.txt")
    args.analysis_prompt_path = os.path.join(prompts_dir, "analysis_prompt_template.txt")
    args.subquestion_prompt_path = os.path.join(prompts_dir, "subquestion_prompt_template.txt")
    args.subanswer_prompt_path = os.path.join(prompts_dir, "subanswer_prompt_template.txt")

    if not args.disable_a5:
        args.rephrasing_prompt_template_path = os.path.join(prompts_dir, "rephrasing_prompt_template.txt")
        if args.modify_prompts_for_rephrasing:
            args.fewshot_cot_prompt_rephrased_path = os.path.join(
                prompts_dir, "fewshot_cot", "fewshot_cot_prompt_rephrased.txt"
            )
            args.fewshot_ost_prompt_rephrased_path = os.path.join(
                prompts_dir, "fewshot_ost", "fewshot_ost_prompt_rephrased.txt"
            )
            args.decompose_prompt_rephrased_path = os.path.join(
                prompts_dir, "decompose", "decompose_prompt_rephrased.txt"
            )
        else:
            args.fewshot_cot_prompt_rephrased_path = os.path.join(prompts_dir, "fewshot_cot", "fewshot_cot_prompt.txt")
            args.fewshot_ost_prompt_rephrased_path = os.path.join(prompts_dir, "fewshot_ost", "fewshot_ost_prompt.txt")
            args.decompose_prompt_rephrased_path = os.path.join(prompts_dir, "decompose", "decompose_prompt.txt")

    args = post_process_args(args)
    print(args)
    save_args(args)
    main(args)
