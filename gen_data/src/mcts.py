# Licensed under the MIT license.
import re
import sys
import pickle
sys.path.append(".")

import os, json
from tqdm import trange
from typing import List, Dict
from copy import deepcopy

try:
    from rapidfuzz import fuzz, process
except:
    pass

from common.model import IO_System
from common.utils import read_txt, read_json
from common.eval import remove_boxed, last_boxed_only_string, is_equiv
from mcts_backbone import MCTS_Searcher, MCTS_Node
from mcts_utils import (
    Node_Type,
    reach_terminal_step,
    concat_solution_trace,
    concat_solution_trace_for_prm,
    cal_reward,
    get_sentences,
    print_tree_from_root,
    find_valid_solution_nodes,
    stochastic_find_best_solution,
)

def verbose_print(s: str, verbose: bool):
    if verbose:
        print(s)


class Generator:
    """Generator generates children nodes"""
    def __init__(self, args, tokenizer, model) -> None:
        self.io = IO_System(args, tokenizer, model)
        self.evaluator = None

        self.max_tokens = args.max_tokens
        self.enable_potential_score = args.enable_potential_score

        self.direct_answers_prompt = read_txt(args.direct_answers_prompt_path)
        self.ost_prompt = read_txt(args.ost_prompt_path)
        self.clarify_prompt = read_txt(args.clarify_prompt_path)
        self.analysis_prompt = read_txt(args.analysis_prompt_path)
        self.subquestion_prompt = read_txt(args.subquestion_prompt_path)
        self.subanswer_prompt = read_txt(args.subanswer_prompt_path)

    def generate_direct_answers(self, user_question: str, paraphrased: bool, solution_trace: Dict[int, Dict[str, str]]):
        direct_answer_list = []

        #! few shot cot
        num_return = 1

        direct_answers_prompt = self.direct_answers_prompt
        existing_solutions = concat_solution_trace(solution_trace)
        io_input = direct_answers_prompt + user_question + '\nExisting Steps:\n' + existing_solutions
        io_output_list = self.io.generate(
            io_input,
            num_return=num_return,
            max_tokens=4096,
            stop_tokens=[]
        )
        direct_answer_list = [io_output.strip() for io_output in io_output_list]
        
        
        return direct_answer_list

    def generate_subquestions(
        self,
        user_question: str,
        solution_trace: Dict[int, Dict[str, str]],
        paraphrased: bool,
    ):
        subquestion_list, reward_list = [], []
        subquestion_prompt = self.subquestion_prompt
        existing_solutions = concat_solution_trace(solution_trace)

        io_input = subquestion_prompt + user_question + '\nExisting Steps:\n' + existing_solutions
        io_output_list = self.io.generate(
            io_input,
            max_tokens=256,
            num_return=2,
            stop_tokens=[
                "\n",
                "\n\n",
            ],
        )
        subquestion_list = [o.strip() for o in io_output_list]


        return subquestion_list

    def generate_subanswers(
        self,
        user_question: str,
        solution_trace: Dict[int, Dict[str, str]],
        paraphrased: bool,
    ):
        subanswer_list, reward_list = [], []

        subanswer_prompt = self.subanswer_prompt
        existing_solutions = concat_solution_trace(solution_trace, parent_is_subquestion=True)
        io_input = subanswer_prompt + user_question + '\nExisting Steps:\n' + existing_solutions + "\n\nLet's solve the sub-problem step by step."
        
        io_output_list = self.io.generate(
            io_input,
            num_return=1,
            max_tokens=4096,
            stop_tokens=[],
        )
        subanswer_list = [o.strip() for o in io_output_list]
        # for suba in subanswer_list:
        #     reward = cal_reward(concat_solution_trace_for_prm(solution_trace) + suba + " ки\n\n")
        #     reward_list.append(reward["score"][-1])

        return subanswer_list

    def generate_clarify(self, user_question: str):
        clarify_list = []
        io_input = self.clarify_prompt
        io_input += "\n\n"
        io_input += "Original Question: " + user_question + "\n"
        io_input += "Clarified Question: "
        io_output_list = self.io.generate(model_input=io_input, max_tokens=512, num_return=2, stop_tokens=["Original"])
        # clarify_user_question_list = [io_output.rstrip("Original ").strip() for io_output in io_output_list]
        for io_output in io_output_list:
            clarify_output = re.sub(r"^Clarified Question:\s*", "", io_output)
            clarify_output = clarify_output.rstrip("Original ").strip()
            sentences = get_sentences(clarify_output)
            if len(sentences) > 3:
                clarify_output = " ".join(sentences[:3])
            clarify_list.append(clarify_output)
        
        
        return clarify_list

    def generate_analysis_question(
        self,
        user_question: str,
        solution_trace: Dict[int, Dict[str, str]],
        paraphrased: bool,
    ):
        analysis_question_list, reward_list = [], []
        existing_solutions = concat_solution_trace(solution_trace)
        analysis_prompt = self.analysis_prompt
        io_input = analysis_prompt + user_question + '\nExisting Steps:\n' + existing_solutions
        io_output_list = self.io.generate(
            io_input,
            max_tokens=512,
            num_return=2,
            stop_tokens=[]
        )
        analysis_question_list = [io_output.strip().strip("Brief analyse:").strip() for io_output in io_output_list]
        for analysis_question in analysis_question_list:
            # 特殊处理
            if "Brief analyse" in analysis_question:
                pattern = r"Briefly analyse:(.*?)(?=\.\s|\n|\n\n)"
                match = re.search(pattern, analysis_question, re.DOTALL)
                
                if match:
                    analysis_question = match.group(1).strip() + "."


        return analysis_question_list

    def generate_next_step(
        self,
        user_question: str,
        solution_trace: Dict[int, Dict[str, str]],
        paraphrased: bool = False,
        parent_is_subquestion: bool = False,
    ):
        next_step_list, reward_list = [], []
        existing_solutions = concat_solution_trace(solution_trace, parent_is_subquestion)
        ost_prompt = self.ost_prompt
        io_input = ost_prompt + user_question + '\nExisting Steps:\n' + existing_solutions
        io_output_list = self.io.generate(model_input=io_input, max_tokens=512, num_return=2, stop_tokens=[])
        # ost_step_list = [io_output.strip().lstrip("Next step:").strip() for io_output in io_output_list]
        for io_output in io_output_list:
            standard_output = ""
            pattern = r"[Next|First] step:(.*?)\n(.*?)(?=\Z)"
            match = re.search(pattern, io_output, re.DOTALL)
            if match:
                step_name = match.group(1).strip()
                process = match.group(2).strip()
                process = re.sub(r"^Specific process:\s*", "", process)
                if not step_name.startswith("**"):
                    step_name = "**" + step_name + "**\n"
                standard_output = step_name + process
            else:
                standard_output = io_output
            next_step_list.append(standard_output)

        return next_step_list


class Reasoning_MCTS_Node(MCTS_Node):
    def __init__(
        self,
        parent: "Reasoning_MCTS_Node",
        depth: int,
        node_type: Node_Type,
        verbose: bool = False,
        # --- For instantiating root node ---
        node_value: float = None,
        process_reward: float = None,
        generator: Generator = None,
        user_question: str = None,
        expected_answer: str = None,
        max_depth_allowed: int = None,
        # --- For instantiating CLARIFY node ---
        clarify: str = None,
        # --- For instantiating ANALYSIS node ---
        analysis: str = None,
        # --- For instantiating DIRECT_ANSWER node ---
        direct_answer: str = None,
        # --- For instantiating SUBQUESTION node ---
        subquestion: str = None,
        is_new_subquestion: bool = None,
        # --- For instantiating NEXT_STEP node ---
        next_step: str = None,
        # --- For instantiating node information ---
        sentence_num: int = None,
        Q: int = 0,
        N: int = 0,
    ) -> None:
        super().__init__()

        #! sanity checks
        try:
            assert depth is not None
            assert node_type is not None
            if node_value is not None:
                assert node_value >= 0, breakpoint()

            if node_type is Node_Type.USER_QUESTION:
                assert depth == 0
                assert all(
                    attr is None
                    for attr in [
                        parent,
                        node_value,
                        clarify,
                        direct_answer,
                        subquestion,
                        is_new_subquestion,
                        next_step,
                    ]
                )
                assert all(
                    attr is not None
                    for attr in [generator, user_question, expected_answer, max_depth_allowed]
                )
            elif node_type is Node_Type.CLARIFY:
                assert depth == 1
                assert all(
                    attr is None
                    for attr in [
                        node_value,
                        generator,
                        user_question,
                        expected_answer,
                        direct_answer,
                        subquestion,
                        is_new_subquestion,
                        next_step,
                        max_depth_allowed,
                    ]
                )
                assert all(attr is not None for attr in [parent, clarify])
            elif node_type is Node_Type.DIRECT_ANSWER:
                assert depth > 0
                assert all(
                    attr is None
                    for attr in [
                        generator,
                        user_question,
                        expected_answer,
                        subquestion,
                        is_new_subquestion,
                        next_step,
                        max_depth_allowed,
                    ]
                )
                assert all(attr is not None for attr in [parent, node_value, direct_answer])
            elif node_type is Node_Type.SUBQUESTION:
                assert depth > 0
                assert all(
                    attr is None
                    for attr in [
                        generator,
                        node_value,
                        user_question,
                        expected_answer,
                        direct_answer,
                        next_step,
                        max_depth_allowed,
                    ]
                )
                assert all(
                    attr is not None for attr in [parent, subquestion, is_new_subquestion]
                )
            elif node_type is Node_Type.SUBANSWER:
                assert depth > 0
                assert all(
                    attr is None
                    for attr in [
                        generator,
                        user_question,
                        node_value,
                        expected_answer,
                        direct_answer,
                        subquestion,
                        is_new_subquestion,
                        next_step,
                        max_depth_allowed,
                    ]
                )
                assert all(attr is not None for attr in [parent])
            elif node_type is Node_Type.NEXT_STEP:
                assert depth > 0
                assert all(
                    attr is None
                    for attr in [
                        generator,
                        user_question,
                        clarify,
                        expected_answer,
                        direct_answer,
                        subquestion,
                        is_new_subquestion,
                        max_depth_allowed,
                    ]
                )
                assert all(attr is not None for attr in [parent, next_step])
        except AssertionError:
            print(f"Instantiating node with type {node_type} failed!")
            breakpoint()
            exit()

        #! attributes
        self.parent = parent  # if parent is None, then the node is the root
        self.children: List["Reasoning_MCTS_Node"] = []
        self.depth = depth
        self.node_type = node_type
        self.node_value = node_value
        self.direct_answer = direct_answer
        self.subquestion = subquestion
        self.is_new_subquestion = is_new_subquestion
        self.next_step = next_step
        self.process_reward = process_reward
        self.clarify = clarify
        self.analysis = analysis
        self.sentence_num = sentence_num
        self.Q = Q
        self.N = N

        if parent is None:  # root
            self.verbose = verbose
            self.user_question = user_question
            self.expected_answer = expected_answer
            self.generator = generator
            self.max_depth_allowed = max_depth_allowed
        else:  # inherit from parent
            self.verbose = parent.verbose
            self.user_question = parent.user_question
            self.expected_answer = parent.expected_answer
            self.generator = parent.generator
            self.max_depth_allowed = parent.max_depth_allowed

        #! keep track of paraphrasing
        if node_type is Node_Type.USER_QUESTION:
            self.paraphrased = False
        elif node_type is Node_Type.CLARIFY:
            self.paraphrased = True
        else:
            assert parent is not None
            self.paraphrased = parent.paraphrased

        #! record number of subquestions till now
        # if parent is None:  # root
        #     self.subquestion_counter = 0
        # else:
        #     if node_type is Node_Type.SUBQUESTION and is_new_subquestion:
        #         self.subquestion_counter = parent.subquestion_counter + 1
        #     else:
        #         self.subquestion_counter = parent.subquestion_counter

        #! record number of one-step thought steps till now
        # if parent is None:  # root
        #     self.next_step_counter = 0
        # else:
        #     if node_type is Node_Type.NEXT_STEP:
        #         self.next_step_counter = parent.next_step_counter + 1
        #     else:
        #         self.next_step_counter = parent.next_step_counter

        #! record solution trace from root to the current node. key: subquestion id
        if parent is None:  # root
            assert self.node_type is Node_Type.USER_QUESTION
            self.solution_trace: Dict[int, Dict[str, str]] = {0: {"user_question": user_question, "expected_answer": expected_answer}}
        else:
            assert self.node_type is not Node_Type.USER_QUESTION
            self.solution_trace = deepcopy(parent.solution_trace)

            if node_type is Node_Type.CLARIFY:
                assert self.depth == parent.depth + 1
                self.solution_trace[self.depth] = {"clarify": clarify, "process_reward": process_reward}
            elif node_type is Node_Type.ANALYSIS:
                assert self.depth == parent.depth + 1
                self.solution_trace[self.depth] = {"analysis": analysis, "process_reward": process_reward}
            elif node_type is Node_Type.DIRECT_ANSWER:
                assert self.depth == parent.depth + 1
                self.solution_trace[self.depth] = {
                    "direct_answer": direct_answer,
                    "process_reward": process_reward,
                    "node_value": node_value
                }
            elif node_type is Node_Type.SUBQUESTION:
                assert self.depth == parent.depth + 1
                self.solution_trace[self.depth] = {
                    "subquestion": subquestion,
                    "process_reward": process_reward,
                }
            elif node_type is Node_Type.NEXT_STEP:
                assert self.depth == parent.depth + 1
                self.solution_trace[self.depth] = {"next_step": next_step, "process_reward": process_reward, "node_value": node_value}


    def __str__(self) -> str:
        type2str = {
            Node_Type.USER_QUESTION: "U",
            Node_Type.CLARIFY: "CL",
            Node_Type.ANALYSIS: "AN",
            Node_Type.DIRECT_ANSWER: "DA",
            Node_Type.SUBQUESTION: "SQ",
            Node_Type.SUBANSWER: "SA",
            Node_Type.NEXT_STEP: "NS",
        }
        return f"{type2str[self.node_type]}-{self.id}"

    def _create_children(self):
        def do_action_generate_direct_answers():
            verbose_print(f"---- Generating direct answers for node {self.id}...", self.verbose)
            direct_answer_list = self.generator.generate_direct_answers(
                user_question=self.user_question, paraphrased=self.paraphrased, solution_trace=self.solution_trace
            )
            for direct_answer in direct_answer_list:
                sentence_num = max(len(get_sentences(direct_answer)), len(direct_answer.split("\n\n")))
                value = 0
                if is_equiv(remove_boxed(last_boxed_only_string(direct_answer)), remove_boxed(last_boxed_only_string(self.expected_answer))):
                    value = 1
                self.children.append(
                    Reasoning_MCTS_Node(
                        parent=self,
                        depth=self.depth + 1,
                        node_type=Node_Type.DIRECT_ANSWER,
                        node_value=value,
                        direct_answer=direct_answer,
                        sentence_num=sentence_num
                    )
                )

        def do_action_generate_subquestions():
            verbose_print(f"---- Generating subquestions for node {self.id}...", self.verbose)

            #! ACTION: generate new subquestions
            subquestion_list = self.generator.generate_subquestions(
                user_question=self.user_question, solution_trace=self.solution_trace, paraphrased=self.paraphrased
            )
            
            for subquestion in subquestion_list:
                self.children.append(
                    Reasoning_MCTS_Node(
                        parent=self,
                        depth=self.depth + 1,
                        node_type=Node_Type.SUBQUESTION,
                        subquestion=subquestion,
                        is_new_subquestion=True
                    )
                )

        def do_action_generate_clarify():
            verbose_print(f"---- Generating clarified user question for node {self.id}...", self.verbose)
            #! ACTION: generate paraphrased question for the root question
            clarify_user_question_list = self.generator.generate_clarify(
                user_question=self.user_question
            )
            
            for clarify_user_question in clarify_user_question_list:
                self.children.append(
                    Reasoning_MCTS_Node(
                        parent=self,
                        depth=self.depth + 1,
                        node_type=Node_Type.CLARIFY,
                        clarify=clarify_user_question
                    )
                )

        def do_action_generate_next_step(parent_is_subquestion: bool=False):
            verbose_print(f"---- Generating one-step thought steps for node {self.id}...", self.verbose)

            #! ACTION: generate one-step thought step
            next_step_list = self.generator.generate_next_step(
                user_question=self.user_question,
                solution_trace=self.solution_trace,
                paraphrased=self.paraphrased,
                parent_is_subquestion=parent_is_subquestion
            )
            for next_step in next_step_list:
                value = None
                if "Final step" in next_step or "So the answer is" in next_step:
                    if is_equiv(remove_boxed(last_boxed_only_string(next_step)), remove_boxed(last_boxed_only_string(self.expected_answer))):
                        value = 1
                    else:
                        value = 0
                self.children.append(
                    Reasoning_MCTS_Node(
                        parent=self,
                        depth=self.depth + 1,
                        node_type=Node_Type.NEXT_STEP,
                        next_step=next_step,
                        node_value=value
                    )
                )

        def do_action_generate_analysis():
            verbose_print(f"---- Generating analysis for node {self.id}...", self.verbose)
            analysis_list = self.generator.generate_analysis_question(
                user_question=self.user_question,
                solution_trace=self.solution_trace,
                paraphrased=self.paraphrased,
            )
            for analysis in analysis_list:
                self.children.append(
                    Reasoning_MCTS_Node(
                        parent=self,
                        depth=self.depth + 1,
                        node_type=Node_Type.ANALYSIS,
                        analysis=analysis,
                    )
                )

        #! create children
        if self.node_type is Node_Type.USER_QUESTION:
            # a1: clarify
            do_action_generate_clarify()
            # a2: analysis
            do_action_generate_analysis()
            # a3: subquestion
            do_action_generate_subquestions()
            # a4: next_step
            do_action_generate_next_step()
            # a5: direct_answer
            do_action_generate_direct_answers()
        elif self.node_type is Node_Type.CLARIFY:
            do_action_generate_analysis()
            do_action_generate_next_step()
            do_action_generate_direct_answers()
            do_action_generate_subquestions()
        elif self.node_type is Node_Type.ANALYSIS:
            do_action_generate_direct_answers()
            do_action_generate_next_step()
            do_action_generate_subquestions()
        elif self.node_type is Node_Type.DIRECT_ANSWER:
            raise ValueError("DIRECT_ANSWER node cannot create children!!")
        elif self.node_type is Node_Type.SUBQUESTION:
            do_action_generate_direct_answers()
            do_action_generate_next_step(parent_is_subquestion=True)
        elif self.node_type is Node_Type.NEXT_STEP:
            do_action_generate_next_step()
            do_action_generate_direct_answers()
            do_action_generate_subquestions()
        else:
            raise ValueError("Invalid node type!!")

        assert self.children
        return self.children

    def is_valid_leaf_node(self):
        #! a valid solution can only be in SUBQUESTION type or DIRECT_ANSWER type
        return (
            self.node_type is Node_Type.NEXT_STEP and reach_terminal_step(self.next_step)
        ) or self.node_type is Node_Type.DIRECT_ANSWER

    def is_valid_solution_node(self):
        #! a valid solution can only be in SUBQUESTION type or DIRECT_ANSWER type or NEXT_STEP type
        return (
            (self.node_type is Node_Type.NEXT_STEP and reach_terminal_step(self.next_step))
            or self.node_type is Node_Type.DIRECT_ANSWER
        )

    def set_potential_score(self, score: float):
        self.potential_score = score

    def find_children(self, rollout_id: int):
        self.children = self.children or self._create_children()
        for child in self.children:
            child.set_rollout_id(rollout_id)
        assert self.children
        return self.children

    def is_terminal(self):
        return self.depth >= self.max_depth_allowed or self.is_valid_leaf_node()

    def calculate_reward(self):
        if self.is_valid_leaf_node():
            assert self.node_value is not None, breakpoint()
            return self.node_value
        else:
            return 0

    def get_traces_and_rewards(self):
        trajectory, content, repeat = [], [], 0
        cur_node = self
        while cur_node.parent:
            trajectory.append(cur_node.id)
            if cur_node.node_type is Node_Type.CLARIFY:
                content.append(cur_node.clarify)
            elif cur_node.node_type is Node_Type.ANALYSIS:
                content.append(cur_node.analysis)
            elif cur_node.node_type is Node_Type.NEXT_STEP:
                content.append(cur_node.next_step)
            elif cur_node.node_type is Node_Type.SUBQUESTION:
                content.append(cur_node.subquestion)
            elif cur_node.node_type is Node_Type.DIRECT_ANSWER:
                content.append(cur_node.direct_answer)
            cur_node = cur_node.parent
        
        unique_elements = set(content)
        repeat = len(content) - len(unique_elements)
        
        return list(reversed(trajectory)), repeat
            
        

def search_for_answers(args, user_question: str, question_id: int, gt_answer: str, generator: Generator):
    verbose_print(
        f"********************* Searching for answers to question {question_id} ********************* ", args.verbose
    )

    #! build an MCTS searcher
    mcts_searcher = MCTS_Searcher(
        exploration_weight=args.mcts_exploration_weight,
        weight_scheduler=args.mcts_weight_scheduler,
        num_rollouts=args.num_rollouts,
        discount=args.mcts_discount_factor,
        verbose=args.verbose,
    )

    #! build the MCTS tree
    root_node = Reasoning_MCTS_Node(
        parent=None,
        depth=0,
        node_type=Node_Type.USER_QUESTION,
        verbose=args.verbose,
        generator=generator,
        user_question=user_question,
        expected_answer=gt_answer,
        max_depth_allowed=args.max_depth_allowed,
        process_reward=0,
    )

    model_solutions = []
    model_all_solutions = []
    model_rollout_nodes = []
    model_rollout_best = []
    for i in (pbar := trange(args.num_rollouts, disable=True, position=0)):
        rollout_node = mcts_searcher.do_rollout(root_node, i)
        model_rollout_nodes.append(rollout_node)

        all_solution_nodes, all_solutions = stochastic_find_best_solution(
            root_node, generator.evaluator, enable_potential_score=args.enable_potential_score
        )
        # model_solutions.append(best_solution)
        model_all_solutions.append(all_solutions)
        # model_rollout_best.append(chosen_node)

    if args.save_tree:
        with open(
            os.path.join(
                args.answer_sheets_dir,
                f"Question {question_id:04d}.tree",
            ),
            "w",
        ) as f:
            print_tree_from_root(
                mcts_searcher=mcts_searcher,
                rollout_id=15,
                root_node=root_node,
                chosen_node=None,
                file=f,
            )
    
    with open(f"{args.answer_sheets_dir}/tree_{question_id:04d}.pkl", 'wb') as f:
        pickle.dump(root_node, f)
    #! record final traces
    # js = [{"trace": node.solution_trace, "rollout_id": node.rollout_id} for node in all_solution_nodes]
    js = []
    for node in all_solution_nodes:
        input_for_prm = concat_solution_trace_for_prm(node.solution_trace)
        step_reward = cal_reward(input_for_prm)
        trajectory, repeat_num = node.get_traces_and_rewards()
        js.append({"trace": node.solution_trace, "rollout_id": node.rollout_id, "trajectory": trajectory, "reward_list": step_reward, "repeat_num": repeat_num})
        # node.solution_trace["rollout_id"] = node.rollout_id
        
    with open(os.path.join(args.answer_sheets_dir, f"Question {question_id:04d} - Final Solutions.json"), "w") as f:
        json.dump(js, f)

    js2 = [{"trace": node.solution_trace, "rollout_id": i} for i, node in enumerate(model_rollout_nodes)]
    with open(os.path.join(args.answer_sheets_dir, f"Question {question_id:04d} - Rollout Solutions.json"), "w") as f:
        json.dump(js2, f)

    js3 = [{"trace": node.solution_trace, "rollout_id": i} for i, node in enumerate(model_rollout_best)]
    with open(os.path.join(args.answer_sheets_dir, f"Question {question_id:04d} - Rollout Best.json"), "w") as f:
        json.dump(js3, f)

    if args.enable_potential_score:
        js = [node.potential_answers_history for node in all_solution_nodes]
        with open(os.path.join(args.answer_sheets_dir, f"Question {question_id:04d} - Potentials.json"), "w") as f:
            json.dump(js, f)

    return model_solutions, i, model_all_solutions

from graphviz import Digraph
def sub_plot(dot, root):
    for child in root.children:
        # print(child)
        if child.node_value == None:
            color = 'Black'
        elif child.node_value == 0:
            color = 'Red'
        elif child.node_value == 1:
            color = 'Green'
        
        content = ""
        if child.node_type is Node_Type.CLARIFY:
            content = child.clarify
        elif child.node_type is Node_Type.ANALYSIS:
            content = child.analysis
        elif child.node_type is Node_Type.NEXT_STEP:
            content = child.next_step
        elif child.node_type is Node_Type.SUBQUESTION:
            content = child.subquestion
        elif child.node_type is Node_Type.DIRECT_ANSWER:
            content = child.direct_answer
        str2 = 'Q: ' + str(child.Q) + '; N: ' + str(child.N) + '\nPR: ' + str(child.process_reward) + '\nType: ' + str(child.node_type) #+ '\nContent:' + str(content)
        dot.node(str(child.id), str2, color=color)
        dot.edge(str(root.id), str(child.id), str(child.id - 1))
        sub_plot(dot, child)
    
if __name__ == "__main__":
    with open('./run_outputs/MATH/Qwen3-4B-Instruct-2507/2026-02-01_15-18-23---[default]/answer_sheets/tree_0000.pkl', 'rb') as f:
        loaded_root = pickle.load(f)
    
    # all_solution_nodes = find_valid_solution_nodes(loaded_root)

    # js = []
    # for node in all_solution_nodes:
    #     trajectory, reward_list, repeat_num = node.get_traces_and_rewards()
    #     if node.node_type is Node_Type.NEXT_STEP:
    #         node.solution_trace[node.depth]["node_value"] = node.node_value
    #     js.append({"trace": node.solution_trace, "rollout_id": node.rollout_id, "trajectory": trajectory, "reward_list": reward_list, "repeat_num": repeat_num})
    #     # node.solution_trace["rollout_id"] = node.rollout_id
        
    # with open("001_result.json", "w") as f:
    #     json.dump(js, f)

    dot = Digraph('Tree', filename='tree.gv')
    str1 = 'N: ' + str(loaded_root.N) + '\nQ: ' + str(loaded_root.Q) + '\nType: ' + str(loaded_root.node_type)
    dot.node(str(loaded_root.id), str1)
    sub_plot(dot, loaded_root)

    print(dot.source)