# Licensed under the MIT license.
import re
import sys
import pickle
sys.path.append(".")
import torch
import numpy as np, os, random, json, math, wandb
from tqdm import trange
from typing import List, Dict, Tuple
from copy import deepcopy
from typing import List, Dict, Optional


from .eval import remove_boxed, last_boxed_only_string, is_equiv
from .mcts_backbone import MCTS_Searcher, MCTS_Node
from .mcts_utils import (
    Node_Type,
    concat_solution_trace,
    concat_solution_trace_for_prm,
    cal_reward,
)


def verbose_print(s: str, verbose: bool):
    if verbose:
        print(s)


# class Generator:
#     """Generator generates children nodes"""

#     def __init__(self, model) -> None:
#         self.model = model
#         self.max_new_tokens = 512
#         self.stop = ['</next_step>', '</clarify>', '</analysis>', '</subquestion>', '</verify>', '</refine>', '</output>', '</direct_answer>']
#         self.topk = 12
#         self.model.llm_engine.model_config.max_logprobs = self.topk + 1
#         self.tokenizer = self.model.llm_engine.tokenizer.tokenizer

#     def predict_next_action(self, solution_trace: Dict[int, Dict[str, str]]):
#         # breakpoint()
#         # Greedy decoding for only the first token of the response.
#         input_sequence = concat_solution_trace(solution_trace) + "<"
#         sampling_params = SamplingParams(n=1, temperature=0.1, max_tokens=1, logprobs=self.topk, stop=self.stop)
#         outputs = self.model.generate(input_sequence, sampling_params, use_tqdm=False)[0].outputs[0].logprobs[0]

#         topk_tokens = {'decoded': [], 'probs': [], 'token_id': [], 'logprobs': []}

#         for token_id, logprob_obj in outputs.items():
            
#             topk_tokens['logprobs'].append({token_id: logprob_obj})
#             topk_tokens['decoded'].append(logprob_obj.decoded_token.strip())
#             topk_tokens['probs'].append(logprob_obj.logprob)
#             topk_tokens['token_id'].append(token_id)

#         topk_tokens['probs'] = torch.exp(torch.tensor(topk_tokens['probs'])).tolist()
#         # breakpoint()

#         all_actions = ["clarify", "analysis", "next_step", "direct_answer", "subquestion", "verify", "refine", "output"]
#         topk_actions = []
#         for action in topk_tokens['decoded']:
#             if action in "clarify" and "clarify" in all_actions:
#                 topk_actions.append("clarify")
#                 all_actions.remove("clarify")
#             elif action in "analysis" and "analysis" in all_actions:
#                 topk_actions.append("analysis")
#                 all_actions.remove("analysis")
#             elif action in "next_step" and "next_step" in all_actions:
#                 topk_actions.append("next_step")
#                 all_actions.remove("next_step")
#             elif action in "direct_answer" and "direct_answer" in all_actions:
#                 topk_actions.append("direct_answer")
#                 all_actions.remove("direct_answer")
#             elif action in "subquestion" and "subquestion" in all_actions:
#                 topk_actions.append("subquestion")
#                 all_actions.remove("subquestion")
#             elif action in "verify" and "verify" in all_actions:
#                 topk_actions.append("verify")
#                 all_actions.remove("verify")
#             elif action in "refine" and "refine" in all_actions:
#                 topk_actions.append("refine")
#                 all_actions.remove("refine")
#             elif action in "output" and "output" in all_actions:
#                 topk_actions.append("output")
#                 all_actions.remove("output")
        

#         topk_actions = topk_actions + all_actions
#         # breakpoint()
#         return topk_actions

#     def generate_ost_step(self, solution_trace: Dict[int, Dict[str, str]]):
#         input_sequence = concat_solution_trace(solution_trace) + "<next_step>\n"
#         sampling_params = SamplingParams(n=1, temperature=0.1, top_p=0.95, max_tokens=self.max_new_tokens, logprobs=2, stop=self.stop)
#         outputs = self.model.generate(input_sequence, sampling_params, use_tqdm=False)

#         return outputs[0].outputs[0].text.strip()


#     def generate_direct_answers(self, solution_trace: Dict[int, Dict[str, str]]):
#         input_sequence = concat_solution_trace(solution_trace) + "<direct_answer>\n"
#         sampling_params = SamplingParams(n=1, temperature=0.1, top_p=0.95, max_tokens=self.max_new_tokens, logprobs=2, stop=self.stop)
#         outputs = self.model.generate(input_sequence, sampling_params, use_tqdm=False)

#         return outputs[0].outputs[0].text.strip()

#     def generate_subquestions(
#         self,
#         solution_trace: Dict[int, Dict[str, str]]
#     ):
#         input_sequence = concat_solution_trace(solution_trace) + "<subquestion>\n"
#         sampling_params = SamplingParams(n=1, temperature=0.1, top_p=0.95, max_tokens=self.max_new_tokens, logprobs=2, stop=self.stop)
#         outputs = self.model.generate(input_sequence, sampling_params, use_tqdm=False)

#         return outputs[0].outputs[0].text.strip()


#     def generate_clarify_user_question(self, solution_trace: Dict[int, Dict[str, str]]):
#         input_sequence = concat_solution_trace(solution_trace) + "<clarify>\n"
#         sampling_params = SamplingParams(n=1, temperature=0.1, top_p=0.95, max_tokens=self.max_new_tokens, logprobs=2, stop=self.stop)
#         outputs = self.model.generate(input_sequence, sampling_params, use_tqdm=False)

#         return outputs[0].outputs[0].text.strip()

#     def generate_analysis_question(self, solution_trace: Dict[int, Dict[str, str]]):
#         input_sequence = concat_solution_trace(solution_trace) + "<analysis>\n"
#         sampling_params = SamplingParams(n=1, temperature=0.1, top_p=0.95, max_tokens=self.max_new_tokens, logprobs=2, stop=self.stop)
#         outputs = self.model.generate(input_sequence, sampling_params, use_tqdm=False)

#         return outputs[0].outputs[0].text.strip()

#     def generate_ost_step(self, solution_trace: Dict[int, Dict[str, str]]):
#         input_sequence = concat_solution_trace(solution_trace) + "<next_step>\n"
#         sampling_params = SamplingParams(n=1, temperature=0.1, top_p=0.95, max_tokens=self.max_new_tokens, logprobs=2, stop=self.stop)
#         outputs = self.model.generate(input_sequence, sampling_params, use_tqdm=False)

#         return outputs[0].outputs[0].text.strip()

#     def generate_verify(self, solution_trace: Dict[int, Dict[str, str]]):
#         input_sequence = concat_solution_trace(solution_trace) + "<verify>\n"
#         sampling_params = SamplingParams(n=1, temperature=0.1, top_p=0.95, max_tokens=self.max_new_tokens, logprobs=2, stop=self.stop)
#         outputs = self.model.generate(input_sequence, sampling_params, use_tqdm=False)

#         return outputs[0].outputs[0].text.strip()

#     def generate_refine(self, solution_trace: Dict[int, Dict[str, str]]):
#         input_sequence = concat_solution_trace(solution_trace) + "<refine>\n"
#         sampling_params = SamplingParams(n=1, temperature=0.1, top_p=0.95, max_tokens=self.max_new_tokens, logprobs=2, stop=self.stop)
#         outputs = self.model.generate(input_sequence, sampling_params, use_tqdm=False)

#         return outputs[0].outputs[0].text.strip()

#     def generate_output(self, solution_trace: Dict[int, Dict[str, str]]):
#         input_sequence = concat_solution_trace(solution_trace) + "<output>\n"
#         sampling_params = SamplingParams(n=1, temperature=0.1, top_p=0.95, max_tokens=self.max_new_tokens, logprobs=2, stop=self.stop)
#         outputs = self.model.generate(input_sequence, sampling_params, use_tqdm=False)

#         return outputs[0].outputs[0].text.strip()


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
        generator = None,
        disable_a5: bool = None,
        user_question: str = None,
        max_depth_allowed: int = None,
        disable_a1: bool = None,
        # -----------------------------------
        # --- For instantiating REPHRASED_USER_QUESTION node ---
        rephrased_user_question: str = None,
        # ------------------------------------------------------
        expected_answer: str = None,
        # --- For instantiating DIRECT_ANSWER node ---
        direct_answer: str = None,
        # --------------------------------------------
        # --- For instantiating SUBQUESTION node ---
        subquestion: str = None,
        subanswer: str = None,
        is_new_subquestion: bool = None,
        # ------------------------------------------
        # --- For instantiating RE_SUBANSWER node ---
        re_subanswer: str = None,
        # -------------------------------------------
        # --- For instantiating OST_STEP node ---
        ost_step: str = None,
        analysis: str = None,
        verify: str = None,
        refine: str = None,
        output: str = None,
        # ---------------------------------------
        # --- For node selection (not in sanity checks yet) ---
        enable_potential_score: bool = None,
        potential_answers: List[str] = None,
        sentence_num: int = None,
        Q: int = 0,
        N: int = 0,
    ) -> None:
        """params:
        subquestion: the node is proposing a new subquestion
        subanswer: the answer corresponding to the new subquestion the node proposed
        re_subanswer: the node is proposing a new subanswer to the parent's subquestion
        """
        super().__init__()

        #! sanity checks
        try:
            assert depth is not None
            assert node_type is not None
            if node_value is not None:
                assert node_value >= 0, breakpoint()

            if node_type is Node_Type.USER_QUESTION:
                assert depth == 0
                # assert all(
                #     attr is None
                #     for attr in [
                #         parent,
                #         node_value,
                #         rephrased_user_question,
                #         direct_answer,
                #         subquestion,
                #         subanswer,
                #         is_new_subquestion,
                #         re_subanswer,
                #         ost_step,
                #     ]
                # )
                # assert all(
                #     attr is not None
                #     for attr in [generator, disable_a5, user_question, expected_answer, max_depth_allowed, disable_a1]
                # )
            elif node_type is Node_Type.CLARIFY:
                assert depth == 1
                assert all(
                    attr is None
                    for attr in [
                        node_value,
                        generator,
                        disable_a5,
                        user_question,
                        expected_answer,
                        direct_answer,
                        subquestion,
                        subanswer,
                        is_new_subquestion,
                        re_subanswer,
                        ost_step,
                        max_depth_allowed,
                        disable_a1,
                    ]
                )
                assert all(attr is not None for attr in [parent, rephrased_user_question])
            elif node_type is Node_Type.DIRECT_ANSWER:
                assert depth > 0
                assert all(
                    attr is None
                    for attr in [
                        generator,
                        disable_a5,
                        user_question,
                        expected_answer,
                        subquestion,
                        subanswer,
                        is_new_subquestion,
                        re_subanswer,
                        ost_step,
                        max_depth_allowed,
                        disable_a1,
                        node_value
                    ]
                )
                assert all(attr is not None for attr in [parent, direct_answer])
            elif node_type is Node_Type.SUBQUESTION:
                assert depth > 0
                assert all(
                    attr is None
                    for attr in [
                        generator,
                        disable_a5,
                        node_value,
                        user_question,
                        expected_answer,
                        direct_answer,
                        re_subanswer,
                        ost_step,
                        max_depth_allowed,
                        disable_a1,
                    ]
                )
                assert all(
                    attr is not None for attr in [parent, subquestion]
                )
            elif node_type is Node_Type.SUBANSWER:
                assert depth > 0
                assert all(
                    attr is None
                    for attr in [
                        generator,
                        disable_a5,
                        user_question,
                        node_value,
                        expected_answer,
                        direct_answer,
                        subquestion,
                        is_new_subquestion,
                        ost_step,
                        re_subanswer,
                        max_depth_allowed,
                        disable_a1,
                    ]
                )
                assert all(attr is not None for attr in [parent, subanswer])
            elif node_type is Node_Type.OST_STEP:
                assert depth > 0
                assert all(
                    attr is None
                    for attr in [
                        generator,
                        disable_a5,
                        user_question,
                        rephrased_user_question,
                        expected_answer,
                        direct_answer,
                        subquestion,
                        subanswer,
                        is_new_subquestion,
                        re_subanswer,
                        max_depth_allowed,
                        disable_a1,
                    ]
                )
                assert all(attr is not None for attr in [parent, ost_step])
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
        self.subanswer = subanswer
        self.is_new_subquestion = is_new_subquestion
        # self.re_subanswer = re_subanswer
        self.ost_step = ost_step
        self.process_reward = process_reward
        self.rephrased_user_question = rephrased_user_question
        self.analysis = analysis
        self.verify = verify
        self.refine = refine
        self.output = output
        self.sentence_num = sentence_num
        self.Q = Q
        self.N = N

        if parent is None:  # root
            self.verbose = verbose
            self.user_question = user_question
            self.expected_answer = expected_answer
            self.generator = generator
            self.disable_a5 = disable_a5
            self.max_depth_allowed = max_depth_allowed
            self.disable_a1 = disable_a1
            self.enable_potential_score = enable_potential_score
        else:  # inherit from parent
            self.verbose = parent.verbose
            self.user_question = parent.user_question
            self.expected_answer = parent.expected_answer
            self.generator = parent.generator
            self.disable_a5 = parent.disable_a5
            self.max_depth_allowed = parent.max_depth_allowed
            self.disable_a1 = parent.disable_a1
            self.enable_potential_score = parent.enable_potential_score

        #! keep track of paraphrasing
        if node_type is Node_Type.USER_QUESTION:
            self.paraphrased = False
        elif node_type is Node_Type.CLARIFY:
            self.paraphrased = True
            # self.user_question = rephrased_user_question
        else:
            assert parent is not None
            self.paraphrased = parent.paraphrased

        #! record solution trace from root to the current node. key: subquestion id
        if parent is None:  # root
            assert self.node_type is Node_Type.USER_QUESTION
            self.solution_trace: Dict[int, Dict[str, str]] = {0: {"user_question": user_question, "expected_answer": expected_answer, "node_id": self.id}}
        else:
            assert self.node_type is not Node_Type.USER_QUESTION
            self.solution_trace = deepcopy(parent.solution_trace)

            if node_type is Node_Type.CLARIFY:
                assert self.depth == parent.depth + 1
                self.solution_trace[self.depth] = {"clarify": rephrased_user_question, "process_reward": process_reward, "node_id": self.id}
            elif node_type is Node_Type.ANALYSIS:
                assert self.depth == parent.depth + 1
                self.solution_trace[self.depth] = {"analysis": analysis, "process_reward": process_reward, "node_id": self.id}
            elif node_type is Node_Type.DIRECT_ANSWER:
                assert self.depth == parent.depth + 1
                self.solution_trace[self.depth] = {
                    "direct_answer": direct_answer,
                    "process_reward": process_reward,
                    "node_id": self.id
                }
            elif node_type is Node_Type.SUBQUESTION:
                assert self.depth == parent.depth + 1
                self.solution_trace[self.depth] = {
                    "subquestion": subquestion,
                    "process_reward": process_reward,
                    "node_id": self.id
                }
            elif node_type is Node_Type.SUBANSWER:
                raise NotImplementedError
            elif node_type is Node_Type.OST_STEP:
                assert self.depth == parent.depth + 1
                self.solution_trace[self.depth] = {"ost_step": ost_step, "process_reward": process_reward, "node_id": self.id}
            elif node_type is Node_Type.VERIFY:
                assert self.depth == parent.depth + 1
                self.solution_trace[self.depth] = {"verify": verify, "process_reward": process_reward, "node_id": self.id}
            elif node_type is Node_Type.REFINE:
                assert self.depth == parent.depth + 1
                self.solution_trace[self.depth] = {"refine": refine, "process_reward": process_reward, "node_id": self.id}
            elif node_type is Node_Type.OUTPUT:
                assert self.depth == parent.depth + 1
                self.solution_trace[self.depth] = {"output": output, "process_reward": process_reward, "node_value": node_value, "node_id": self.id}


    def __str__(self) -> str:
        type2str = {
            Node_Type.USER_QUESTION: "U",
            Node_Type.CLARIFY: "CL",
            Node_Type.ANALYSIS: "AN",
            Node_Type.DIRECT_ANSWER: "DA",
            Node_Type.SUBQUESTION: "SQ",
            Node_Type.SUBANSWER: "SA",
            Node_Type.OST_STEP: "TS",
            Node_Type.VERIFY: "VERIFY",
            Node_Type.REFINE: "REFINE",
            Node_Type.OUTPUT: "OUT",
        }
        return f"{type2str[self.node_type]}-{self.id}"

    def _create_children(self):
        # breakpoint()
        # check if self is generated first
        self.generate_content()

        #! create children
        if self.node_type is Node_Type.USER_QUESTION:
            # analysis
            predict_input = concat_solution_trace(self.solution_trace) + "<"
            all_tokens = self.generator.predict_next_action(predict_input)
            intersect = set(all_tokens).intersection(set(["clarify", "subquestion", "analysis", "next_step"]))
            if not intersect:
                all_tokens = ["clarify", "analysis", "subquestion", "next_step"]
            for action in all_tokens:
                if action in "clarify":
                    self.children.append(
                        Reasoning_MCTS_Node(
                            parent=self,
                            depth=self.depth + 1,
                            node_type=Node_Type.CLARIFY,
                            rephrased_user_question="",
                            process_reward=None,
                        )
                    )
                elif action in "subquestion":
                    self.children.append(
                        Reasoning_MCTS_Node(
                            parent=self,
                            depth=self.depth + 1,
                            node_type=Node_Type.SUBQUESTION,
                            subquestion="",
                            process_reward=None,
                        )
                    )
                elif action in "analysis":
                    self.children.append(
                        Reasoning_MCTS_Node(
                            parent=self,
                            depth=self.depth + 1,
                            node_type=Node_Type.ANALYSIS,
                            analysis="",
                            process_reward=None,
                        )
                    )
                elif action in "next_step":
                    self.children.append(
                        Reasoning_MCTS_Node(
                            parent=self,
                            depth=self.depth + 1,
                            node_type=Node_Type.OST_STEP,
                            ost_step="",
                            process_reward=None,
                        )
                    )

        elif self.node_type is Node_Type.CLARIFY:
            predict_input = concat_solution_trace(self.solution_trace) + "<"
            all_tokens = self.generator.predict_next_action(predict_input)
            intersect = set(all_tokens).intersection(set(["analysis", "subquestion", "next_step", "direct_answer"]))
            if not intersect:
                all_tokens = ["analysis", "subquestion", "next_step", "direct_answer"]
            for action in all_tokens:
                if action in "direct_answer":
                    self.children.append(
                        Reasoning_MCTS_Node(
                            parent=self,
                            depth=self.depth + 1,
                            node_type=Node_Type.DIRECT_ANSWER,
                            direct_answer="",
                            process_reward=None,
                        )
                    )
                elif action in "subquestion":
                    self.children.append(
                        Reasoning_MCTS_Node(
                            parent=self,
                            depth=self.depth + 1,
                            node_type=Node_Type.SUBQUESTION,
                            subquestion="",
                            process_reward=None,
                        )
                    )
                elif action in "analysis":
                    self.children.append(
                        Reasoning_MCTS_Node(
                            parent=self,
                            depth=self.depth + 1,
                            node_type=Node_Type.ANALYSIS,
                            analysis="",
                            process_reward=None,
                        )
                    )
                elif action in "next_step":
                    self.children.append(
                        Reasoning_MCTS_Node(
                            parent=self,
                            depth=self.depth + 1,
                            node_type=Node_Type.OST_STEP,
                            ost_step="",
                            process_reward=None,
                        )
                    )
            
        elif self.node_type is Node_Type.ANALYSIS:
            predict_input = concat_solution_trace(self.solution_trace) + "<"
            all_tokens = self.generator.predict_next_action(predict_input)
            intersect = set(all_tokens).intersection(set(["subquestion", "next_step", "direct_answer"]))
            if not intersect:
                all_tokens = ["subquestion", "next_step", "direct_answer"]
            for action in all_tokens:
                if action in "direct_answer":
                    self.children.append(
                        Reasoning_MCTS_Node(
                            parent=self,
                            depth=self.depth + 1,
                            node_type=Node_Type.DIRECT_ANSWER,
                            direct_answer="",
                            process_reward=None,
                        )
                    )
                elif action in "subquestion":
                    self.children.append(
                        Reasoning_MCTS_Node(
                            parent=self,
                            depth=self.depth + 1,
                            node_type=Node_Type.SUBQUESTION,
                            subquestion="",
                            process_reward=None,
                        )
                    )
                elif action in "next_step":
                    self.children.append(
                        Reasoning_MCTS_Node(
                            parent=self,
                            depth=self.depth + 1,
                            node_type=Node_Type.OST_STEP,
                            ost_step="",
                            process_reward=None,
                        )
                    )

        elif self.node_type is Node_Type.DIRECT_ANSWER:
            predict_input = concat_solution_trace(self.solution_trace) + "<"
            all_tokens = self.generator.predict_next_action(predict_input)
            intersect = set(all_tokens).intersection(set(["output", "verify"]))
            if not intersect:
                all_tokens = ["output", "verify"]
            for action in all_tokens:
                if action in "output":
                    input_sequence = concat_solution_trace(self.solution_trace) + "<output>\n"
                    result = self.generator.generate_next_action(input_sequence)
                    value = 0
                    if is_equiv(remove_boxed(last_boxed_only_string(result)), remove_boxed(last_boxed_only_string(self.expected_answer))):
                        value = 1
                    self.children.append(
                        Reasoning_MCTS_Node(
                            parent=self,
                            depth=self.depth + 1,
                            node_type=Node_Type.OUTPUT,
                            output=result,
                            process_reward=None,
                            node_value=value
                        )
                    )
                elif action in "verify":
                    self.children.append(
                        Reasoning_MCTS_Node(
                            parent=self,
                            depth=self.depth + 1,
                            node_type=Node_Type.VERIFY,
                            verify="",
                            process_reward=None,
                        )
                    )
            
        elif self.node_type is Node_Type.SUBQUESTION:
            predict_input = concat_solution_trace(self.solution_trace) + "<"
            all_tokens = self.generator.predict_next_action(predict_input)
            intersect = set(all_tokens).intersection(set(["next_step", "direct_answer"]))
            if not intersect:
                all_tokens = ["next_step", "direct_answer"]
            for action in all_tokens:
                if action in "direct_answer":
                    self.children.append(
                        Reasoning_MCTS_Node(
                            parent=self,
                            depth=self.depth + 1,
                            node_type=Node_Type.DIRECT_ANSWER,
                            direct_answer="",
                            process_reward=None,
                        )
                    )
                elif action in "next_step":
                    self.children.append(
                        Reasoning_MCTS_Node(
                            parent=self,
                            depth=self.depth + 1,
                            node_type=Node_Type.OST_STEP,
                            ost_step="",
                            process_reward=None,
                        )
                    )

        elif self.node_type is Node_Type.OST_STEP:
            predict_input = concat_solution_trace(self.solution_trace) + "<"
            all_tokens = self.generator.predict_next_action(predict_input)
            intersect = set(all_tokens).intersection(set(["next_step", "direct_answer", "subquestion", "verify", "output"]))
            if not intersect:
                all_tokens = ["next_step", "direct_answer", "subquestion", "verify", "output"]
            for action in all_tokens:
                if action in "direct_answer":
                    self.children.append(
                        Reasoning_MCTS_Node(
                            parent=self,
                            depth=self.depth + 1,
                            node_type=Node_Type.DIRECT_ANSWER,
                            direct_answer="",
                            process_reward=None,
                        )
                    )
                elif action in "subquestion":
                    self.children.append(
                        Reasoning_MCTS_Node(
                            parent=self,
                            depth=self.depth + 1,
                            node_type=Node_Type.SUBQUESTION,
                            subquestion="",
                            process_reward=None,
                        )
                    )
                elif action in "next_step":
                    self.children.append(
                        Reasoning_MCTS_Node(
                            parent=self,
                            depth=self.depth + 1,
                            node_type=Node_Type.OST_STEP,
                            ost_step="",
                            process_reward=None,
                        )
                    )
                elif action in "verify":
                    self.children.append(
                        Reasoning_MCTS_Node(
                            parent=self,
                            depth=self.depth + 1,
                            node_type=Node_Type.VERIFY,
                            verify="",
                            process_reward=None,
                        )
                    )
                elif action in "output":
                    input_sequence = concat_solution_trace(self.solution_trace) + "<output>\n"
                    result = self.generator.generate_next_action(input_sequence)
                    value = 0
                    if is_equiv(remove_boxed(last_boxed_only_string(result)), remove_boxed(last_boxed_only_string(self.expected_answer))):
                        value = 1
                    self.children.append(
                        Reasoning_MCTS_Node(
                            parent=self,
                            depth=self.depth + 1,
                            node_type=Node_Type.OUTPUT,
                            output=result,
                            process_reward=None,
                            node_value=value,
                        )
                    )

        elif self.node_type is Node_Type.VERIFY:
            predict_input = concat_solution_trace(self.solution_trace) + "<"
            all_tokens = self.generator.predict_next_action(predict_input)
            intersect = set(all_tokens).intersection(set(["next_step", "refine", "direct_answer", "subquestion", "verify", "output"]))
            if not intersect:
                all_tokens = ["next_step", "refine", "direct_answer", "subquestion", "verify", "output"]
            for action in all_tokens:
                if action in "direct_answer":
                    self.children.append(
                        Reasoning_MCTS_Node(
                            parent=self,
                            depth=self.depth + 1,
                            node_type=Node_Type.DIRECT_ANSWER,
                            direct_answer="",
                            process_reward=None,
                        )
                    )
                elif action in "subquestion":
                    self.children.append(
                        Reasoning_MCTS_Node(
                            parent=self,
                            depth=self.depth + 1,
                            node_type=Node_Type.SUBQUESTION,
                            subquestion="",
                            process_reward=None,
                        )
                    )
                elif action in "next_step":
                    self.children.append(
                        Reasoning_MCTS_Node(
                            parent=self,
                            depth=self.depth + 1,
                            node_type=Node_Type.OST_STEP,
                            ost_step="",
                            process_reward=None,
                        )
                    )
                elif action in "verify":
                    self.children.append(
                        Reasoning_MCTS_Node(
                            parent=self,
                            depth=self.depth + 1,
                            node_type=Node_Type.VERIFY,
                            verify="",
                            process_reward=None,
                        )
                    )
                elif action in "output":
                    input_sequence = concat_solution_trace(self.solution_trace) + "<output>\n"
                    result = self.generator.generate_next_action(input_sequence)
                    value = 0
                    if is_equiv(remove_boxed(last_boxed_only_string(result)), remove_boxed(last_boxed_only_string(self.expected_answer))):
                        value = 1
                    self.children.append(
                        Reasoning_MCTS_Node(
                            parent=self,
                            depth=self.depth + 1,
                            node_type=Node_Type.OUTPUT,
                            output=result,
                            process_reward=None,
                            node_value=value,
                        )
                    )
                elif action in "refine":
                    self.children.append(
                        Reasoning_MCTS_Node(
                            parent=self,
                            depth=self.depth + 1,
                            node_type=Node_Type.REFINE,
                            refine="",
                            process_reward=None,
                        )
                    )

        elif self.node_type is Node_Type.REFINE:
            predict_input = concat_solution_trace(self.solution_trace) + "<"
            all_tokens = self.generator.predict_next_action(predict_input)
            intersect = set(all_tokens).intersection(set(["next_step", "direct_answer", "output", "subquestion", "verify"]))
            if not intersect:
                all_tokens = ["next_step", "direct_answer", "output", "subquestion", "verify"]
            for action in all_tokens:
                if action in "direct_answer":
                    self.children.append(
                        Reasoning_MCTS_Node(
                            parent=self,
                            depth=self.depth + 1,
                            node_type=Node_Type.DIRECT_ANSWER,
                            direct_answer="",
                            process_reward=None,
                        )
                    )
                elif action in "subquestion":
                    self.children.append(
                        Reasoning_MCTS_Node(
                            parent=self,
                            depth=self.depth + 1,
                            node_type=Node_Type.SUBQUESTION,
                            subquestion="",
                            process_reward=None,
                        )
                    )
                elif action in "next_step":
                    self.children.append(
                        Reasoning_MCTS_Node(
                            parent=self,
                            depth=self.depth + 1,
                            node_type=Node_Type.OST_STEP,
                            ost_step="",
                            process_reward=None,
                        )
                    )
                elif action in "verify":
                    self.children.append(
                        Reasoning_MCTS_Node(
                            parent=self,
                            depth=self.depth + 1,
                            node_type=Node_Type.VERIFY,
                            verify="",
                            process_reward=None,
                        )
                    )
                elif action in "output":
                    input_sequence = concat_solution_trace(self.solution_trace) + "<output>\n"
                    result = self.generator.generate_next_action(input_sequence)
                    value = 0
                    if is_equiv(remove_boxed(last_boxed_only_string(result)), remove_boxed(last_boxed_only_string(self.expected_answer))):
                        value = 1
                    self.children.append(
                        Reasoning_MCTS_Node(
                            parent=self,
                            depth=self.depth + 1,
                            node_type=Node_Type.OUTPUT,
                            output=result,
                            process_reward=None,
                            node_value=value,
                        )
                    )

        else:
            raise Exception("Invalid node type")

        assert self.children
        return self.children

    def is_valid_leaf_node(self):
        #! a valid solution can only be in SUBQUESTION type or DIRECT_ANSWER type
        return self.node_type is Node_Type.OUTPUT

    def is_valid_solution_node(self):
        #! a valid solution can only be in SUBQUESTION type or DIRECT_ANSWER type or OST_STEP type
        return self.node_type is Node_Type.OUTPUT

    def set_potential_score(self, score: float):
        self.potential_score = score

    def find_children(self, rollout_id: int):
        self.children = self.children or self._create_children()
        for child in self.children:
            child.set_rollout_id(rollout_id)
        assert self.children
        return self.children
    
    def generate_content(self):
        if self.node_type is Node_Type.CLARIFY and not self.rephrased_user_question:
            verbose_print(f"---- Generating clarify for node {self.id}...", self.verbose)
            input_sequence = concat_solution_trace(self.solution_trace) + "<clarify>\n"
            self.rephrased_user_question = self.generator.generate_next_action(input_sequence)
            self.solution_trace[self.depth]["clarify"] = self.rephrased_user_question
        elif self.node_type is Node_Type.ANALYSIS and not self.analysis:
            verbose_print(f"---- Generating analysis for node {self.id}...", self.verbose)
            input_sequence = concat_solution_trace(self.solution_trace) + "<analysis>\n"
            self.analysis = self.generator.generate_next_action(input_sequence)
            self.solution_trace[self.depth]["analysis"] = self.analysis
        elif self.node_type is Node_Type.DIRECT_ANSWER and not self.direct_answer:
            verbose_print(f"---- Generating sirect answer for node {self.id}...", self.verbose)
            input_sequence = concat_solution_trace(self.solution_trace) + "<direct_answer>\n"
            self.direct_answer = self.generator.generate_next_action(input_sequence, max_tokens=1024)
            self.solution_trace[self.depth]["direct_answer"] = self.direct_answer
        elif self.node_type is Node_Type.SUBQUESTION and not self.subquestion:
            verbose_print(f"---- Generating subquestion for node {self.id}...", self.verbose)
            input_sequence = concat_solution_trace(self.solution_trace) + "<subquestion>\n"
            self.subquestion = self.generator.generate_next_action(input_sequence)
            self.solution_trace[self.depth]["subquestion"] = self.subquestion
        elif self.node_type is Node_Type.OST_STEP and not self.ost_step:
            verbose_print(f"---- Generating one-step thought steps for node {self.id}...", self.verbose)
            input_sequence = concat_solution_trace(self.solution_trace) + "<next_step>\n"
            self.ost_step = self.generator.generate_next_action(input_sequence)
            self.solution_trace[self.depth]["ost_step"] = self.ost_step
        elif self.node_type is Node_Type.VERIFY and not self.verify:
            verbose_print(f"---- Generating verify for node {self.id}...", self.verbose)
            input_sequence = concat_solution_trace(self.solution_trace) + "<verify>\n"
            self.verify = self.generator.generate_next_action(input_sequence)
            self.solution_trace[self.depth]["verify"] = self.verify
        elif self.node_type is Node_Type.REFINE and not self.refine:
            verbose_print(f"---- Generating refine for node {self.id}...", self.verbose)
            input_sequence = concat_solution_trace(self.solution_trace) + "<refine>\n"
            self.refine = self.generator.generate_next_action(input_sequence)
            self.solution_trace[self.depth]["refine"] = self.refine
        elif self.node_type is Node_Type.OUTPUT and not self.output:
            raise Exception("Output node should have output")

    def is_terminal(self):
        if self.depth >= self.max_depth_allowed:
            self.generate_content()
            if self.node_value is None:
                self.node_value = 0
        return self.depth >= self.max_depth_allowed or self.is_valid_leaf_node()

    def calculate_reward(self):
        if self.is_valid_leaf_node():
            assert self.node_value is not None, breakpoint()
            return self.node_value
        else:
            return 0

    def skip_backprop(self):
        return self.node_type is Node_Type.USER_QUESTION or self.node_type is Node_Type.REPHRASED_USER_QUESTION

    def get_traces_and_rewards(self):
        trajectory, reward_list, content, repeat = [], [], [], 0
        cur_node = self
        while cur_node.parent:
            trajectory.append(cur_node.id)
            reward_list.append(cur_node.process_reward)
            if cur_node.node_type is Node_Type.CLARIFY:
                content.append(cur_node.rephrased_user_question)
            elif cur_node.node_type is Node_Type.ANALYSIS:
                content.append(cur_node.analysis)
            elif cur_node.node_type is Node_Type.OST_STEP:
                content.append(cur_node.ost_step)
            elif cur_node.node_type is Node_Type.SUBQUESTION:
                content.append(cur_node.subquestion)
            elif cur_node.node_type is Node_Type.DIRECT_ANSWER:
                content.append(cur_node.direct_answer)
            cur_node = cur_node.parent
        
        unique_elements = set(content)
        repeat = len(content) - len(unique_elements)
        
        return list(reversed(trajectory)), list(reversed(reward_list)), repeat
            
def search_for_answers(user_question: str, num_rollouts: int, question_id: str, gt_answer: str, generator, save_path, search_depth: int=16):
    verbose_print(
        f"********************* Searching for answers to question {question_id} ********************* ", True
    )

    question_pattern = r"<\|eot_id\|><\|start_header_id\|>user<\|end_header_id\|>\n\n(.*?)<\|eot_id\|>"
    match = re.search(question_pattern, user_question, re.DOTALL)

    if match:
        user_question = match.group(1)
    else:
        question_pattern = r"<\|im_end\|>\n<\|im_start\|>user\n(.*?)<\|im_end\|>"
        match2 = re.search(question_pattern, user_question, re.DOTALL)
        if match2:
            user_question = match2.group(1)

    #! build an MCTS searcher
    mcts_searcher = MCTS_Searcher(
        exploration_weight=2,
        weight_scheduler="const",
        num_rollouts=num_rollouts,
        discount=1.0,
        verbose=True,
    )

    #! build the MCTS tree
    root_node = Reasoning_MCTS_Node(
        parent=None,
        depth=0,
        node_type=Node_Type.USER_QUESTION,
        verbose=True,
        generator=generator,
        disable_a5=False,
        user_question=user_question,
        expected_answer=gt_answer,
        max_depth_allowed=search_depth,
        disable_a1=False,
        enable_potential_score=False,
        process_reward=0,
    )

    model_rollout_nodes = []
    all_step_rewards = []
    for i in (pbar := trange(num_rollouts, disable=True, position=0)):
        rollout_node = mcts_searcher.do_rollout(root_node, i)
        model_rollout_nodes.append(rollout_node)
        input_for_prm = concat_solution_trace_for_prm(rollout_node.solution_trace)
        step_reward = cal_reward(input_for_prm)
        step_reward[-1] = 0.6*rollout_node.node_value + 0.4*step_reward[-1]
        print(step_reward)
        all_step_rewards.append(step_reward)

    if save_path:
        js2 = [{"trace": node.solution_trace, "rollout_id": i} for i, node in enumerate(model_rollout_nodes)]
        with open(os.path.join(save_path, f"{question_id} - Rollout Solutions.json"), "w") as f:
            json.dump(js2, f)

    all_outputs = [concat_solution_trace(node.solution_trace, contain_question=False) for node in model_rollout_nodes]
    return all_outputs, all_step_rewards

# def configurate_method():

#     torch.manual_seed(42)
#     model_path = "/shared-nas/zhangkongcheng.zkc/models/checkpoints/sft_full_qwen_math4508k/checkpoint-24"

#     model = LLM(model=model_path, seed=42, dtype="half", max_model_len=4096)
#     tokenizer = AutoTokenizer.from_pretrained(model_path)

#     generator = Generator(model=model)

#     return generator, tokenizer

# if __name__ == "__main__":
#     generator, _ = configurate_method()
#     question = r"When the vectors $\\begin{pmatrix} -5 \\\\ 1 \\end{pmatrix}$ and $\\begin{pmatrix} 2 \\\\ 3 \\end{pmatrix}$ are both projected onto the same vector $\\mathbf{v},$ the result is $\\mathbf{p}$ in both cases.  Find $\\mathbf{p}.$"
#     print(question)
#     search_for_answers(question, 2, r"Hence, $\\mathbf{p} = \\boxed{\\begin{pmatrix} -34/53 \\\\ 119/53 \\end{pmatrix}}.$", generator)
