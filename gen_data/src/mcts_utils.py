# Licensed under the MIT license.

from enum import Enum, unique
import re
import copy
import math
from typing import Dict
from itertools import islice
from colorama import Fore, Style
import math
import requests
from nltk.tokenize import sent_tokenize

@unique
class Node_Type(Enum):
    USER_QUESTION = "USER_QUESTION"
    CLARIFY = "CLARIFY"
    ANALYSIS = "ANALYSIS"
    NEXT_STEP = "NEXT_STEP"
    DIRECT_ANSWER = "DIRECT_ANSWER"
    SUBQUESTION = "SUBQUESTION"
    SUBANSWER = "SUBANSWER"
    
def reach_terminal_step(step: str):
    assert step is not None

    if "Final step" in step or "So the answer is" in step:
        return True
    return False


def print_tree_from_root(mcts_searcher, rollout_id, root_node, chosen_node=None, file=None):
    color_print = False if file else True

    def my_print(text):
        if file:
            file.write(text + "\n")
        else:
            print(text)

    def print_tree(parent_node, node, file, rollout_id):
        to_print = ""

        num_indent = 4
        dash = "-" * num_indent * node.depth
        space = " " * num_indent * node.depth

        attributes = f"Q: {round(mcts_searcher.Q[node], 2)}" + "; " + f"N: {mcts_searcher.N[node]}" + "; "
        attributes += f"V: {round(node.node_value, 2)}" if node.node_value is not None else "V: None"
        # attributes += f"; PR: {round(node.process_reward, 4)}"

        uct_value = "UCT: " + str(
            round(mcts_searcher._compute_uct(parent_node=parent_node, node=node, rollout_id=rollout_id), 2)
        )
        attributes += "; " + uct_value

        solution_marker = "(T) " if node.is_valid_solution_node() else ""

        node_info = "[" + solution_marker + node.__str__() + ": " + attributes + "]"
        if chosen_node and node == chosen_node:
            node_info = "[" + node_info + "]"
        node_info += " "

        if color_print and node.is_valid_solution_node():
            node_details = Fore.RED + Style.BRIGHT + node_info + Fore.RESET + Style.RESET_ALL
        else:
            node_details = node_info

        if node.node_type is Node_Type.USER_QUESTION:
            gt = node.expected_answer.replace("\n", " ")
            node_details += f"User: {node.user_question}" + "\n" + space + " " * len(node_info) + f"Ground truth: {gt}"
        elif node.node_type is Node_Type.CLARIFY:
            node_details += f"Clarify: {node.clarify}"
        elif node.node_type is Node_Type.ANALYSIS:
            node_details += f"Analy: {node.analysis}"
        elif node.node_type is Node_Type.DIRECT_ANSWER:
            node_details += f"Ans: {node.direct_answer}"
        elif node.node_type is Node_Type.SUBQUESTION:
            node_details += f"Sub-Q: {node.subquestion}"
        elif node.node_type is Node_Type.SUBANSWER:
            node_details += f"Sub-A: {node.subanswer}"
        elif node.node_type is Node_Type.NEXT_STEP:
            node_details += f"NST: {node.next_step}"

        to_print += dash + node_details

        my_print(to_print)

        for child in node.children:
            print_tree(node, child, file, rollout_id)

        if node.depth == 0:
            my_print("\n" + "=" * 50 + "\n")

    print_tree(parent_node=None, node=root_node, file=file, rollout_id=rollout_id)

def get_sentences(input):
    sentences = sent_tokenize(input)
    return sentences

def cal_reward(input):
    url = ""
    data = {"input": input}
    resp = requests.post(url, json=data)
    return resp.json()

def concat_solution_trace(solution_trace: Dict[int, Dict[str, str]], parent_is_subquestion: bool=False) -> str:
    """Return: concatenated subqs and subas and ost steps, next subquestion id"""
    existing_steps = []
    existing_trace = "None"
    solution_trace1 = copy.deepcopy(solution_trace)
    solution_trace1.pop(0)
    for solution in solution_trace1.items():
        step = list(solution[1].items())[0][1]
        existing_steps.append(step)
    
    if existing_steps:
        if parent_is_subquestion:
            subquestion = existing_steps[-1]
            pattern = r"Let's (.*?) now"
            match = re.search(pattern, subquestion)
            if match:
                subquestion = match.group(1)
            existing_trace = "\n".join(existing_steps[:-1]) + "\nNext step:" + subquestion
        else:
            existing_trace = "\n".join(existing_steps)
    
    return existing_trace

def concat_solution_trace_for_prm(solution_trace: Dict[int, Dict[str, str]]) -> str:
    traces = ""
    for key, value in islice(solution_trace.items(), 0, None):
        if "user_question" in value:
            traces += f"{value['user_question']} "
        elif "clarify" in value:
            traces += f"Step {key}: {value['clarify']} ки\n"
        elif "analysis" in value:
            traces += f"Step {key}: {value['analysis']} ки\n"
        elif "subquestion" in value:
            traces += f"Step {key}: {value['subquestion']} ки\n"
        elif "next_step" in value:
            traces += f"Step {key}: {value['next_step']} ки\n"
        elif "direct_answer" in value:
            traces += f"Step {key}: {value['direct_answer']} ки\n"
        elif "verify" in value:
            traces += f"Step {key}: {value['verify']} ки\n"
        elif "refine" in value:
            traces += f"Step {key}: {value['refine']} ки\n"
        elif "output" in value:
            traces += f"Step {key}: {value['output']} ки\n"
        else:
            breakpoint()
            raise NotImplementedError
    
    return traces.strip()

def find_valid_solution_nodes(root_node):
    valid_solution_nodes = []

    def recursion(node):
        if node.is_valid_solution_node():
            valid_solution_nodes.append(node)
            return

        if not node.children:  #! no children
            return

        for child in node.children:
            recursion(child)

    recursion(root_node)

    return valid_solution_nodes

def stochastic_find_best_solution(
    root_node,
    evaluator,
    enable_potential_score,
):
    # todo: what strategy do we use to select best node?
    """The function finds the best solution from the solution nodes in the MCTS tree.
    Return: top answer, top solution, confidence of the top answer, the corresponding node of the answer, all solution nodes
    """
    solution_nodes = find_valid_solution_nodes(root_node)

    if len(solution_nodes) == 0:
        return None, None

    def extract_solution_from_node(node):
        if node.node_type is Node_Type.SUBQUESTION:
            return node.subanswer
        elif node.node_type is Node_Type.DIRECT_ANSWER:
            return node.direct_answer
        else:
            return None

    solutions = [extract_solution_from_node(node) for node in solution_nodes]

    def calculate_potential_score_for_solution_node(node):
        model_answer = evaluator.extract_answer_from_model_completion(extract_solution_from_node(node))
        potential_answers_history = node.potential_answers_history  # {depth -> [potential answers]}
        assert potential_answers_history[node.depth] is None

        potential_score = 1
        for depth, depth_potential_answers in potential_answers_history.items():
            if depth < node.depth:
                depth_score = sum(
                    evaluator.check_answers_equiv(dpa, model_answer) for dpa in depth_potential_answers
                ) / len(depth_potential_answers)
                potential_score *= depth_score

        node.set_potential_score(potential_score)
        return potential_score

    prior_weights = (
        [calculate_potential_score_for_solution_node(node) for node in solution_nodes]
        if enable_potential_score
        else None
    )
    return solution_nodes, solutions
    # breakpoint()
    # top_answer, top_completion, top_completion_id, top_confidence = evaluator.stochastic_find_most_confident_answer(
    #     completions=solutions, prior_weights=prior_weights
    # )
    # return top_answer, top_completion, top_confidence, solution_nodes[top_completion_id], solution_nodes, solutions
