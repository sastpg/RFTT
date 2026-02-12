import json
from itertools import islice

def calculate_distance(list1, list2):
    list1 = [0] + list1
    list2 = [0] + list2
    i, j = len(list1) - 1, len(list2) - 1
    while i >= 0 and j >= 0:
        if list1[i] == list2[j]:
            distance = len(list1) - 1 -i + len(list2) - 1 - j
            return distance, i
        elif list1[i] > list2[j]:
            i -= 1
        else:
            j -= 1

def get_correct_wrong_intersection(correct_answer, wrong_answer):
    for correct in correct_answer:
        for wrong in wrong_answer:
            correct_final = correct["trace"][f"{len(correct['trajectory'])}"]['direct_answer'] if correct["trace"][f"{len(correct['trajectory'])}"].get('direct_answer')!=None else correct["trace"][f"{len(correct['trajectory'])}"]['next_step']
            wrong_final = wrong["trace"][f"{len(wrong['trajectory'])}"]['direct_answer'] if wrong["trace"][f"{len(wrong['trajectory'])}"].get('direct_answer')!=None else wrong["trace"][f"{len(wrong['trajectory'])}"]['next_step']
            distance, i = calculate_distance(correct["trajectory"], wrong["trajectory"])
            if distance < 4 and "\\boxed" in correct_final and "\\boxed" in wrong_final:
                return correct, wrong, i
    return {}, {}, 999

def get_correct_wrong_independent(correct_answer, wrong_answer):
    for correct in correct_answer:
        for wrong in wrong_answer:
            trajectory1, trajectory2 = correct["trajectory"], wrong["trajectory"]
            if len(trajectory1)+ len(trajectory2) == len(set(trajectory1+trajectory2)):
                return correct, wrong
    return {}, {}

def get_correct_independent(correct_answer):
    i = 0
    for answer1 in correct_answer[i:]:
        for answer2 in correct_answer[i:]:
            trajectory1, trajectory2 = answer1["trajectory"], answer2["trajectory"]
            if len(trajectory1)+ len(trajectory2) == len(set(trajectory1+trajectory2)):
                return answer1, answer2
    return {}, {}

def concat_trace(trace):
    reasoning_chain = ""
    for key, value in islice(trace.items(), 1, None):
        if "clarify" in value:
            reasoning_chain += f"<clarify>\n{value['clarify'].strip()}\n</clarify>\n"
        elif "analysis" in value:
            reasoning_chain += f"<analysis>\n{value['analysis'].strip()}\n</analysis>\n"
        elif "subquestion" in value:
            reasoning_chain += f"<subquestion>\n{value['subquestion'].strip()}\n</subquestion>\n"
        elif "next_step" in value:
            reasoning_chain += f"<next_step>\n{value['next_step'].strip()}\n</next_step>\n"
        elif "direct_answer" in value:
            reasoning_chain += f"<direct_answer>\n{value['direct_answer'].strip()}\n</direct_answer>\n"
        elif "verify" in value:
            reasoning_chain += f"<verify>\n{value['verify'].strip()}\n</verify>\n"
        elif "refine" in value:
            reasoning_chain += f"<refine>\n{value['refine'].strip()}\n</refine>\n"
        elif "output" in value:
            reasoning_chain += f"<output>\n{value['output'].strip()}\n</output>\n"
        else:
            raise Exception("Undefined!")
    return reasoning_chain.strip()

def extract_trajectory(file_path):
    correct_answers, wrong_answers = [], []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        for item in data:
            trace = item["trace"]
            num_step = max(trace.keys(), key=int)
            is_correct = trace[num_step]['node_value']
            if is_correct:
                correct_answers.append(item)
            else:
                wrong_answers.append(item)
    except Exception as e:
        print(f"Error {file_path}: {e}")

    correct_answer_sorted = sorted(
        correct_answers,
        key=lambda x: (x["repeat_num"], -len(x["trajectory"]), len(x["trace"][f"{len(x['trajectory'])}"]['direct_answer'] if x["trace"][f"{len(x['trajectory'])}"].get('direct_answer')!=None else x["trace"][f"{len(x['trajectory'])}"]['next_step']), -sum(x["reward_list"])/len(x["trajectory"]))  #  trajectory 长度和 reward_list 平均值降序
    )

    wrong_answer_sorted = sorted(
        wrong_answers,
        key=lambda x: (x["repeat_num"], -len(x["trajectory"]), len(x["trace"][f"{len(x['trajectory'])}"]['direct_answer'] if x["trace"][f"{len(x['trajectory'])}"].get('direct_answer')!=None else x["trace"][f"{len(x['trajectory'])}"]['next_step']), sum(x["reward_list"])/len(x["trajectory"]))  # trajectory 长度降序，reward_list 平均值升序
    )

    # NOTE: tua_c and tau_w are correct and wrong trajectories that share the same prefix in the tree.
    tau_c, tau_w, _ = get_correct_wrong_intersection(correct_answer_sorted, wrong_answer_sorted)

    # if not wrong_data:
    #     correct_data2, wrong_data = get_correct_wrong_independent(correct_answer_sorted, wrong_answer_sorted)
    # else:
    #     correct_data2, _ = get_correct_wrong_independent(correct_answer_sorted, [wrong_data])

    # if not wrong_data:
    #     wrong_data = wrong_answer_sorted[0] if wrong_answer_sorted else {}

    # NOTE: Find other trajectories (withou intersection) with correct answers
    # TODO: Implement other algorithms to select trajectories
    tau_c1, tau_c2 = get_correct_independent(correct_answer_sorted)
    if not tau_c1:
        tau_c1 = correct_answer_sorted[0] if correct_answer_sorted else {}

    info = [tau_c, tau_w, tau_c1, tau_c2]
    for i in range(len(info)):
        if info[i]:
            info[i]["text"] = concat_trace(info[i]["trace"])

    with open(f"{file_path[:-5]}" + " Extract.json", 'w', encoding='utf-8') as output_f:
        json.dump(info, output_f, ensure_ascii=False, indent=4)


# Example
folder_path = 'run_outputs/Example - Final Solutions.json'
extract_trajectory(folder_path)