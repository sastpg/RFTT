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



def extract_trajectory(file_path):
    correct_answers, wrong_answers = [], []
    try:
        # 打开JSON文件并读取数据
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
        key=lambda x: (x["repeat_num"], -len(x["trajectory"]), len(x["trace"][f"{len(x['trajectory'])}"]['direct_answer'] if x["trace"][f"{len(x['trajectory'])}"].get('direct_answer')!=None else x["trace"][f"{len(x['trajectory'])}"]['next_step']), -sum(x["reward_list"])/len(x["trajectory"]))  # repeat_num 升序， trajectory 长度和 reward_list 平均值降序
    )

    wrong_answer_sorted = sorted(
        wrong_answers,
        key=lambda x: (x["repeat_num"], -len(x["trajectory"]), len(x["trace"][f"{len(x['trajectory'])}"]['direct_answer'] if x["trace"][f"{len(x['trajectory'])}"].get('direct_answer')!=None else x["trace"][f"{len(x['trajectory'])}"]['next_step']), sum(x["reward_list"])/len(x["trajectory"]))  # repeat_num 升序， trajectory 长度降序，reward_list 平均值升序
    )

    correct_data1, wrong_data, _ = get_correct_wrong_intersection(correct_answer_sorted, wrong_answer_sorted)

    if not wrong_data:
        correct_data2, wrong_data = get_correct_wrong_independent(correct_answer_sorted, wrong_answer_sorted)
    else:
        correct_data2, _ = get_correct_wrong_independent(correct_answer_sorted, [wrong_data])

    if not wrong_data:
        wrong_data = wrong_answer_sorted[0] if wrong_answer_sorted else {}

    correct_data3, correct_data4 = get_correct_independent(correct_answer_sorted)
    if not correct_data3:
        correct_data3 = correct_answer_sorted[0] if correct_answer_sorted else {}

    info = [correct_data1, wrong_data, correct_data2, correct_data3, correct_data4]
        

# 示例调用
folder_path = 'Question 0001 - Final Solutions.json'  # 替换为你的文件夹路径
extract_trajectory(folder_path)