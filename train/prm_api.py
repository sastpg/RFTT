from transformers import AutoTokenizer, LlamaTokenizer
from transformers import AutoModelForCausalLM
import torch
import uvicorn
from pydantic import BaseModel
from fastapi import FastAPI
import re
app = FastAPI()

class Input(BaseModel):
    input: str

good_token = '+'
bad_token = '-'
step_tag = 'ки'
device = torch.device("cuda:7" if torch.cuda.is_available() else "cpu")

model_path = "/path/to/math-shepherd-mistral-7b-prm"

tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
candidate_tokens = tokenizer.encode(f"{good_token} {bad_token}")[1:]  # [648, 387]
step_tag_id = tokenizer.encode(f"{step_tag}")[-1]  # 12902


model = AutoModelForCausalLM.from_pretrained(model_path).eval()
model.to(device)

def make_input_for_prm(input):
    question_pattern = r"<\|begin_of_text\|><\|start_header_id\|>system<\|end_header_id\|>\n\nYou are a helpful assistant.<\|eot_id\|><\|start_header_id\|>user<\|end_header_id\|>\n\n(.*?)<\|eot_id\|><\|start_header_id\|>assistant<\|end_header_id\|>\n\n(.*?)<\|eot_id\|>"
    match = re.search(question_pattern, input, re.DOTALL)
    print("Input:")
    print(input, match)
    question, raw_solution = match.group(1), match.group(2)
    question = question.strip()
    solutions = ""
    pattern = r'<(analysis|subquestion|next_step|clarify|direct_answer|output|verify|refine)>(.*?)</\1>'
    
    # 使用findall方法查找所有匹配项，设置re.DOTALL使得.可以匹配换行符
    matches = re.findall(pattern, raw_solution, re.DOTALL)
    if not matches:
        solutions = raw_solution.strip()
    # 创建一个字典来保存不同标签的内容
    i = 1
    for tag, content in matches:
        # 去除内容中的多余空白并添加到字典中
        clean_content = content.strip()
        solutions += f"Step {i}: {clean_content} ки\n"
        i += 1
    solutions = solutions.strip()

    return f"{question} {solutions}"


@app.post("/get_reward")
def get_reward(request: Input):
    ori_input = request.input
    # input_for_prm = make_input_for_prm(ori_input)
    input_for_prm = ori_input
    print(input_for_prm)
    input_id = torch.tensor([tokenizer.encode(input_for_prm)]).to(device)
    
    if len(input_id[0]) > 5120:
        del input_id
        torch.cuda.empty_cache()
        return {"score": [0.2] * input_for_prm.count("ки")}
    
    
    with torch.no_grad():
        logits = model(input_id).logits[:, :, candidate_tokens].to(device)
        scores = logits.softmax(dim=-1)[:, :, 0].to(device)
        step_scores = scores[input_id == step_tag_id]
        del input_id, logits, scores
        torch.cuda.empty_cache()
        return {"score": step_scores.tolist()}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=5000)