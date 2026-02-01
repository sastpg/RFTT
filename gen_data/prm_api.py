from transformers import AutoTokenizer
from transformers import AutoModelForCausalLM
import torch
import uvicorn
from pydantic import BaseModel
from fastapi import FastAPI
app = FastAPI()

class Input(BaseModel):
    input: str

good_token = '+'
bad_token = '-'
step_tag = 'ки'
device = torch.device("cuda:7" if torch.cuda.is_available() else "cpu")

model_path = "/path/to/math-shepherd-mistral-7b-prm"

tokenizer = AutoTokenizer.from_pretrained(model_path)
candidate_tokens = tokenizer.encode(f"{good_token} {bad_token}")[1:]  # [648, 387]
step_tag_id = tokenizer.encode(f"{step_tag}")[-1]  # 12902


model = AutoModelForCausalLM.from_pretrained(model_path).eval()
model.to(device)

@app.post("/get_reward")
def get_reward(request: Input):
    torch.cuda.empty_cache()
    input_for_prm = request.input
    input_id = torch.tensor([tokenizer.encode(input_for_prm)]).to(device)

    with torch.no_grad():
        logits = model(input_id).logits[:, :, candidate_tokens].to(device)
        scores = logits.softmax(dim=-1)[:, :, 0].to(device)
        step_scores = scores[input_id == step_tag_id].to(device)
        return {"score": step_scores.tolist()}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8008)