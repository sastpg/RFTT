from torch.utils.data import Dataset
from tqdm import tqdm


# def preprocess_data(data, input_template=None, input_key="input", apply_chat_template=None) -> str:
#     if apply_chat_template:
#         chat = data[input_key]
#         if isinstance(chat, str):
#             chat = [{"role": "user", "content": chat}]
#         prompt = apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
#     else:
#         prompt = data[input_key]
#         if input_template:
#             prompt = input_template.format(prompt)
#     return prompt

def preprocess_data(data, input_template=None, input_key="input", apply_chat_template=None) -> str:
    if apply_chat_template:
        chat = data[input_key]
        if isinstance(chat, str):
            chat = [{"role": "user", "content": chat}]
        prompt = apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
    else:
        prompt = data[input_key]
        if input_template:
            prompt = input_template.format(prompt)
    solution = data["solution"]
    unique_id = data["unique_id"]
    return prompt, solution, unique_id


class PromptDataset(Dataset):
    """
    Dataset for PPO model

    Args:
        dataset: dataset for PPO model
        tokenizer: tokenizer for PPO model
        max_length: max length of input
    """

    def __init__(
        self,
        dataset,
        tokenizer,
        strategy,
        input_template=None,
    ) -> None:
        super().__init__()
        self.strategy = strategy
        self.tokenizer = tokenizer

        # chat_template
        self.input_template = input_template
        input_key = getattr(self.strategy.args, "input_key", None)
        apply_chat_template = getattr(self.strategy.args, "apply_chat_template", False)

        if apply_chat_template:
            apply_chat_template = self.tokenizer.apply_chat_template

        self.prompts = []
        self.solutions = []
        self.ids = []
        for data in tqdm(dataset, desc="Preprocessing data", disable=not self.strategy.is_rank_0()):
            prompt, solution, id = preprocess_data(data, input_template, input_key, apply_chat_template)
            self.prompts.append(prompt)
            self.solutions.append(solution)
            self.ids.append(id)

    def __len__(self):
        length = len(self.prompts)
        return length

    def __getitem__(self, idx):
        return self.prompts[idx], self.solutions[idx], self.ids[idx]
