import json
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor, as_completed

def read_jsonl(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            # Strip out whitespace and check if the line is not empty
            if line.strip():
                data.append(json.loads(line))
    return data

def get_response(args):
    # Set OpenAI's API key and API base to use vLLM's API server.
    openai_api_key = "0"
    openai_api_base = "http://localhost:8090/v1"
    question, solution, level = args
    client = OpenAI(
        api_key=openai_api_key,
        base_url=openai_api_base,
    )
    chat_response = client.chat.completions.create(
        model="/path/to/model",
        messages=[
            {"role": "user", "content": question},
        ],
        temperature=0.01,
        timeout=150,
        max_tokens=4096,
        # frequency_penalty=1.05
    )
    info = json.dumps({"question": question, "solution": solution, "level":level, "model_answer": chat_response.choices[0].message.content}, ensure_ascii=False)
    return info
def main():
    # num of concurrent requests
    CONCURRENT_REQUESTS = 10
    file_path = 'data/math_test.jsonl'
    test_data = read_jsonl(file_path)

    with ThreadPoolExecutor(max_workers=CONCURRENT_REQUESTS) as executor:
        futures = [executor.submit(get_response, (test['problem'], test['solution'], test['level'])) for test in test_data]
        for future in as_completed(futures):
            try:
                result = future.result()
                with open("results.jsonl", 'a+', encoding="utf-8") as f:
                    f.write(result + '\n')
            except Exception as e:
                print(f"Request generated an exception: {e}")

if __name__ == "__main__":
    main()