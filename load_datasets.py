from datasets import load_dataset, concatenate_datasets
from nltk.tokenize.treebank import TreebankWordDetokenizer
import numpy as np
from tqdm import tqdm

def get_prompt_and_response_ids(dataset, get_prompt_ids_f, get_response_ids_f, n_exmp, max_len, context_len, min_len, return_dataset=False):
    prompt_ids, response_ids = [], []
    if return_dataset:
        dataset_small = []
    for i, item in enumerate(tqdm(dataset)):
        ids1, ids2 = get_prompt_ids_f(item), get_response_ids_f(item)
        if ids1 is not None and len(ids1) + max_len < context_len and len(ids2) >= min_len:
            prompt_ids.append(ids1)
            response_ids.append(ids2[:max_len])
            if return_dataset:
                dataset_small.append(item)
            if len(prompt_ids) == n_exmp:
                break
        #if i % 100 == 0:
        #    print("i = {}, len(prompt_ids) = {}".format(i, len(prompt_ids)))
    if len(prompt_ids) < n_exmp:
        assert False, "Not enough examples: len(prompt_ids) = {}, n_exmp = {}".format(len(prompt_ids), n_exmp)
    if return_dataset:
        return prompt_ids, response_ids, dataset_small
    return prompt_ids, response_ids

def is_chat_model(model_name):
    return 'chat' in model_name.lower() or 'instruct' in model_name.lower()

### wikitext

def load_wikitext(min_len=200):
    dataset = load_dataset('iohadrubin/wikitext-103-raw-v1', trust_remote_code=True)
    return (dataset['train'].filter(lambda exmp: len(exmp['text']) >= min_len))['text']

def get_prompt_ids_wikitext(item, tokenizer, prompt_len=64):
    # item: str
    return tokenizer.encode_prompt(item)[:prompt_len]

def get_response_ids_wikitext(item, tokenizer, prompt_len=64):
    return tokenizer.encode_prompt(item)[prompt_len:]

def load_writingprompts():
    dataset = load_dataset('euclaise/writingprompts')
    return dataset['train']

### writingprompts

detokenizer = TreebankWordDetokenizer()

def detokenize(string):
    return detokenizer.detokenize(string.split())

def item_to_messages_writingprompts(item):
    def instruction_template(prompt):
        if "[ W P ]" in prompt:
            prompt = prompt[prompt.index("[ W P ]") + len("[ W P ]"):].lstrip().rstrip()
        if "[ WP ]" in prompt:
            prompt = prompt[prompt.index("[ WP ]") + len("[ WP ]"):].lstrip().rstrip()
        return "Write a story based on the following prompt: {}".format(prompt)
    
    messages = [
        {'role': 'user', 'content': instruction_template(detokenize(item['prompt']))},
        {'role': 'assistant', 'content': detokenize(item['story'])}
    ]
    return messages

def get_prompt_ids_writingprompts(item, tokenizer, prompt_len=64, apply_chat_template=False):
    messages = item_to_messages_writingprompts(item)
    prompt_ids = tokenizer.encode_prompt(messages[:-1], apply_chat_template=apply_chat_template)
    if prompt_len > 0:
        prompt_ids += tokenizer.encode_response(messages[-1]['content'])[:prompt_len]
    return prompt_ids

def get_response_ids_writingprompts(item, tokenizer, prompt_len=64):
    messages = item_to_messages_writingprompts(item)
    return tokenizer.encode_response(messages[-1]['content'])[prompt_len:]

### codecontests; adapted from https://github.com/ScalingIntelligence/large_language_monkeys/tree/main/llmonk

IMAGE_TAGS = ["<image>", "[Image]"]

CODELLAMA_PROMPT = "Write code to solve the following coding problem that obeys the constraints and passes the example test cases. The output code needs to read from and write to standard IO. Please wrap your code answer using ```:"

def load_codecontests():
    dataset = load_dataset("deepmind/code_contests")
    return concatenate_datasets([dataset['train'], dataset['valid'], dataset['test']])

def item_to_messages_codecontests(item, python_only=False):
    messages = [{'role': 'user', 'content': "{}\n{}".format(CODELLAMA_PROMPT, item['description'])}]
    if python_only:
        solutions = get_python_solutions(item)
    else:
        solutions = [solution for solution in item['solutions']['solution'] if solution.isascii()]
    if len(solutions) == 0 or has_image_tags(item['description']):
        return None
    else:
        solution_i = np.argmax([len(solution) for solution in solutions])
        messages.extend([{'role': 'assistant', 'content': "```{}```".format(solutions[solution_i].strip())}])
        return messages

def get_prompt_ids_codecontests(item, tokenizer, prompt_len=64, apply_chat_template=False, python_only=False):
    messages = item_to_messages_codecontests(item, python_only=python_only)
    if messages is None:
        return None
    prompt_ids = tokenizer.encode_prompt(messages[:-1], apply_chat_template=apply_chat_template)
    if prompt_len > 0:
        prompt_ids += tokenizer.encode_response(messages[-1]['content'])[:prompt_len]
    return prompt_ids

def get_response_ids_codecontests(item, tokenizer, prompt_len=64, python_only=False):
    messages = item_to_messages_codecontests(item, python_only=python_only)
    if messages is None:
        return None
    return tokenizer.encode_response(messages[-1]['content'])[prompt_len:]

def has_image_tags(description: str) -> bool:
    for tag in IMAGE_TAGS:
        if tag in description:
            return True
    return False

PYTHON3_LANGUAGE_ID = 3
def get_python_solutions(item: dict, filter_non_ascii=True, incorrect_solutions=False) -> list[str]:
    if incorrect_solutions:
        solution_key = "incorrect_solutions"
    else:
        solution_key = "solutions"

    python_solutions = []
    for i, (solution, lang_id) in enumerate(
        zip(
            item[solution_key]["solution"],
            item[solution_key]["language"],
        )
    ):
        if filter_non_ascii:
            if not solution.isascii():
                continue

        if lang_id == PYTHON3_LANGUAGE_ID:
            python_solutions.append(solution)

    return python_solutions