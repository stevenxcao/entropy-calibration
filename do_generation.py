import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ['VLLM_ATTENTION_BACKEND'] = 'XFORMERS'
os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'fork'
# Note: You may need to set VLLM_NCCL_SO_PATH manually
os.environ['VLLM_NCCL_SO_PATH'] = '~/.cupy/cuda_lib/12.x/nccl/2.16.2/lib/libnccl.so.2'

import sys
from tqdm import tqdm
import pickle
from modeling import vLLMModel, Tokenizer
from load_datasets import load_wikitext, get_prompt_ids_wikitext, get_response_ids_wikitext, \
    load_writingprompts, get_prompt_ids_writingprompts, get_response_ids_writingprompts, \
    load_codecontests, get_prompt_ids_codecontests, get_response_ids_codecontests, \
    get_prompt_and_response_ids, is_chat_model

dataset_names = ('codecontests', 'wikitext', 'writingprompts')
model_name = sys.argv[1] #'allenai/OLMo-2-1124-7B' #'Qwen/Qwen2.5-3B'
revision = sys.argv[2] #'stage1-step105000-tokens441B' #'main'
model_name_short = model_name[model_name.index('/')+1:]
save_folder = 'results'

if not os.path.exists(save_folder):
    os.makedirs(save_folder, exist_ok=True)

n_exmp = 5000
prompt_len = 128
bsz = 128
max_len = 2048
min_len = 1024
context_len = 4096

vllm_model = vLLMModel(model_name, revision=revision)
tokenizer = Tokenizer(model_name)

if is_chat_model(model_name):
    temperatures = (1.,)
    apply_chat_template = True
else:
    if revision == 'main':
        temperatures = (1., 0.95, 0.9, 0.85, 0.8)
    else:
        temperatures = (1.,)
    apply_chat_template = False

for dataset_name in dataset_names:
    if dataset_name == 'wikitext':
        dataset = load_wikitext(min_len=200)
        get_prompt_ids_f = lambda item: get_prompt_ids_wikitext(item, tokenizer, prompt_len=prompt_len)
        get_response_ids_f = lambda item: get_response_ids_wikitext(item, tokenizer, prompt_len=prompt_len)
    elif dataset_name == 'writingprompts':
        dataset = load_writingprompts()
        get_prompt_ids_f = lambda item: get_prompt_ids_writingprompts(
            item, tokenizer, prompt_len=prompt_len, apply_chat_template=apply_chat_template)
        get_response_ids_f = lambda item: get_response_ids_writingprompts(item, tokenizer, prompt_len=prompt_len)
    elif dataset_name == 'codecontests': 
        dataset = load_codecontests()
        get_prompt_ids_f = lambda item: get_prompt_ids_codecontests(
            item, tokenizer, prompt_len=prompt_len, apply_chat_template=apply_chat_template)
        get_response_ids_f = lambda item: get_response_ids_codecontests(item, tokenizer, prompt_len=prompt_len)
    else:
        assert False

    prompt_ids, response_ids = get_prompt_and_response_ids(
        dataset, get_prompt_ids_f, get_response_ids_f, n_exmp, max_len, context_len, min_len)

    for temperature in temperatures:
        title = '{}_{}_revision={}_temp={}'.format(model_name_short, dataset_name, revision, temperature)
        generation_ids = []
        for i in tqdm(range(0, len(prompt_ids), bsz)):
            generation_ids.extend(vllm_model.generate(prompt_ids[i:i+bsz], temperature=temperature, max_tokens=max_len))

        print("Prompt: {}".format(tokenizer.decode(prompt_ids[0], add_special_tokens=True)))
        print("Response: {}".format(tokenizer.decode(response_ids[0], add_special_tokens=True)))
        print("Generation: {}".format(tokenizer.decode(generation_ids[0], add_special_tokens=True)))

        save_path = save_folder + '/{}_all_ids.pkl'.format(title)
        print("Saving to {}...".format(save_path))
        with open(save_path, 'wb') as f:
            pickle.dump([prompt_ids, response_ids, generation_ids], f)