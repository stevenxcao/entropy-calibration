import torch
import torch.nn.functional as F
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, GPTNeoXForCausalLM, GemmaForCausalLM
from vllm import LLM, SamplingParams
from tqdm import tqdm

N_PARALLEL = torch.cuda.device_count()
COMPUTE_DTYPE = torch.bfloat16

class Tokenizer:
    def __init__(self, model_name):
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def encode_prompt(self, prompt, apply_chat_template=False):
        # encode prompt, which should not end in any eos token or end of message token because it will be used as prompt for generation
        if isinstance(prompt, str):
            ids = self.tokenizer(prompt, add_special_tokens=True)['input_ids']
            if ids[-1] == self.tokenizer.eos_token_id:
                ids = ids[:-1]
        else:
            if apply_chat_template:
                if prompt[-1]['role'] == 'user':
                    ids = self.tokenizer.apply_chat_template(prompt, add_generation_prompt=True)
                elif prompt[-1]['role'] == 'assistant':
                    ids = self.tokenizer.apply_chat_template(prompt[:-1], add_generation_prompt=True)
                    ids += self.tokenizer(prompt[-1]['content'], add_special_tokens=False)['input_ids']
            else:
                message_str = ""
                for message in prompt:
                    if len(message_str) > 0:
                        message_str += "\n"
                    if message['role'] == 'user':
                        message_str += "Q: "
                    elif message['role'] == 'assistant':
                        message_str += "A: "
                    else:
                        assert False, "message['role']={} not recognized".format(message['role'])
                    message_str += message['content']
                if prompt[-1]['role'] == 'user':
                    message_str += "\nA: "
                return self.encode_prompt(message_str, apply_chat_template=False)
        return ids

    def encode_response(self, response, add_special_tokens=False):
        # if we want to concatenate outputs of encode_prompt and encode_response, set add_special_tokens=False
        return self.tokenizer(response, add_special_tokens=add_special_tokens)['input_ids']

    def decode(self, token_ids, add_special_tokens=False):
        return self.tokenizer.decode(token_ids, add_special_tokens=add_special_tokens)

class vLLMModel:
    def __init__(self, model_name, gpu_util=0.8, revision='main'):
        self.model_name = model_name
        self.revision = revision
        self.llm = LLM(model=self.model_name, revision=revision, trust_remote_code=True, gpu_memory_utilization=gpu_util, tensor_parallel_size=N_PARALLEL)
    
    def generate(self, prompt_ids, temperature=1, max_tokens=200, n_repeats=1):
        # prompt_ids: List[List[int]]
        # returns generation_ids: List[List[int]]
        if n_repeats > 1:
            return self.generate_multiple(prompt_ids, temperature=temperature, max_tokens=max_tokens, n_repeats=n_repeats)
        sampling_params = SamplingParams(
            temperature=temperature,
            max_tokens=max_tokens,
            detokenize='False',
        )
        outputs = self.llm.generate(prompt_token_ids=prompt_ids, sampling_params=sampling_params)
        generation_ids = [list(output.outputs[0].token_ids) for output in outputs]
        return generation_ids

    def generate_multiple(self, prompt_ids, temperature=1, max_tokens=200, n_repeats=1):
        sampling_params = SamplingParams(
            n=n_repeats,
            temperature=temperature,
            max_tokens=max_tokens,
            detokenize='False',
        )
        outputs = self.llm.generate(prompt_token_ids=prompt_ids, sampling_params=sampling_params)
        generation_ids = [[list(o.token_ids) for o in output.outputs] for output in outputs]
        return generation_ids

class HFModel:
    def __init__(self, model_name, revision='main'):
        self.model_name = model_name
        self.revision = revision
        self.quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=COMPUTE_DTYPE,
            bnb_4bit_use_double_quant=True,
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, revision=revision, torch_dtype=COMPUTE_DTYPE, quantization_config=self.quantization_config, device_map='auto')

    def compute_logprobs_batched(self, prompt_ids, response_ids, bsz=8, temperature=1, compute_ent=True):
        logprobs = []
        for i in tqdm(range(0, len(prompt_ids), bsz)):
            ids1, ids2 = prompt_ids[i:i+bsz], response_ids[i:i+bsz]
            logprobs.extend(self.compute_logprobs(ids1, ids2, temperature=temperature, compute_ent=compute_ent))
        return logprobs
    
    def compute_logprobs(self, prompt_ids, response_ids, temperature=1., compute_ent=True):
        inp, pad_mask = ids_to_tensor([ids1 + ids2 for ids1, ids2 in zip(prompt_ids, response_ids)], dtype=torch.int32, device='cuda') # (b, l)
        with torch.no_grad():
            if isinstance(self.model, GPTNeoXForCausalLM) or isinstance(self.model, GemmaForCausalLM):
                logprobs = self.model(inp, attention_mask=pad_mask).logits.to(torch.float32) # (b, l, v)
            else:
                logprobs = self.model(inp, pad_mask=pad_mask).logits.to(torch.float32) # (b, l, v)
        logprobs = F.log_softmax(logprobs, dim=-1) # (b, l, v)
        logprobs = F.log_softmax(logprobs / temperature, dim=-1)
        # compute_ent = True: get ent for each response token (i.e. when predicting each response token, not after each response token)
        # compute_ent = False: get logprob for each response token
        if compute_ent:
            logprobs = -1 * torch.sum(logprobs * torch.exp(logprobs), dim=-1) # (b, l)
            logprobs = logprobs.cpu().numpy()
            logprobs = [lp[len(ids1)-1:len(ids1)-1+len(ids2)] for lp, ids1, ids2 in zip(logprobs, prompt_ids, response_ids)]
        else:
            b, l, v = logprobs.shape
            b_idx = torch.arange(b, device='cuda')
            l_idx = torch.arange(l, device='cuda')
            logprobs = logprobs[b_idx[:,None],l_idx[None,:-1],inp[:,1:]] # (b, l-1); first prediction lines up with 2nd token
            logprobs = logprobs.cpu().numpy()
            logprobs = [lp[len(ids1):len(ids1)+len(ids2)] for lp, ids1, ids2 in zip(logprobs, prompt_ids, response_ids)]
        return logprobs

def ids_to_tensor(all_ids, dtype=torch.int32, device='cuda'):
    max_len = np.max([len(ids) for ids in all_ids])
    inp = np.zeros((len(all_ids), max_len), dtype=np.int32)
    pad_mask = np.zeros((len(all_ids), max_len), dtype=np.int32)
    for i, ids in enumerate(all_ids):
        inp[i,:len(ids)] = ids
        pad_mask[i,:len(ids)] = 1
    inp = torch.from_numpy(inp).to(dtype=dtype, device=device)
    pad_mask = torch.from_numpy(pad_mask).to(dtype=dtype, device=device)
    return inp, pad_mask