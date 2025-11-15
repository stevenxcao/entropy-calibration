import os
from os import listdir
from os.path import join, isfile, exists
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

import sys
import pickle
from modeling import HFModel

model_name = sys.argv[1] #'Qwen/Qwen2.5-3B'
revision = sys.argv[2] #'main'
model_name_short = model_name[model_name.index('/')+1:]
save_folder = 'results'

bsz = 4

hf_model = HFModel(model_name, revision=revision)

def parse_filename(filename, suffix):
    # ex: Qwen2.5-3B_codecontests_revision=main_temp=1.0_all_ids.pkl
    filename = filename[:filename.index(suffix)] # Qwen2.5-3B_codecontests_revision=main_temp=1.0
    i = filename.index('_')
    model_name, filename = filename[:i], filename[i+1:] # codecontests_revision=main_temp=1.0
    i = filename.index('_')
    dataset_name, filename = filename[:i], filename[i+1:] # revision=main_temp=1.0
    i = filename.index('_')
    revision, temperature = filename[:i], filename[i+1:]
    temperature = temperature[temperature.index('=')+1:]
    revision = revision[revision.index('=')+1:]
    return model_name, dataset_name, revision, float(temperature) # Qwen2.5-3B, codecontests, main, 1.0

suffix = '_all_ids.pkl'
for filename in listdir(save_folder):
    full_path = join(save_folder, filename)
    if isfile(full_path) and full_path.endswith(suffix):
        load_model_name, load_dataset_name, load_revision, load_temperature = parse_filename(filename, suffix)
        if load_model_name == model_name_short and load_revision == revision:
            title = '{}_{}_revision={}_temp={}'.format(load_model_name, load_dataset_name, load_revision, load_temperature)
            save_path = save_folder + '/{}_logprobs_and_ents.pkl'.format(title)

            if exists(save_path):
                print('Ents and logprobs file {} already exists, skipping.'.format(save_path))
            else:
                print('Computing ents and logprobs for {}...'.format(full_path))
                with open(full_path, 'rb') as f:
                    prompt_ids, response_ids, generation_ids = pickle.load(f)
                
                response_logprobs = hf_model.compute_logprobs_batched(
                    prompt_ids, response_ids, temperature=load_temperature, bsz=bsz, compute_ent=False)
                generation_ents = hf_model.compute_logprobs_batched(
                    prompt_ids, generation_ids, temperature=load_temperature, bsz=bsz, compute_ent=True)

                #save_path = save_folder + '/{}_logprobs_and_ents.pkl'.format(title)
                print("Saving to {}...".format(save_path))
                with open(save_path, 'wb') as f:
                    pickle.dump([response_logprobs, generation_ents], f)