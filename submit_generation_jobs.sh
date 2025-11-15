#!/bin/bash

ACCOUNT="$1"
PARTITION="$2"

function submit_job() {
    sbatch --account=$ACCOUNT --cpus-per-task=4 --gres=gpu:${4:-1} --job-name=generation-job --mem=32G --open-mode=append --output=${5} --partition=$PARTITION --time=14-0 --wrap "stdbuf -oL -eL python -u ${1} ${2} ${3}"
}

model_names=(
    'meta-llama/Llama-3.2-1B'
    'meta-llama/Llama-3.2-1B-Instruct'
    'meta-llama/Llama-3.2-3B'
    'meta-llama/Llama-3.2-3B-Instruct'
    'meta-llama/Llama-3.1-8B'
    'meta-llama/Llama-3.1-8B-Instruct'
    'meta-llama/Llama-3.1-70B'
    'meta-llama/Llama-3.1-70B-Instruct'
    'Qwen/Qwen2.5-0.5B'
    'Qwen/Qwen2.5-0.5B-Instruct'
    'Qwen/Qwen2.5-1.5B'
    'Qwen/Qwen2.5-1.5B-Instruct'
    'Qwen/Qwen2.5-3B'
    'Qwen/Qwen2.5-3B-Instruct'
    'Qwen/Qwen2.5-7B'
    'Qwen/Qwen2.5-7B-Instruct'
    'Qwen/Qwen2.5-14B'
    'Qwen/Qwen2.5-14B-Instruct'
    'Qwen/Qwen2.5-32B'
    'Qwen/Qwen2.5-32B-Instruct'
    'Qwen/Qwen2.5-72B'
    'Qwen/Qwen2.5-72B-Instruct'
    'meta-llama/Llama-2-7b-hf'
    'meta-llama/Llama-2-7b-chat-hf'
    'meta-llama/Llama-2-13b-hf'
    'meta-llama/Llama-2-13b-chat-hf'
    'meta-llama/Llama-2-70b-hf'
    'meta-llama/Llama-2-70b-chat-hf'
    'EleutherAI/pythia-410m'
    'EleutherAI/pythia-1.4b'
    'EleutherAI/pythia-2.8b'
    'EleutherAI/pythia-6.9b'
    'EleutherAI/pythia-12b'
)

gpu_counts=(1 1 1 1 1 1 4 4 1 1 1 1 1 1 1 1 2 2 4 4 4 4 1 1 1 1 4 4 1 1 1 1 2)

revision='main'

mkdir -p logs

# Submit jobs for each model
for i in "${!model_names[@]}"; do
    model="${model_names[$i]}"
    gpu_count="${gpu_counts[$i]}"
    model_short=$(echo "$model" | sed 's|.*/||')
    output_file="logs/generation_${model_short}.out"
    submit_job "do_generation.py" "$model" "$revision" "$gpu_count" "$output_file"
done