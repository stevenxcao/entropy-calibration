# entropy-calibration
This repository contains code for the following paper:  
[On the Entropy Calibration of Language Models](https://neurips.cc/virtual/2025/loc/san-diego/poster/119303)  
Steven Cao, Gregory Valiant, Percy Liang  
Neurips 2025
### Dependencies
This code was tested with `python 3.12.9`, `pytorch 2.6.0`, `transformers 4.52.4`, `datasets 3.5.0`, `bitsandbytes 0.45.4`, `nltk 3.9.1`, `vllm 0.8.2`, and `xformers 0.0.29.post2`.
### Running the code
To reproduce the plots in the paper, first run `submit_generation_jobs.sh` and wait for the jobs to finish. Then, run `submit_ents_jobs.sh` and wait for the jobs to finish. Finally, run `load_and_plot.ipynb` to produce the plots.
