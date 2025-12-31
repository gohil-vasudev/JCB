# JCB
Official implementation of my LLM jailbreaking paper "Effective and Efficient Jailbreaks of Black-Box LLMs with Cross-Behavior Attacks".


## Abstract
[WIP]



## Quick Start

### Installation

```bash
git clone https://github.com/gohil-vasudev/JCB.git
cd JCB
conda create --name JCB_env python=3.12
conda activate JCB_env
pip install -r requirements.txt
```

### Compute Requirements
Although the core JCB algorithm does not require any GPUs and can be run on a low-end consumer-grade CPU, GPUs may be required to query the target LLMs and to use the HarmBench classifier for determining if a response constitutes a successful jailbreak or not. The actual GPU configuration required depends on the target LLM, whether it is available as a cloud service (e.g., closed-source LLMs) or not, and its number of parameters. The largest non-cloud-based LLM we target is Llama-2-70B-Chat, and it requires a GPU with 140GB VRAM. So, we recommend running JCB on a machine with one Nvidia A100 80GB GPU for target LLMs with <15B parameters. For larger target LLMs (e.g., Llama-2-70B-Chat), a machine with at least two Nvidia A100 80GB GPUs will be required.

### Running JCB

To run JCB, you need to use the `./scripts/run_pipeline.py`. Examples are shown in the `run_gpt_3.5_turbo_1106.sh` and `run_llama.sh` bash scripts.
Note that you need to set the appropriate API keys in [configs/method_configs/JCB_config.yaml](configs/method_configs/JCB_config.yaml) and [configs/model_configs/models.yaml](configs/model_configs/models.yaml). Additionally, you will also need your HuggingFace token for using gated models from HuggingFace (e.g., Llama2). See [run_llama.sh](run_llama.sh) for an example of how your HuggingFace token can be set.

Examples for running JCB on GPT-3.5-Turbo-1106 and Llama-2-7b-chat-hf are shown below:
```bash
# Run JCB against GPT-3.5-Turbo-1106 on the HarmBench dataset
python ./scripts/run_pipeline.py --methods JCB --models gpt-3.5-turbo-1106 --step all --mode local --base_save_dir ./results

# Run JCB against GPT-3.5-Turbo-1106 on the AdvBench subset dataset
python ./scripts/run_pipeline.py --behaviors_path ./data/behavior_datasets/extra_behavior_datasets/advbench_subset_behaviors.csv --methods JCB --models gpt-3.5-turbo-1106 --step 1 --mode local --base_save_dir ./results_advbench_subset

# Run JCB against Llama-2-7b-chat-hf on the HarmBench dataset
huggingface-cli login --token <your_huggingface_token>
python ./scripts/run_pipeline.py --methods JCB --models llama2_7b --step all --mode local --base_save_dir ./results
```

The final attack success rate (ASR) results for the HarmBench dataset (which requires running the HarmBench classifier) are saved in the `results/JCB/<target_model>/results/` directory and the logs can be viewed in the `results/JCB/<target_model>/JCB_logs/` directory. Since we do not need the HarmBench classifier for the AdvBench dataset, its final ASR results can be seen directly in the `results/JCB/<target_model>/JCB_logs/` directory.


### Evaluating JCB Against Your Own Models
All models reported in our paper are already supported in our codebase. However, if you wish to evaluate JCB against other HuggingFace transformers models, please refer to the instructions in the HarmBench repository.


## Acknowledgments and Citation

This codebase is heavily inspired from the HarmBench repository ([link](https://github.com/centerforaisafety/HarmBench)). We thank the authors of HarmBench for open-sourcing their code. If aspects of the HarmBench repository appearing in JCB are useful to you in your research, we ask that you consider citing the HarmBench paper.

If you find JCB useful in your research, please consider citing our paper (link and bib entry TBD)
