## GPT-3.5-Turbo-1106

echo "=======Running GPT-3.5-Turbo-1106======="
## step '1': Simply running JCB (no HarmBench classifier for final ASR computation)
# python ./scripts/run_pipeline.py --methods JCB --models gpt-3.5-turbo-1106 --step 1 --mode local --base_save_dir ./results

## step '3': Running the HarmBench classifier on the jailbreak prompts obtained from step '1' to compute the final ASR
# python ./scripts/run_pipeline.py --methods JCB --models gpt-3.5-turbo-1106 --step 3 --mode local --base_save_dir ./results

## step 'all': Running step '1' followed by step '3'
python ./scripts/run_pipeline.py --methods JCB --models gpt-3.5-turbo-1106 --step all --mode local --base_save_dir ./results


## Uncomment below for running on advbench_subset dataset (Note that we do not need to run step 3 for the advbench dataset)
# python ./scripts/run_pipeline.py --behaviors_path ./data/behavior_datasets/extra_behavior_datasets/advbench_subset_behaviors.csv --methods JCB --models gpt-3.5-turbo-1106 --step 1 --mode local --base_save_dir ./results_advbench_subset

