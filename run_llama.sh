huggingface-cli login --token <your_huggingface_token>

# Llama 2 7b and 13b models

echo "=======Running Llama2-7B======="
# python -u ./scripts/run_pipeline.py --methods JCB --models llama2_7b --step 1 --mode local --base_save_dir ./results
# python -u ./scripts/run_pipeline.py --methods JCB --models llama2_7b --step 3 --mode local --base_save_dir ./results
python -u ./scripts/run_pipeline.py --methods JCB --models llama2_7b --step all --mode local --base_save_dir ./results
## Deleting the model after its experiment is complete to manage disk space
rm -rf ~/.cache/huggingface/hub/models--*Llama*2*7b*

echo "=======Running Llama2-13B======="
# python -u ./scripts/run_pipeline.py --methods JCB --models llama2_13b --step 1 --mode local  --base_save_dir ./results
# python -u ./scripts/run_pipeline.py --methods JCB --models llama2_13b --step 3 --mode local  --base_save_dir ./results
python -u ./scripts/run_pipeline.py --methods JCB --models llama2_13b --step all --mode local  --base_save_dir ./results
## Deleting the model after its experiment is complete to manage disk space
rm -rf ~/.cache/huggingface/hub/models--*Llama*2*13b*chat*hf*
