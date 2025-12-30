import re
from .judges import GPTJudge, OpenSourceJudge
from .language_models import GPT, Claude, Gemini, HuggingFace, VLLM, TogetherAILLM
from ..model_utils import get_template
import copy
import ray
import logging
import os

def load_JCB_models(targetLM_args,
                    judgeLM_args,
                    target_max_n_tokens,
                    judge_max_n_tokens,
                    ):
    ray.init(ignore_reinit_error=True)

    t_name, j_name = [args['model_name_or_path'] for args in [targetLM_args, judgeLM_args]] 

    # ====== Load target model =========
    preloaded_target_model = None
    targetLM = TargetLM(max_n_tokens=target_max_n_tokens,
                        temperature=0.0,
                        preloaded_model=preloaded_target_model,
                        **targetLM_args
                        )
    
    # ====== Load judge model =========
    preloaded_judge_model = None
    if j_name == t_name: 
        preloaded_judge_model = targetLM.model
        judgeLM_args = targetLM_args

    judgeLM = load_judge(max_n_tokens=judge_max_n_tokens, preloaded_model=preloaded_judge_model, **judgeLM_args)
    return targetLM, judgeLM


class TargetLM():
    """
        Base class for target language models.
        
        Generates responses for prompts using a language model. The self.model attribute contains the underlying generation model.
    """
    def __init__(self, 
            max_n_tokens: int, 
            temperature: float,
            preloaded_model: object = None,
            **model_kwargs, 
            ):
        
        self.model_name_or_path = model_kwargs['model_name_or_path']
        self.temperature = temperature
        self.max_n_tokens = max_n_tokens

        self.template = get_template(**model_kwargs, return_fschat_conv=True)
        self.model = preloaded_model if preloaded_model else load_indiv_model(**model_kwargs)
        
        # check if model is an opensource model which is a RayActor
        self.use_ray = isinstance(self.model, ray.actor.ActorHandle)

    def get_response(self, prompts_list, context_str=None):
        logging.info(f"self.model: {self.model}")
        logging.info(f"self.model type: {type(self.model)}")
        batchsize = len(prompts_list)
        convs_list = [copy.deepcopy(self.template) for _ in range(batchsize)]
        full_prompts = []
        for conv, prompt in zip(convs_list, prompts_list):
            if context_str:
                prompt = f"{context_str}\n\n---\n\n{prompt}"

            conv.append_message(conv.roles[0], prompt)
            if ("gpt" in self.model_name_or_path) and ("gpt-oss" not in self.model_name_or_path):
                full_prompts.append([{"role": "user", "content": prompt}])
            elif "gemini-" in self.model_name_or_path:
                full_prompts.append(conv.messages[-1][1])
            elif "mistralai/Mixtral-8x7B-Instruct-v0.1" in self.model_name_or_path: ## For using TogetherAI API for Mixtral-8X7B since for faster inference
                full_prompts.append([{"role": "user", "content": prompt}])
            elif ("meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8" in self.model_name_or_path)  or ("meta-llama/Llama-3-8b-chat-hf" in self.model_name_or_path) or ("meta-llama/Llama-3-70b-chat-hf" in self.model_name_or_path) or ("meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo" in self.model_name_or_path) or ("openai/gpt-oss-20b" in self.model_name_or_path) or ("deepseek-ai/DeepSeek-V3.1" in self.model_name_or_path) or ("deepseek-ai/DeepSeek-R1" in self.model_name_or_path): ## For using TogetherAI API
                full_prompts.append([{"role": "user", "content": prompt}])
            elif ("claude-3-5-haiku-20241022" in self.model_name_or_path) or ("claude-3-7-sonnet-20250219" in self.model_name_or_path):
                full_prompts.append([{"role": "user", "content": prompt}])
            else:
                if "llama-2" in self.model_name_or_path.lower():
                    conv.append_message(conv.roles[1], None)
                    full_prompts.append(conv.get_prompt())
                else:
                    new_conv = copy.deepcopy(conv)
                    new_conv.system_message = ''
                    new_conv.append_message(new_conv.roles[1], None)
                    full_prompts.append(new_conv.get_prompt())
        
        generate_function = self.model.batched_generate if not self.use_ray else self.model.batched_generate.remote
        logging.info(f"full_prompts sent to batched_generate: {full_prompts}")
        outputs_list = generate_function(full_prompts, 
                                        max_n_tokens = self.max_n_tokens,  
                                        temperature = self.temperature,
                                        stop_tokens = ["}"]
                                        )
        logging.info(f"full_response: {outputs_list}")
        return outputs_list if not self.use_ray else ray.get(outputs_list)

def load_judge(model_name_or_path, max_n_tokens, preloaded_model=None, token=None, **model_kwargs):
    if "gpt" in model_name_or_path:
        judge_model = GPT(model_name = model_name_or_path, token=token) 
        return GPTJudge(model_name_or_path, max_n_tokens, judge_model)
    else:
        if preloaded_model is None:
            preloaded_model = load_indiv_model(model_name_or_path=model_name_or_path, **model_kwargs)
        return OpenSourceJudge(model_name_or_path, max_n_tokens, judge_model=preloaded_model)

def load_indiv_model(model_name_or_path, use_vllm=False, token=None, **model_kwargs):
    if ('gpt-' in model_name_or_path) and ("gpt-oss" not in model_name_or_path):
        lm = GPT(model_name_or_path, token=token)
    elif 'claude-' in model_name_or_path:
        lm = Claude(model_name_or_path, token=token)
    elif 'gemini-' in model_name_or_path:
        lm = Gemini(model_name_or_path, token=token)
    elif "mistralai/Mixtral-8x7B-Instruct-v0.1" in model_name_or_path: ## For using TogetherAI API for Mixtral-8X7B since for faster inference
        # os.environ["TOGETHER_API_KEY"] = token
        lm = TogetherAILLM(model_name_or_path, token=token)
    elif ("meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8" in model_name_or_path) or ("meta-llama/Llama-3-8b-chat-hf" in model_name_or_path) or ("meta-llama/Llama-3-70b-chat-hf" in model_name_or_path) or ("meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo" in model_name_or_path) or ("openai/gpt-oss-20b" in model_name_or_path) or ("deepseek-ai/DeepSeek-V3.1" in model_name_or_path) or ("deepseek-ai/DeepSeek-R1" in model_name_or_path): ## For using TogetherAI API
        lm = TogetherAILLM(model_name_or_path, token=token)
    else:
        # We use ray to manage opensource models here
        num_gpus = model_kwargs.get("num_gpus", 1)
        if use_vllm:
            # need to init as ray actors to load multiple vllm models in one environment (Related issue: https://github.com/vllm-project/vllm/issues/1676)
            # vllm.engine.ray_utils.py will allocate num_gpus for tensor_parallel (num_gpus) > 1 so we don't pass in num_gpus
            options = {"num_gpus": 1} if num_gpus == 1 else {}
            lm = VLLM.options(**options).remote(model_name_or_path, token=token, **model_kwargs)
        else:
            lm = ray.remote(HuggingFace).options(num_gpus=num_gpus).remote(model_name_or_path, token=token, **model_kwargs)
        ray.get(lm.is_initialized.remote()) # init ray model
    return lm
