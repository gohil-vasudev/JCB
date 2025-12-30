from tqdm import tqdm
import numpy as np
from ..baseline import SingleBehaviorRedTeamingMethod
from .conversers import load_JCB_models
import copy
import json

import os
import csv
import logging
import pandas as pd
import random
import pickle
import time
import yaml

from .selector import Selector
from .mutator import Mutator
from .evaluator import Evaluator

class JCB(SingleBehaviorRedTeamingMethod):
    use_ray = True
    
    def __init__(self, target_model, selector_algorithm,
                synonym_replacement_prob, judge_model, num_iterations, timeout_thresh,
                target_max_n_tokens, judge_max_n_tokens, question_jailbreak_count_thresh,
                LLM_judge_cutoff_score=8.5, **kwargs):

        self.selector_algorithm = selector_algorithm
        self.synonym_replacement_prob = synonym_replacement_prob
        self.LLM_judge_cutoff_score = LLM_judge_cutoff_score
        self.question_jailbreak_count_thresh = question_jailbreak_count_thresh
        self.num_iterations = num_iterations
        self.timeout_thresh = timeout_thresh
        targetLM_args, judgeLM_args = target_model, judge_model

        self.targetLM, self.judgeLM = load_JCB_models(targetLM_args,
                                                           judgeLM_args,
                                                           target_max_n_tokens,
                                                           judge_max_n_tokens,
                                                        )


    def generate_test_cases(self, base_save_dir, behaviors, verbose=False):
        """
        Generates test cases for the provided behaviors. The outputs of this method are passed to the
        save_test_cases method, which saves test cases and logs to disk.

        :param behaviors: a list of behavior dictionaries specifying the behaviors to generate test cases for
        :param verbose: whether to print progress
        :return: a dictionary of test cases, where the keys are the behavior IDs and the values are lists of test cases
        """

        self.base_save_dir = base_save_dir
        test_cases = {}
        logs = {}
        test_cases, logs, attack_success_completions = self.run_jcb(behaviors, verbose)

        
        model_name_or_path = self.targetLM.model_name_or_path
        with open("./configs/model_configs/models.yaml") as file:
            model_configs = yaml.full_load(file)
        for tmp_model_name in model_configs:
            if model_configs[tmp_model_name]["model"]["model_name_or_path"] == model_name_or_path:
                model_name = tmp_model_name
                break
        experiment_name = model_name
        save_path = os.path.join(self.base_save_dir, "JCB", experiment_name, 'completions', f'{model_name}.json')
        print(f'Saving completions to {save_path}...')
        os.makedirs(os.path.dirname(save_path), exist_ok=True) if os.path.dirname(save_path) else None
        with open(save_path, 'w') as file:
            json.dump(attack_success_completions, file, indent=4)
        print('Saved generated completions')

        return test_cases, logs
    
    def run_jcb(self, behaviors_list, verbose):
        start_time = time.time()
        behaviors_data = {row['BehaviorID']: row for row in behaviors_list}
        model_name = self.targetLM.model_name_or_path
        stop_time = 0
        
        with open("./configs/model_configs/models.yaml") as file:
            model_configs = yaml.full_load(file)
        experiment_name = model_name
        for tmp_model_name in model_configs:
            if model_configs[tmp_model_name]["model"]["model_name_or_path"] == model_name:
                experiment_name = tmp_model_name
                break
        results_dir = os.path.join(self.base_save_dir, "JCB", experiment_name, "JCB_logs", "")
        

        os.makedirs(results_dir,exist_ok=True)

        full_experiment_config = f"model_name_{model_name}_selector_{self.selector_algorithm}_LLM_judge_{self.judgeLM.judge_name}_cutoff_score_{self.LLM_judge_cutoff_score}_synonym_replacement_prob_{self.synonym_replacement_prob}"
        experiment_config = f"{model_name}_{self.selector_algorithm}_{self.judgeLM.judge_name}_{self.LLM_judge_cutoff_score}_{self.synonym_replacement_prob}"
        experiment_config = experiment_config.replace("/","_")
        full_experiment_config = full_experiment_config.replace("/","_")
        print(f"full_experiment_config: {full_experiment_config}")
        
        ## op file names
        successful_cases_save_filepath=f"{results_dir}successful_cases_{experiment_config}.csv"
        log_file_path=f"{results_dir}experiment_{experiment_config}.log"
        log_dict_file_path=f"{results_dir}experiment_{experiment_config}_log_dict.pkl"
        log_csv_file_path=f"{results_dir}experiment_{experiment_config}.csv"
        
        jailbroken_questions = []
        jailbroken_behavior_ids = []
        
        my_init_seeds_filepath = "./baselines/JCB/initial_seeds_from_ChatGPT.csv"

        initial_seeds = self.get_init_seeds(my_init_seeds_filepath)
        
        
        
        runner = Runner(initial_seeds, behaviors_data,\
                        experiment_name=experiment_name,\
                        targetLM=self.targetLM,\
                        question_placeholder_text="[INSERT PROMPT HERE]",\
                        selector_algorithm=self.selector_algorithm,\
                        LLM_judge_for_contextual_behaviors=self.judgeLM,\
                        LLM_judge_cutoff_score=self.LLM_judge_cutoff_score,\
                        successful_cases_save_filepath=successful_cases_save_filepath,\
                        log_file_path=log_file_path,\
                        log_dict_file_path=log_dict_file_path,\
                        log_csv_file_path=log_csv_file_path,\
                        synonym_replacement_prob=self.synonym_replacement_prob,\
                        question_jailbreak_count_thresh=self.question_jailbreak_count_thresh,\
                        start_time=start_time,timeout_thresh=self.timeout_thresh,\
                        full_experiment_config=full_experiment_config,\
                        jailbroken_questions=jailbroken_questions,jailbroken_behavior_ids=jailbroken_behavior_ids)
        stop_time, test_cases, logs, attack_success_completions = runner.run(iterations=self.num_iterations)
        
        
        print(f"Number of successes: {np.sum([v['reward'] for v in runner.log])}/{len(runner.log)}")
        print(f"Average reward: {np.average([v['reward'] for v in runner.log])}")
        print(f"Total runtime: {stop_time-start_time}s")

        return test_cases, logs, attack_success_completions

    def get_init_seeds(self,seed_template_filepath):
        df = pd.read_csv(seed_template_filepath)
        seeds = df["text"].values
        return seeds


class Runner:
    def __init__(self, initial_seeds, behaviors_data,\
                 experiment_name=None,\
                 targetLM=None,\
                 question_placeholder_text="[INSERT PROMPT HERE]",\
                 selector_algorithm="weighted_random",\
                 LLM_judge_for_contextual_behaviors=None,\
                 LLM_judge_cutoff_score=8.5,\
                 successful_cases_save_filepath="./successful_cases.csv",\
                 log_file_path="./experiment.log",\
                 log_dict_file_path="./experiment_log_dict.pkl",\
                 log_csv_file_path="./experiment.csv",\
                 synonym_replacement_prob=0.1,\
                 question_jailbreak_count_thresh=1,\
                 start_time=None,timeout_thresh=None,\
                 full_experiment_config=None, jailbroken_questions=None,\
                 jailbroken_behavior_ids=None):
        self.seed_pool = initial_seeds.copy()
        self.behaviors_data = copy.deepcopy(behaviors_data)
        self.behavior_ids = list(self.behaviors_data.keys())
        self.questions = [self.behaviors_data[id]['Behavior'] for id in self.behaviors_data]
        self.targetLM = targetLM
        self.experiment_name = experiment_name
        self.question_jailbreak_counts = {q:0 for q in self.questions}
        self.behavior_id_jailbreak_counts = {id:0 for id in self.behavior_ids}
        self.question_placeholder_text = question_placeholder_text
        self.selector = Selector.create_selector(selector_algorithm, initial_seeds)
        self.synonym_replacement_prob=synonym_replacement_prob
        self.mutator = Mutator(self.question_placeholder_text,self.synonym_replacement_prob)
        self.evaluator = Evaluator()
        self.LLM_judge_for_contextual_behaviors = LLM_judge_for_contextual_behaviors
        self.LLM_judge_cutoff_score = LLM_judge_cutoff_score
        self.question_jailbreak_count_thresh = question_jailbreak_count_thresh
        self.start_time = start_time
        self.timeout_thresh = timeout_thresh
        self.jailbroken_questions = jailbroken_questions
        self.jailbroken_behavior_ids = jailbroken_behavior_ids
        self.log = []  # To track the process
        self.successful_cases = []  # To store successful seeds and responses
        self.successful_cases_save_filepath = successful_cases_save_filepath
        # Configure the logger
        logging.basicConfig(filename=log_file_path, filemode='w', force=True, level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')
        logging.info(f"Full Experiment Config: {full_experiment_config}")
        self.log_dict_file_path = log_dict_file_path
        self.log_csv_file_path = log_csv_file_path

        self.test_cases = {}
        self.logs = {}
        self.attack_success_completions = {}

    def log_iteration(self, iteration, elapsed_time, target_question, target_behavior_id, selected_seed, mutated_seed, final_prompt, response_content, reward):
        self.log.append({
            "iteration": iteration,
            "elapsed_time": elapsed_time,
            "target_behavior_id": target_behavior_id,
            "target_question": target_question,
            "selected_seed": selected_seed,
            "mutated_seed": mutated_seed,
            "final_prompt": final_prompt,
            "response_content": response_content,
            "reward": reward
        })
        if self.target_behavior_id not in self.logs:
            self.logs[self.target_behavior_id] = []
        self.logs[self.target_behavior_id].append({
            "iteration": iteration,
            "elapsed_time": elapsed_time,
            "target_behavior_id": target_behavior_id,
            "target_question": target_question,
            "selected_seed": selected_seed,
            "mutated_seed": mutated_seed,
            "final_prompt": final_prompt,
            "response_content": response_content,
            "reward": reward
        })

    def save_successful_cases(self):
        with open(self.successful_cases_save_filepath, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["Target Behavior ID", "Target Question", "Selected Seed", "Mutated Seed", "Response"])
            for case in self.successful_cases:
                writer.writerow([case["target_behavior_id"], case["target_question"], case["selected_seed"], case["mutated_seed"], case["response_content"]])
                        
                
    def fill_question_placeholder(self, mutated_seed):
        tags = self.behaviors_data[self.target_behavior_id]['Tags'].split(', ')
        if 'context' in tags:
            context_string = self.behaviors_data[self.target_behavior_id]["ContextString"]
            final_prompt = mutated_seed.replace(self.question_placeholder_text,self.target_question + ": " + context_string)
        else:
            final_prompt = mutated_seed.replace(self.question_placeholder_text,self.target_question)
        return final_prompt

    def run(self, iterations=100):
        logging.info("JCB run started")
        exit_condition = False
        exit_condition_jailbroke_all_questions = False
        exit_condition_timeout = False

        for iteration in tqdm(range(iterations)):
            try:
                logging.info(f"iteration: {iteration}")
                self.target_behavior_id = random.choice(self.behavior_ids)
                self.target_question = self.behaviors_data[self.target_behavior_id]['Behavior']
                if self.target_question in self.jailbroken_questions:
                    print(f"Note! {self.target_question}, {self.target_behavior_id} has already been jailbroken")
                    print(f"Current self.behavior_ids: {self.behavior_ids}")
                    logging.info(f"Note! {self.target_question}, {self.target_behavior_id} has already been jailbroken")
                    logging.info(f"Current self.behavior_ids: {self.behavior_ids}")
                
                logging.info(f"target_behavior_id: {self.target_behavior_id}")
                logging.info(f"target_question: {self.behaviors_data[self.target_behavior_id]['Behavior']}")
                logging.debug("Selection started")
                selected_seed = self.selector.select_seed()
                logging.debug("Selection completed")
                logging.info(f"selected_seed: {selected_seed}")
                if self.synonym_replacement_prob>0:
                    logging.debug("Mutation started")
                    mutated_seed = self.mutator.mutate_seed(selected_seed)
                    logging.debug("Mutation completed")
                else:
                    logging.info("No mutation (for ablation)")
                    mutated_seed = copy.deepcopy(selected_seed)
                
                logging.info(f"mutated_seed: {mutated_seed}")
                final_prompt = self.fill_question_placeholder(mutated_seed)
                logging.debug("Question placeholder replaced")
                logging.info(f"final_prompt: {final_prompt}")
                self.test_cases[self.target_behavior_id] = [final_prompt]
                logging.debug("Target LLM invoked")
                response_content = self.targetLM.get_response([final_prompt], context_str=None)[0] ## Forcing this context_str to None because I want the question_placeholder function to format the final prompt
                self.attack_success_completions[self.target_behavior_id] = [dict(test_case=final_prompt, generation=response_content)]
                logging.info(f"response_content: {response_content}")
                logging.debug("Target LLM response collected")
                logging.debug("Evaluation started")
                reward = self.evaluator.evaluate_response(self.behaviors_data, self.target_behavior_id, final_prompt, response_content, self.LLM_judge_for_contextual_behaviors, self.LLM_judge_cutoff_score)
                logging.info(f"reward: {reward}")
                logging.debug("Evaluation completed")
    
                self.selector.update_weights(selected_seed, reward)

                if reward > 0:
                    self.successful_cases.append({"target_behavior_id": self.target_behavior_id, "target_question": self.target_question, "selected_seed": selected_seed, "mutated_seed": mutated_seed, "response_content": response_content}) 
                    self.test_cases[self.target_behavior_id] = [final_prompt]
                    if self.target_question in self.jailbroken_questions:
                        print(f"Note! {self.target_question}, {self.target_behavior_id} has already been jailbroken")
                        print(f"Current self.behavior_ids: {self.behavior_ids}")
                        logging.info(f"Note! {self.target_question}, {self.target_behavior_id} has already been jailbroken")
                        logging.info(f"Current self.behavior_ids: {self.behavior_ids}")
                    
                    self.question_jailbreak_counts[self.target_question]+=1
                    if self.question_jailbreak_counts[self.target_question] >= self.question_jailbreak_count_thresh:
                        logging.info(f"Removing the following question from consideration as it has been jailbroken {self.question_jailbreak_count_thresh} times: {self.target_question}")
                        logging.info(f"Removing the following behavior id from consideration as it has been jailbroken {self.question_jailbreak_count_thresh} times: {self.target_behavior_id}")
                        
                        self.questions.remove(self.target_question)
                        self.behavior_ids.remove(self.target_behavior_id)
                        self.jailbroken_questions.append(self.target_question)
                        self.jailbroken_behavior_ids.append(self.target_behavior_id)
                        if len(self.jailbroken_questions)==len(self.behaviors_data):
                            exit_condition_jailbroke_all_questions = True

                
                elapsed_time = time.time() - self.start_time
                self.log_iteration(iteration, elapsed_time, self.target_question, self.target_behavior_id, selected_seed, mutated_seed, final_prompt, response_content, reward)

                if elapsed_time >= self.timeout_thresh:
                    exit_condition_timeout = True
                if exit_condition_jailbroke_all_questions or exit_condition_timeout:
                    exit_condition = True
                
                if exit_condition:
                    if exit_condition_jailbroke_all_questions:
                        logging.info("Exiting since all questions have been jailbroken")
                        print("Exiting since all questions have been jailbroken")
                    if exit_condition_timeout:
                        logging.info(f"Exiting since time limit of {self.timeout_thresh} seconds reached")
                        print(f"Exiting since time limit of {self.timeout_thresh} seconds reached")
                    break
                
                if os.path.exists(f"STOP_{self.experiment_name}"):
                    print("Caught StopEarly: stopping early.")
                    logging.info("Caught StopEarly: stopping early.")
                    self.save_successful_cases()
                    stop_time = time.time()
                    logging.info("JCB run completed")
                    print("JCB run completed")
                    logging.info(f"No. of questions successfully jailbroken within {iteration+1} iterations: {len(self.jailbroken_questions)}/{len(self.behaviors_data)}")
                    logging.info(f"ASR: {len(self.jailbroken_questions)*100/len(self.behaviors_data)} %")
                    logging.info(f"Number of successes: {np.sum([v['reward'] for v in self.log])}/{len(self.log)}")
                    logging.info(f"Average reward: {np.average([v['reward'] for v in self.log])}")
                    logging.info(f"Total runtime: {stop_time-self.start_time}s")
                    print(f"No. of questions successfully jailbroken within {iteration+1} iterations: {len(self.jailbroken_questions)}/{len(self.behaviors_data)}")
                    print(f"ASR: {len(self.jailbroken_questions)*100/len(self.behaviors_data)} %")
                    self.save_successful_cases()
                    with open(self.log_dict_file_path,'wb') as f:
                        pickle.dump(self.log,f)
                    with open(self.log_csv_file_path, mode='w', newline='') as file:
                        writer = csv.writer(file)
                        col_heads = list(self.log[0].keys())
                        writer.writerow(col_heads)
                        for idx in range(len(self.log)):
                            writer.writerow([self.log[idx][c] for c in col_heads])
                    os.remove(f"STOP_{self.experiment_name}")
                    return stop_time, self.test_cases, self.logs, self.attack_success_completions
            

            except Exception as e:
                self.save_successful_cases()
                logging.info(f"An error occurred at iteration {iteration}: {e}")
                print(f"An error occurred at iteration {iteration}: {e}")
        
        
        stop_time = time.time()
        logging.info("JCB run completed")
        print("JCB completed")
        logging.info(f"No. of questions successfully jailbroken within {iteration+1} iterations: {len(self.jailbroken_questions)}/{len(self.behaviors_data)}")
        logging.info(f"ASR: {len(self.jailbroken_questions)*100/len(self.behaviors_data)} %")
        logging.info(f"Number of successes: {np.sum([v['reward'] for v in self.log])}/{len(self.log)}")
        logging.info(f"Average reward: {np.average([v['reward'] for v in self.log])}")
        logging.info(f"Total runtime: {stop_time-self.start_time}s")
        print(f"No. of questions successfully jailbroken within {iteration+1} iterations: {len(self.jailbroken_questions)}/{len(self.behaviors_data)}")
        print(f"ASR: {len(self.jailbroken_questions)*100/len(self.behaviors_data)} %")
        self.save_successful_cases()
        with open(self.log_dict_file_path,'wb') as f:
            pickle.dump(self.log,f)
        with open(self.log_csv_file_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            col_heads = list(self.log[0].keys())
            writer.writerow(col_heads)
            for idx in range(len(self.log)):
                writer.writerow([self.log[idx][c] for c in col_heads])
        return stop_time, self.test_cases, self.logs, self.attack_success_completions


