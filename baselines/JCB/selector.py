#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import random
import numpy as np

class Selector:
    @staticmethod
    def create_selector(algorithm, initial_seeds):
        if algorithm == "random":
            return RandomSelector(initial_seeds)
        elif algorithm == 'weighted_random':
            return WeightedRandomSelector(initial_seeds)
        elif algorithm == "epsilon_greedy":
            return EpsilonGreedySelector(initial_seeds)
        elif algorithm == "ucb":
            return UCBSelector(initial_seeds)
        else:
            raise ValueError("Unsupported selector algorithm")

class BaseSelector:
    def __init__(self, initial_seeds):
        self.rewards = {seed: 0 for seed in initial_seeds}
        self.weights = {seed: 1.0 for seed in initial_seeds}

    def update_weights(self, seed, reward):
        raise NotImplementedError("Must be implemented by subclasses")

class RandomSelector(BaseSelector):
    def select_seed(self):
        return random.choice(list(self.rewards.keys()))

    def update_weights(self, seed, reward):
        self.rewards[seed] += reward

class WeightedRandomSelector(BaseSelector):
    def select_seed(self):
        # Use self.rewards as weights for selecting a seed
        seeds = list(self.rewards.keys())
        return random.choices(seeds, weights=list(self.weights.values()), k=1)[0]

    def update_weights(self, seed, reward):
        self.rewards[seed] += reward
        if reward>0:
            self.weights[seed]+=reward

class EpsilonGreedySelector(BaseSelector):
    def select_seed(self):
        epsilon = 0.1
        if random.random() < epsilon:
            return random.choice(list(self.rewards.keys()))
        else:
            return max(self.rewards, key=lambda x: self.rewards[x])

    def update_weights(self, seed, reward):
        self.rewards[seed] += reward

class UCBSelector(BaseSelector):
    def __init__(self, initial_seeds):
        super().__init__(initial_seeds)
        self.seed_counts = {seed: 0 for seed in initial_seeds}  # Tracks how many times each seed was selected
        self.total_count = 0  # Total number of selections
    
    def select_seed(self):
        if self.total_count == 0:
            # If no seeds have been selected yet, choose a random seed to start
            return random.choice(list(self.rewards.keys()))
        # Apply the UCB formula to select the seed
        def ucb_score(seed):
            if self.seed_counts[seed] == 0:
                return float('inf')  # Ensures unvisited seeds are explored first
            avg_reward = self.rewards[seed] / self.seed_counts[seed]
            exploration_term = np.sqrt(2 * np.log(self.total_count) / self.seed_counts[seed])
            return avg_reward + exploration_term
        # Select the seed with the highest UCB score
        return max(self.rewards.keys(), key=ucb_score)
    def update_weights(self, seed, reward):
        # Update rewards, counts, and total count
        self.rewards[seed] += reward
        self.seed_counts[seed] += 1
        self.total_count += 1
    
