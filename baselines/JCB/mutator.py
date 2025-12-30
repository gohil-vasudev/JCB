#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from nltk.corpus import wordnet
from nltk.corpus import stopwords
from nltk import pos_tag
import random
import re


def get_wordnet_pos(treebank_tag):
    """Convert Treebank POS tags to WordNet POS tags."""
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return None     



class Mutator:
    def __init__(self,question_placeholder_text,synonym_replacement_prob):
        self.question_placeholder_text = question_placeholder_text
        self.ignore_regex = re.compile(re.escape(self.question_placeholder_text))
        self.synonym_replacement_prob = synonym_replacement_prob
    
    def replace_word_advanced(self, match):
        word = match.group()
        pos = pos_tag([word])[0][-1]
        if random.random() < self.synonym_replacement_prob:
            stop_words = set(stopwords.words('english'))
            if self.ignore_regex.fullmatch(word) or word.lower() in stop_words:
                return word
            if word.isalpha():
                wn_pos = get_wordnet_pos(pos)
                return self.get_synonym_advanced(word,pos=wn_pos)
            return word
        else:
            return word
    
    def get_synonym_advanced(self, word, pos=None):
        synonyms = wordnet.synsets(word, pos=pos) if pos else wordnet.synsets(word)
        if synonyms and synonyms[0].lemmas():
            synonym = random.choice(synonyms).lemmas()[0].name()
            synonym = synonym.replace("_", " ")
            return synonym
        return word

    def mutate_seed(self, seed):
        regex_pattern = re.escape(self.question_placeholder_text).replace(r"\ ", " ") + r"|\b\w+\b|[^\w\s]" # To use dynamic string variable instead of hardcoding the placeholder
        output_string = re.sub(regex_pattern, self.replace_word_advanced, seed)
        return output_string