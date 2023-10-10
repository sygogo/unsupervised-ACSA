import os
import pickle

from transformers import AutoTokenizer, BertForMaskedLM
import torch
from tqdm import tqdm

from src.filter_words import filter_words

top_k = 5


class VocabGenerator:

    def __init__(self, bert_path, reviews, aspect_categories_list, aspect_categories_seed_list):
        self.mlm_model = BertForMaskedLM.from_pretrained(bert_path).cuda()
        self.tokenizer = AutoTokenizer.from_pretrained(bert_path)
        self.reviews = reviews
        self.aspect_categories_seeds = aspect_categories_seed_list
        self.aspect_categories_list = aspect_categories_list

    def __call__(self):
        aspect_categories = self.aspect_categories_list
        aspect_seeds = self.aspect_categories_seeds
        aspect_vocabularies = self.generate_vocabularies(aspect_categories, aspect_seeds, self.reviews)
        return aspect_vocabularies

    def generate_vocabularies(self, categories, seeds, reviews):
        # Initialise empty frequency table
        freq_table = {}
        for cat in categories:
            freq_table[cat] = {}

        # Populate vocabulary frequencies for each category
        for category in categories:
            print(f'Generating vocabulary for {category} category...')
            for line in tqdm(reviews):
                text = line.strip()
                seeds_set = self.aspect_categories_seeds[category]
                if len(set(text.split()) & set(seeds_set)) > 0:
                    ids = self.tokenizer(text, return_tensors='pt', truncation=True)['input_ids']
                    tokens = self.tokenizer.convert_ids_to_tokens(ids[0])
                    word_predictions = self.mlm_model(ids.cuda())[0]
                    word_scores, word_ids = torch.topk(word_predictions, top_k, -1)
                    word_ids = word_ids.squeeze(0)
                    for idx, token in enumerate(tokens):
                        if token in seeds[category]:
                            self.update_table(freq_table, category, self.tokenizer.convert_ids_to_tokens(word_ids[idx]))

        # Remove words appearing in multiple vocabularies (generate disjoint sets)
        for category in categories:
            for key in freq_table[category]:
                for cat in categories:
                    if freq_table[cat].get(key) != None and freq_table[cat][key] < freq_table[category][key]:
                        del freq_table[cat][key]

        vocabularies = {}

        for category in categories:
            words = []
            for key in freq_table[category]:
                words.append((freq_table[category][key], key))
            words.sort(reverse=True)
            vocabularies[category] = words
        return vocabularies

    def update_table(self, freq_table, cat, tokens):
        for token in tokens:
            if token in filter_words or '##' in token:
                continue
            freq_table[cat][token] = freq_table[cat].get(token, 0) + 1
