import json
import os.path
import pickle

from src.AspectLabeler import AspectLabeler
from src.VocabGenerator import VocabGenerator

train_data_path = "data/raw/restaurant_2014_train.tsv"
test_data_path = "data/raw/restaurant_2014_test.tsv"
bert_path = "/data/wangsiyu/data/bert-base-uncased"
config_json_path = 'config.json'
processed_data_path = 'data/processed'

aspect_config = json.load(open(config_json_path, 'r'))

if __name__ == '__main__':
    domain = 'restaurant'
    reviews = []
    aspect_vocabularies_file = '{}/{}-aspect_vocabularies.pkl'.format(processed_data_path, domain)
    aspect_category_seed_list = aspect_config[domain]
    aspect_category_list = [k for k in aspect_config[domain]]
    # step1 extend aspect vocabularies
    if os.path.exists(aspect_vocabularies_file):
        aspect_vocabularies = pickle.load(open(aspect_vocabularies_file, 'rb'))
    else:
        for line in open(train_data_path, 'r').readlines():
            sentence = line.split('\t')[0]
            reviews.append(sentence)
        vocabGenerator = VocabGenerator(bert_path, reviews, aspect_category_list, aspect_category_seed_list)
        aspect_vocabularies = vocabGenerator()
        pickle.dump(aspect_vocabularies, open(aspect_vocabularies_file, 'wb'))

    # step2 label sentence
    labeler = AspectLabeler(bert_path, aspect_vocabularies)
    for line in open(test_data_path, 'r').readlines():
        sentence = line.split('\t')[0]
        aspect = labeler.assign_aspect(sentence)
        print(sentence)
        print(aspect)
