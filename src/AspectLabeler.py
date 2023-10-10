from transformers import BertForMaskedLM, AutoTokenizer, BertTokenizer
import torch
import math

from src.filter_words import filter_words


class AspectLabeler(object):
    def __init__(self, bert_path, aspect_vocabularies):
        self.mlm_model = BertForMaskedLM.from_pretrained(bert_path).cuda()
        self.tokenizer = BertTokenizer.from_pretrained(bert_path)
        self.aspect_vocabularies = aspect_vocabularies
        self.seed_set = self.get_seed_set()
        self.aspect_seed_sequences = self.construct_aspect_seed_sequences()

    def onehot(self, input, num_classes):
        one_hot = torch.zeros(size=(4, num_classes))
        one_hot.scatter_(dim=1, index=input, value=1)
        return one_hot.cuda()

    def get_seed_set(self):
        seed_set = set([])
        for cate in self.aspect_vocabularies:
            seeds = self.aspect_vocabularies[cate]
            for s in seeds:
                seed_set.add(s[1])
        return seed_set

    def construct_aspect_seed_sequences(self):
        seed_list = []
        length = []
        cate_list = []
        for cate in self.aspect_vocabularies:
            seeds = self.aspect_vocabularies[cate]
            seed_list.append(' '.join([k for i, k in seeds]))
            length.append(len(seeds))
            cate_list.append(cate)
        data = self.tokenizer(seed_list, add_special_tokens=False, return_tensors='pt', truncation=True, padding=True).data
        sentence_token_ids, token_type_ids, attention_mask = data['input_ids'], data["token_type_ids"], data["attention_mask"]
        return self.onehot(sentence_token_ids, self.tokenizer.vocab_size), torch.Tensor(length).cuda(), cate_list

    def find_index(self, sentence):
        indexs = []
        for i, s in enumerate(sentence):
            if s in self.seed_set:
                indexs.append(i)
        return indexs

    def assign_aspect(self, sentence):
        data = self.tokenizer(sentence, add_special_tokens=False, return_tensors='pt', truncation=True, is_split_into_words=False).data
        sentence_token_ids, token_type_ids, attention_mask = data['input_ids'], data["token_type_ids"], data["attention_mask"]
        sentence_new = [self.tokenizer.convert_ids_to_tokens(i) for i in sentence_token_ids]
        indexs = self.find_index(sentence_new[0])
        dists = self.mlm_model(sentence_token_ids.cuda(), attention_mask.cuda()).logits
        dists = dists[:,indexs,:]
        aspect_seq, seed_length, cate_list = self.aspect_seed_sequences
        aspect_dists = torch.softmax(torch.mm(aspect_seq, dists.sum(dim=1).permute(1, 0)) / seed_length.unsqueeze(-1), 0)
        aspect_index = torch.argmax(aspect_dists).item()
        return cate_list[aspect_index]
