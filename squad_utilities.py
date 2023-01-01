#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 16 15:40:38 2021

@author: newsjunkie345
"""

import pandas as pd
import json
from pathlib import Path
import torch
from itertools import chain
import spacy
import numpy as np
from joblib import Parallel, delayed
from collections import defaultdict
import gensim
import gensim.downloader as api


EMBED_DIR = "C:\\Users\\cfavr\\gensim-data\\"
EMBED_FILE = "glove-wiki-gigaword-300"
SPACY_MODEL = 'en_core_web_md'


def chunker(iterable, total_length, chunksize):
    return (iterable[pos: pos + chunksize] for pos in range(0, total_length, chunksize))

def flatten(list_of_lists):
    "Flatten a list of lists to a combined list"
    return [item for sublist in list_of_lists for item in sublist]

class SpacyTokenizer(object):
    def __init__(self, nlp_file=SPACY_MODEL, sentence_separation=False):
        self.nlp_file = nlp_file
        self.sentence_separation = sentence_separation

        # mapping of nlp pipeline functions to words... will run each document separately
        self.pipeline_tasks = {'num_words': self.calculate_num_words_doc,
                               'num_sentences': self.calculate_num_sentences_doc,
                               #'local_word_count': self.local_word_count,
                               'global_word_count': self.add_to_global_word_count,
                               'global_embed_mean': self.add_to_global_mean_calc,
                               'global_embed_covariance': self.add_to_global_covariance_calc
                               }  # could combine with local_word_count so don't do twice if returning both
        self.summary_statistics = {}
        self.load_spacy_model()

    def tokenize(self, text, truncation=True, max_length=512,  sentence_separation='default', max_sentences=None,
                 padding='max_length', add_special_tokens=False, return_tensors='pt'):
        doc = self.nlp_model(text)
        sentence_separation = self.sentence_separation if sentence_separation == 'default' else sentence_separation

        if sentence_separation:
            input_ids = []
            sent_count = 0
            for sent in doc.sents:
                if max_sentences is not None and sent_count >= max_sentences:
                    continue
                toked = [self.token2id[str(tok)] for tok in sent if tok.has_vector and str(tok) in self.token2id.keys()]
                if truncation:
                    toked = toked[:max_length]
                toked += [self.padding_value] * max((max_length - len(toked)), 0)
                input_ids.append(toked)
                sent_count += 1

        else:
            input_ids = [self.token2id[str(tok)] for tok in doc if tok.has_vector and str(tok) in self.token2id.keys()]
            if truncation:
                input_ids = input_ids[:max_length]
            input_ids += [self.padding_value] * max((max_length - len(input_ids)), 0)
        if return_tensors == 'pt':
            return {'input_ids': torch.tensor(input_ids, dtype=torch.int64)}
        else:
            return {'input_ids': input_ids}

    def decode(self, ids):
        return ' '.join([self.id2token[int(id.detach().numpy())] for id in ids if int(id.detach().numpy()) != self.padding_value])

    def __call__(self, text, truncation=True, max_length=512, sentence_separation='default', max_sentences=None,
                 padding='max_length', add_special_tokens=False, return_tensors='pt'):
        return self.tokenize(text, max_length=max_length, sentence_separation=sentence_separation, max_sentences=max_sentences,
                             padding=padding, add_special_tokens=add_special_tokens, return_tensors=return_tensors)

    def __getitem__(self,idx):
        if idx == self.padding_value:
            return '<PAD>'
        return self.id2token[idx]

    def __dict__(self,tok):
        return self.token2id[tok]

    def load_spacy_model(self,embed_file=None):
        if embed_file is None:
            embed_file = self.nlp_file
        else:
            self.nlp_file = embed_file

        self.nlp_model = spacy.load(embed_file)
        #if self.sentence_separation:
        #    self.nlp_model.add_pipe('sentencizer')

        self.weights = torch.FloatTensor(self.nlp_model.vocab.vectors.data)

        # Add in an additional row at the end of the weights matrix, for compatibility issues with torch.Embedding
        self.weights = torch.cat((self.weights,torch.tensor(np.zeros((1,self.weights.shape[1])))))
        self.token2hash = self.nlp_model.vocab.strings
        self.token2id = {i: self.nlp_model.vocab.vectors.key2row[self.token2hash[i]] for i in list(self.nlp_model.vocab.strings)
                        if self.token2hash[i] in self.nlp_model.vocab.vectors.key2row.keys()}
        self.id2token = {self.nlp_model.vocab.vectors.key2row[self.token2hash[i]]: i for i in list(self.nlp_model.vocab.strings)
                        if self.token2hash[i] in self.nlp_model.vocab.vectors.key2row.keys()}

        self.raw_vocab_size = len(self.token2id) #all possible tokens, regardless of if there's an embedding
        self.vocab_size = max(self.token2id.values()) + 1
        self.padding_value = self.vocab_size #set padding to one more than the final index
        self.vocab_size += 1 # add the padding value to the vocab size

    ##### WORD EMBEDDING FUNCTIONS #####
    def get_weight(self,tok):
        return self.weights[self.token2id[tok]]

    @staticmethod
    def _norm_weight(vec):
        return np.sqrt(sum(x * x for x in vec))

    def cosine_similarity(self, word1, word2, normalize=True):
        vec1 = self.get_weight(word1)
        vec2 = self.get_weight(word2)
        mult = np.dot(vec1,vec2)
        norm = 1.0/(self._norm_weight(vec1)*self._norm_weight(vec2)) if normalize else 1.0
        return mult*norm

    def top_similar(self,word,n=10):
        topidx = np.dot(self.weights[self.token2id[word]],self.weights.T).argsort()[-1*n-1:-1][::-1]
        return [self.id2token[i] for i in topidx]

    ######### BATCH PROCESSING FUNCTIONS ###########
    def preprocess_pipeline(self, texts, pipeline_tasks=['num_words','num_sentences',
                                                         'global_embed_mean','global_embed_covariance'], batch_size=50):
        """
        NEEDS MAJOR REWORKING...
        1) Don't have it store docs... this was only done for calculating the covariance after the fact... instead
        use a rolling covariance estimate
        2) Expand out the pipeline tasks to incorporate entity extraction, if desired

        """
        preproc_pipe = {k: [] for k in pipeline_tasks if 'global' not in k}
        if isinstance(texts,str):
            texts = [texts]

        #with self.nlp_model.select_pipes(enable='sentencizer'):
        store_docs = []   # needed for the covariance calculation, to prevent having to run the nlp pipeline twice
        with self.nlp_model.select_pipes(enable=['tok2vec', 'parser']):
            for doc in self.nlp_model.pipe(texts, batch_size=batch_size):
                for task in preproc_pipe:
                    preproc_pipe[task].append(self.pipeline_tasks[task](doc))

                # any summary statistics designated by 'global'
                for summ_stat in list(filter(lambda x: 'global' in x and 'covariance' not in x, pipeline_tasks)):
                    self.pipeline_tasks[summ_stat](doc)

                store_docs.append(doc)

            if 'global_embed_mean' in pipeline_tasks:
                self.calc_global_embed_mean()
            if 'global_embed_covariance' in pipeline_tasks:
                for doc in store_docs:  # run through the docs again for covariance, now that the mean is present
                    self.pipeline_tasks['global_embed_covariance'](doc)
                self.calc_global_embed_covariance()
        return preproc_pipe

    def num_sentences_parallel(self, texts, chunksize=30, njobs=6):
        executor = Parallel(n_jobs=njobs, backend='multiprocessing', prefer="processes")
        do = delayed(self.calculate_num_sentences)
        tasks = (do(chunk) for chunk in chunker(texts, len(texts), chunksize=chunksize))
        result = executor(tasks)
        return flatten(result)

    ######## spaCy TEXT PROCESSING FUNCTIONS #########
    def calculate_num_sentences_doc(self, doc):
        if not isinstance(doc, spacy.tokens.doc.Doc):
            doc = self.nlp_model(doc)
        return len([i for i in doc.sents])

    def calculate_num_words_doc(self, doc):
        if not isinstance(doc, spacy.tokens.doc.Doc):
            doc = self.nlp_model(doc)
        return len(doc)

    def add_to_global_word_count(self, doc):
        if 'global_word_count' not in self.summary_statistics:
            self.summary_statistics['global_word_count'] = defaultdict(lambda: 0)
        for tok in doc:
            self.summary_statistics['global_word_count'][tok] += 1
        return

    def add_to_global_mean_calc(self, doc):
        if 'global_embed_sum' not in self.summary_statistics:
            self.summary_statistics['global_embed_sum'] = torch.zeros(self.weights.shape[-1])
        if 'global_word_count_total' not in self.summary_statistics:
            self.summary_statistics['global_word_count_total'] = 0

        word_ct = len(doc)
        self.summary_statistics['global_word_count_total'] += word_ct
        self.summary_statistics['global_embed_sum'] += doc.vector * word_ct

    def add_to_global_mean_calc_TOKENFORLOOP(self, doc):
        if 'global_embed_sum' not in self.summary_statistics:
            self.summary_statistics['global_embed_sum'] = torch.zeros(self.weights.shape[-1])

        if 'global_word_count_total' not in self.summary_statistics:
            self.summary_statistics['global_word_count_total'] = 0

        for tok in doc:
            #self.summary_statistics['global_word_count'][tok] += 1
            self.summary_statistics['global_word_count_total'] += 1
            self.summary_statistics['global_embed_sum'] += tok.vector

    def add_to_global_covariance_calc(self, doc):
        if 'global_embed_mean' not in self.summary_statistics:
            raise Exception('Embedding mean not calculated ahead of the covariance matrix calculation!')
        embed_mean = self.summary_statistics['global_embed_mean']
        if 'global_embed_cov_sum' not in self.summary_statistics:
            self.summary_statistics['global_embed_cov_sum'] = torch.zeros((self.weights.shape[-1],self.weights.shape[-1]))
        for tok in doc:
            mean_diff = torch.tensor(tok.vector) - embed_mean
            cov_mat = torch.outer(mean_diff,mean_diff)
            self.summary_statistics['global_embed_cov_sum'] += cov_mat

    def calc_global_embed_mean(self):
        self.summary_statistics['global_embed_mean'] = self.summary_statistics['global_embed_sum']/self.summary_statistics['global_word_count_total']

    def calc_global_embed_covariance(self):
        self.summary_statistics['global_embed_covariance'] = self.summary_statistics['global_embed_cov_sum']/(self.summary_statistics['global_word_count_total']-1)


class DocProcessor(object):
    """
    Inputs spaCy "Doc" objects and runs various nlp-related functions, such as getting tokens, entities, counts
    """
    def __init__(self, nlp, return_values=['tokens','ents','tokens_excl_ents','tokens_replace_ents', 'ent_types',
                'ent_count','token_count'], output_format='text', ent_dictionary=None):
        self.nlp = nlp
        self.return_values = [return_values] if isinstance(return_values,str) else return_values
        self.func_map = {'tokens': self.get_tokens,
                         'ents': self.get_ents,
                         'tokens_excl_ents': self.get_tokens_excl_ents,
                         'tokens_replace_ents': self.replace_ents_by_types,
                         'ent_types': self.get_ent_types,
                         'ent_count': self.ent_count,
                         'token_count': self.token_count}

        self.output_format = output_format
        self.transform_values = ['tokens', 'tokens_excl_ents', 'tokens_replace_ents', 'ents']

        # set up spacy-specific stuff
        self.retrieve_spacy_model_parameters()
        self.ent_dictionary = ent_dictionary
        self.ent_set = set(ent_dictionary.token2id)

    def __call__(self, doc):
        return_values = self.return_values
        if isinstance(doc,str):
            doc = self.nlp(doc)

        results = {}
        if 'tokens' in return_values and self.output_format == 'vector_mean':
            results['tokens'] = doc.vector
            return_values = [v for v in return_values if v!='tokens']

        if 'ents' in return_values and 'tokens_replace_ents' in return_values:
            toks, ents = self.replace_ents_by_types(doc)
            results['ents'] = ents
            results['tokens_replace_ents'] = toks
            return_values = [v for v in return_values if v not in ['ents','tokens_replace_ents']]
            results['ents_text'] = ' '.join(results['ents'])

        for value in return_values:
            results[value] = self.func_map[value](doc)

        for value in [v for v in results if v in self.transform_values and v != 'tokens']:
            results[value] = self.transform_feats(results[value], value_type=value)
        return results

    ### OUTPUT TRANSFORMATION FUNCTIONS ###
    def transform_feats(self, feats, value_type=None):
        if value_type=='ents':
            int_toks = set(self.ent_dictionary.token2id.keys())
            if value_type == 'ents':  # for entities, return as bag of words, the dimension as the number of entities in the corpus
                int_list = [self.ent_dictionary.token2id[tok] for tok in feats if tok in int_toks]
                return int_list

        ner_replace = {'NORP':'political', 'work_of_art':'artwork', 'GPE': 'geopolitical'}
        output = [ner_replace[i] if i in ner_replace else i for i in feats]

        if self.output_format == 'text':
            return output
        elif self.output_format == 'text-join':
            return ' '.join(output)

        #int_toks = set(self.ent_dictionary.token2id.keys()) if value_type=='ents' else set(self.token2id.keys())
        #if value_type == 'ents':   # for entities, return as bag of words, the dimension as the number of entities in the corpus
        #    int_list = [self.ent_dictionary.token2id[tok] for tok in output if tok in int_toks]
        #    return int_list
            #return self.convert_ints_to_bag_of_words(int_list)

        if self.output_format in ['integer','bag_of_words']:
            int_list = [self.token2id[tok] for tok in output if tok in self.token2id]
            if self.output_format == 'bag_of_words':
                return self.convert_ints_to_bag_of_words(int_list)
            else:
                return int_list
        elif self.output_format in ['vector', 'vector_mean']:
            vectors = [self.weights[self.token2id[tok]] for tok in output if tok in self.token2id]
            if self.output_format == 'vector_mean':
                return np.vstack(vectors).mean(axis=0)

    def convert_ints_to_bag_of_words(self, int_list):
        bow = np.zeros(self.vocab_size)
        for int in int_list:
            bow[int] += 1
        return bow

    def get_tokens(self, doc):
        return [tok.lemma_ for tok in doc]

    def replace_ents_by_types(self, doc, include_ents=True):
        ents = [ent for ent in doc.ents if ent.lemma_ in self.ent_set]
        next_ent = None if len(ents) == 0 else ents.pop(0)
        return_ents = []
        return_toks = []
        for tok in doc:
            if next_ent is not None:
                if tok.i == next_ent.start:
                    ent_text = next_ent.lemma_
                    return_toks.append(tok.ent_type_)
                    if ent_text not in return_ents:
                        return_ents.append(ent_text)
                    continue
                elif next_ent.start < tok.i < next_ent.end:
                    if tok.i == next_ent.end - 1:
                        next_ent = None if len(ents) == 0 else ents.pop(0)
                    continue
            return_toks.append(tok.lemma_)
        if include_ents:
            return return_toks, return_ents

    def get_ents(self, doc):
        return [ent.lemma_ for ent in doc.ents if ent.lemma_ in self.ent_set]

    def get_ent_types(self, doc):
        return [doc[ent.start].ent_type_ for ent in doc.ents if ent in self.ent_set]

    def get_tokens_excl_ents(self, doc, include_ents=False):
        #rm_ents = list(filter(lambda x: x.ent_type==0, doc))
        ents = self.get_ents(doc)
        #rm_ents = list(filter(lambda x: x not in set(self.ent_dictionary.token2id), doc))
        rm_ents = [tok.lemma_ for tok in doc if tok.lemma_ not in ents]
        if include_ents:
            return (rm_ents, ents)
        else:
            return rm_ents

    def ent_count(self, doc):
        return len(doc.ents)

    def token_count(self, doc):
        return len(doc)

    def retrieve_spacy_model_parameters(self):
        self.weights = torch.FloatTensor(self.nlp.vocab.vectors.data)

        # Add in an additional row at the end of the weights matrix, for compatibility issues with torch.Embedding
        self.weights = torch.cat((self.weights, torch.tensor(np.zeros((1, self.weights.shape[1])))))
        self.token2hash = self.nlp.vocab.strings
        self.token2id = {i: self.nlp.vocab.vectors.key2row[self.token2hash[i]] for i in list(self.nlp.vocab.strings)
                         if self.token2hash[i] in self.nlp.vocab.vectors.key2row.keys()}
        self.id2token = {self.nlp.vocab.vectors.key2row[self.token2hash[i]]: i for i in list(self.nlp.vocab.strings)
                         if self.token2hash[i] in self.nlp.vocab.vectors.key2row.keys()}

        self.raw_vocab_size = len(self.token2id)  # all possible tokens, regardless of if there's an embedding
        self.vocab_size = max(self.token2id.values()) + 1
        self.padding_value = self.vocab_size  # set padding to one more than the final index
        self.vocab_size += 1  # add the padding value to the vocab size


class GensimTokenizer(object):
    def __init__(self,embed_file=EMBED_FILE):
        self.embed_file = embed_file
        self.load_gensim_model()

    def tokenize(self, text, max_length=512, padding='max_length', add_special_tokens=False, return_tensors='pt'):
        toked = list(gensim.utils.tokenize(text))
        toked = [self.token2id[w] for w in toked]
        toked += [0]*(max_length-len(toked))
        if return_tensors=='pt':
            return {'input_ids': torch.tensor(toked,dtype=torch.int64)}
        else:
            return {'input_ids': toked}

    def decode(self, ids):
        return ' '.join([self.id2token[id] for id in ids])

    def __call__(self, text, max_length=512, padding='max_length', add_special_tokens=False, return_tensors='pt'):
        return self.tokenize(text, max_length=max_length, padding=padding,
                             add_special_tokens=add_special_tokens, return_tensors=return_tensors)

    def load_gensim_model(self,embed_file=None):
        if embed_file is None:
            embed_file = self.embed_file
        self.embed_model = api.load(embed_file)
        self.weights = torch.FloatTensor(self.embed_model.wv.vectors)
        self.vocabulary = self.embed_model.vocab
        self.token2id = {w: self.vocabulary[w].index for w in self.vocabulary.keys()}
        self.id2token = {self.vocabulary[w].index: w for w in self.vocabulary.keys()}
        self.vocab_size = len(self.embed_model.vocab)


def get_qas_ids_from_file_dict(file_dict):
    return list(chain(*[chain(*[[j['id'] for j in chain(i['qas'])] \
                for i in k['paragraphs']]) for k in file_dict['data']]))


def read_squad(path,include_no_answers=True):
    path = Path(path)
    with open(path, 'rb') as f:
        squad_dict = json.load(f)

    qas_ids = []
    contexts = []
    questions = []
    answers = []
    for group in squad_dict['data']:
        for passage in group['paragraphs']:
            context = passage['context']
            for qa in passage['qas']:
                qas_id = qa['id']
                question = qa['question']
                if include_no_answers and len(qa['answers']) == 0:
                    qas_ids.append(qas_id)
                    contexts.append(context)
                    questions.append(question)
                    answers.append({'text':'','answer_start':-1})                    

                for answer in qa['answers']:
                    qas_ids.append(qas_id)
                    contexts.append(context)
                    questions.append(question)
                    answers.append(answer)

    return qas_ids, contexts, questions, answers


def add_end_idx(answers, contexts):
    """
    The answers only contain the raw 'text' and 'answer_start' position.
    Add the 'answer_end' position.
    NOTE : The positions are relative to CHARACTERS not WORDS.
    """
    for answer, context in zip(answers, contexts):
        gold_text = answer['text']
        start_idx = answer['answer_start']
        end_idx = start_idx + len(gold_text)

        # sometimes squad answers are off by a character or two – fix this
        if context[start_idx:end_idx] == gold_text:
            answer['answer_end'] = end_idx
        elif context[start_idx-1:end_idx-1] == gold_text:
            answer['answer_start'] = start_idx - 1
            answer['answer_end'] = end_idx - 1     # When the gold label is off by one character
        elif context[start_idx-2:end_idx-2] == gold_text:
            answer['answer_start'] = start_idx - 2
            answer['answer_end'] = end_idx - 2     # When the gold label is off by two characters



def get_max_ans_len(answers):
    maxlen = 0
    for ans in answers:
        length = len(ans['text'].split())
        if length > maxlen:
            maxlen = length
    return maxlen


def setup_qa_df(qas_ids,contexts,questions,answers):
    df = pd.DataFrame()
    df['qas_id'] = qas_ids
    df['context'] = contexts
    df['question'] = questions
    df['answer'] = [answer['text'] for answer in answers]
    df['answer_start'] = [answer['answer_start'] for answer in answers]
    df['answer_end'] = [answer['answer_end'] for answer in answers]
    df = df.drop_duplicates().reset_index(drop=True)
    return df


def grab_sample_from_dataloader(data_module,train_test='train'):
    if not 'train':
        dl = data_module.test_dataloader() # retrieve the test dataloader
    else:
        dl = data_module.train_dataloader() # retrieve the test dataloader    
    return next(iter(dl))


def setup_input_and_run_model(data_module,model,raw_sample=None):
    if raw_sample is None:
        example = grab_sample_from_dataloader(data_module)
    loss, logits = model(input_ids=example['input_ids'],attention_mask=example['attention_mask'],labels=example['labels'])
    return example, loss, logits


def convert_inttokens_to_text(int_tokens):
    if len(int_tokens.shape) == 2:
        text = ' '.join([str(i) for i in int_tokens[0].detach().numpy()])
        for sent in int_tokens[1:]:
            text += ' + ' + ' '.join([str(i) for i in sent.detach().numpy()])
    else:
        text = ' '.join([str(i) for i in int_tokens.detach().numpy()])
    return text


def tokenize_to_dict(tokenizer, text, output_len, sentence_separation='default', max_sentences=None,
                     text_label=None, make_pad_negative=False, return_tokens_as_str=False):
    text_label = '' if text_label is None else text_label
    encodings = tokenizer(text, truncation=True, max_length=output_len, sentence_separation=sentence_separation,
                          max_sentences=max_sentences, padding="max_length", add_special_tokens=True, return_tensors='pt')
    if make_pad_negative:
        input_ids = encodings['input_ids']
        input_ids[input_ids == 0] = -100
        encodings['input_ids'] = input_ids

    tok_return = encodings.pop('input_ids')
    if return_tokens_as_str:
        tok_return = convert_inttokens_to_text(tok_return)
    encodings[(text_label+'_input_ids').lstrip('_')] = tok_return
    if 'attention_mask' in encodings.keys():
        encodings[(text_label+'_attention_mask').lstrip('_')] = encodings.pop('attention_mask')
    return encodings


def generate_answers(inputs,seq_len,model,input_col='input_ids',output_col='Ans_Predict',tokenizer=None):
    if tokenizer is not None:      
         inputs = tokenizer(inputs,padding=True,return_tensors="pt")
         inputs = inputs['input_ids']

    generation_output = model.generate(torch.tensor(inputs),max_length=seq_len)
    answers = [tokenizer.decode(i) for i in generation_output.detach().numpy()]
    return answers


def generate_answers_df(df,seq_len,model,tokenizer,df_col='context'):
    inputs = list(df[df_col])
    answers = generate_answers(inputs,seq_len,model,tokenizer=tokenizer)
    return answers

def process_generated_answer(data_module,epoch=None,outfile='predict_file.json'):
    if epoch is None:
        ans_cols = [col for col in data_module.test_df if 'test_ans_pred' in col]
        max_epoch = max([int(col.split('_')[-1]) for col in ans_cols])
        epoch = max_epoch
    ans_col = 'test_ans_pred_epoch_'+str(epoch)
    data_module.test_df[ans_col] = data_module.test_df[ans_col]\
                                            .str.replace('<pad>','').str.replace('</s>','').str.strip()
    to_output = data_module.test_df[['qas_id',ans_col]].drop_duplicates()
    to_output.index = to_output['qas_id']
    #to_output.index = ["\""+i+"\"" for i in to_output.index]
    #to_output['pred_answers'] = "\"" + to_output['pred_answers'] + "\""
    to_output[ans_col].to_json(outfile)
    print("PREDICTION FILE {} GENERATED".format(outfile))
        

def squad_prediction_output(data_module,model,outfile='predict_file.json'):
    data_module.test_df['pred_answers'] = pd.Series(generate_answers_df(data_module.test_df,\
                                            50,model,data_module.tokenizer),\
                                            index=data_module.test_df.index)
    data_module.test_df['pred_answers'] = data_module.test_df['pred_answers']\
                                            .str.replace('<pad>','').str.replace('</s>','').str.strip()
    to_output = data_module.test_df[['qas_id','pred_answers']]
    to_output.index = to_output['qas_id']
    #to_output.index = ["\""+i+"\"" for i in to_output.index]
    #to_output['pred_answers'] = "\"" + to_output['pred_answers'] + "\""
    to_output['pred_answers'].drop_duplicates().to_json(outfile)
    print("PREDICTION FILE {} GENERATED".format(outfile))


def trained_to_pretrained_weight_check(model,chkptpath):
    # load a copy of the model from checkpiont
    from squad_models import SquadModel
    pretrained_model = SquadModel.load_from_checkpoint(chkptpath+'/best-checkpoint-v2.ckpt')
    pretrained_model.freeze()
    
    def get_softmax_weights_from_model(pl_model):
        for array in pl_model.model.parameters():
            softmaxlayer = array   #will ultimately return the last layer (softmax)
        return softmaxlayer.detach().numpy()
    
    trained = get_softmax_weights_from_model(model)
    pretrained = get_softmax_weights_from_model(pretrained_model)
    return trained, pretrained


def stage_squad_data_pytorch(train_file,test_file):
    train_ids, train_contexts, train_questions, train_answers = read_squad(train_file)
    val_ids, val_contexts, val_questions, val_answers = read_squad(test_file)

    add_end_idx(train_answers, train_contexts)
    add_end_idx(val_answers, val_contexts)

    maxlength = get_max_ans_len(train_answers)
    ### Max answer length is 43, so set to 50, and set Q-context to 512

    train_df = setup_qa_df(train_ids, train_contexts, train_questions, train_answers)
    val_df = setup_qa_df(val_ids, val_contexts, val_questions, val_answers)

    return train_df, val_df







