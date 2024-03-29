#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 16 15:38:19 2021

MIGHT WANT TO REWRITE THE TOKENIZATION/SETUP PART TO RUN PREPROCESS PIPELINE...
Right now it runs a for loop over each item in the NLP_Dataset class. This is a heavily insufficient use of spaCy.
Currently getting memory errors as well with running the NLP_Dataset .setup() function. Cannot do the whole train_df

--- in the loop, runs tokenize_to_dict separately for each

Memory issues... cannot run the full df. Yet, as written relies on the iter(), which is a generator, line at a time


12/31/22
Rewriting NLPDataset to incorporate the best of EmotionsDataset. EmotionsDataset is a little too stripped down on its
    own, but we want NLPDataset to incorporate the spaCy pipeline in the iter function, as EmotionsDataset does.

    Need to merge "DocProcessor" and SpacyTokenizer back together!!!!!!

@author: newsjunkie345
"""

import torch
import transformers
import pandas as pd
import numpy as np
import datasets
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader
from squad_utilities import concat_text_cols, tokenize_to_dict, SpacyTokenizer, SpacyDocProcessor
from itertools import chain
import spacy
import re
import csv
from gensim import corpora
import os

EMBED_FILE = "glove-wiki-gigaword-300"
SPACY_MODEL = 'en_core_web_md'


class EmotionsDataset(Dataset):
    def __init__(self, df, text_col='text', return_values=['tokens'], text_output_format='vector_mean',
                 active_pipes='all', pipeline_batch_size=500, token_dict_train=False, ent_dict_train=False,
                 ent_dictionary=None):
        self.nlp = spacy.load(SPACY_MODEL)
        self.df = df
        self.text_col = text_col
        self.active_pipes = active_pipes
        self.pipeline_batch_size = pipeline_batch_size
        self.return_values = return_values
        self.feat_cols = self.return_values
        self.text_output_format = text_output_format
        self.token_dict_train = token_dict_train
        if token_dict_train:
            self.dictionary = corpora.Dictionary()
        self.ent_dict_train = ent_dict_train
        if ent_dictionary is None and self.ent_dict_train:
            self.ent_dictionary = corpora.Dictionary()
        else:
            self.ent_dictionary = ent_dictionary
        self.values_to_process = [v for v in self.return_values if v not in self.df.columns] # when the value is already present, don't recalculate!
        self.doc_processor = DocProcessor(self.nlp, return_values=self.values_to_process, output_format=text_output_format,
                                          ent_dictionary=self.ent_dictionary)
        self.setup()

    def __len__(self):
        return len(self.df)

    def setup(self):
        pipe_results = pd.DataFrame(iter(self), index=self.df.index)
        for col in pipe_results.columns:
            self.df[col] = pipe_results[col]
        return

    def __getitem__(self, idx):
        """
        For training, set return values to
        """
        row = self.df.iloc[idx]
        labels = row.loc[[f'emotion_{i}' for i in range(9)]].values.astype(float)
        return_vals = {feat_col: row.loc[feat_col] for feat_col in self.feat_cols}
        if 'ents' in return_vals:
            return_vals['ents'] = self.convert_ints_to_bag_of_words(return_vals['ents'], len(self.ent_dictionary))
        return_vals.update({'labels': labels})
        return return_vals
        #return {'inputs': feats, 'labels': labels}

    @staticmethod
    def convert_ints_to_bag_of_words(int_list, n_classes):
        bow = np.zeros(n_classes)
        for int in int_list:
            bow[int] += 1
        return bow

    def change_pipeline(self, active_pipes=None, return_values=None, ent_dictionary=None):
        if active_pipes is not None:
            self.active_pipes = active_pipes
        if return_values is not None:
            self.return_values = return_values
        if ent_dictionary is not None:
            self.ent_dictionary = ent_dictionary
        self.doc_processor = DocProcessor(self.nlp, return_values=self.return_values, ent_dictionary=ent_dictionary)

    def __iter__(self):
        #for index, line in pd.DataFrame(self.df[self.text_col]).iterrows():
        if self.active_pipes != 'all':
            with self.nlp.select_pipes(enable=self.active_pipes):  # if wanted basic tokenization/sentencization, set active_pipes to ['tok2vec','parser']
                for doc in self.nlp.pipe(self.df[self.text_col], batch_size=self.pipeline_batch_size):
                    return_vals = self.doc_processor(doc)
                    if self.ent_dict_train and 'ents' in return_vals:
                        self.ent_dictionary.add_documents(return_vals['ents'])
                    if self.token_dict_train and 'tokens' in return_vals:
                        self.dictionary.add_documents(return_vals['tokens'])
                    yield return_vals

        else:
            for doc in self.nlp.pipe(self.df[self.text_col], batch_size=self.pipeline_batch_size):
                return_vals = self.doc_processor(doc)
                if self.ent_dict_train and 'ents' in return_vals:
                    self.ent_dictionary.add_documents([return_vals['ents']])
                if self.token_dict_train and 'tokens' in return_vals:
                    self.dictionary.add_documents([return_vals['tokens']])
                yield return_vals


class NLPDataset(Dataset):
    def __init__(self, df, tokenizer, text_cols=None, sentence_len=128, negative_pads=False,
                 sentence_separation=True, aggregate_by='row', separate_cols_on_return=True,
                 return_values=['tokens'], text_output_format='vector_mean',
                 active_pipes='all', pipeline_batch_size=500, token_dict_train=False, ent_dict_train=False,
                 ent_dictionary=None, processed_file='processed_squad_train.csv'):
        """
        Run through a dataframe with one or multiple columns of text. For each included column, return the tokenized text
        as a separate item.

        Inputs...
        df : the dataframe with text samples
        tokenizer : the tokenizer to use. Typically spaCy.
            NOTE ... will check the tokenizer for "sentence_separation", a boolean expressing whether to break text apart
                into multiple sentences.
                -If multiple sentences, each item in the dataset will correspond to one unique sentence. Otherwise,
                     the entire block of text will be returned
        text_cols : the name of the df columns containing text. When indexing each dataset item, will maintain the order
                    provided by the text_cols list. Can optionally supply a list of tuples, each of which contains
                    a set of text columns to be aggregated together.
        sentence_len : the length of the sentence to return. If the number of tokens < sentence_len, will pad
        sentence_separation : if True, will split apart sentences, such that each document will be a list of sentenecs.
        aggregate_by : will determine what to consider as one "item" for the dataset.
            OPTIONS :
                Default: 'row'. If row, then will return all sentences of all text columns for the entire row together.
                'column'. If column, then will return all sentences of a text column.
                'sentence'. If sentence, will return each sentences of each text column separately.

        """

        self.df = df
        self.tokenizer = tokenizer
        if isinstance(text_cols, str):
            self.text_cols = [text_cols]
        else:
            self.text_cols = text_cols if text_cols is not None else []
        self.sentence_len = sentence_len
        self.negative_pads = negative_pads
        self.sentence_separation = sentence_separation
        self.tokenizer.sentence_separation = sentence_separation
        self.aggregate_by = aggregate_by
        self.separate_cols_on_return = separate_cols_on_return #only applicable if aggregate_by == 'row'... will return each text col separately instead of concatenating them
        self.run_tokenizer_on_output = True

        #if self.sentence_separation:
        #    self.count_sentences(self.text_cols)

        self.idx_starts = self.index_text_samples(text_cols)
        self.already_tokenized = False
        self.processed_file = processed_file

        self.return_values = return_values
        self.text_output_format = text_output_format
        self.active_pipes = active_pipes
        self.pipeline_batch_size = pipeline_batch_size
        self.token_dict_train = token_dict_train
        self.ent_dict_train = ent_dict_train
        self.ent_dictionary = ent_dictionary
        self.setup()

    def retrieve_text(self, idx):
        if self.aggregate_by == 'row':
            example_row = self.df.iloc[idx]
            if isinstance(self.text_cols,str):
                return {self.text_cols: example_row[self.text_cols]}

            if not self.separate_cols_on_return:
                text_cols = [tuple(chain(*[[col] if isinstance(col,str) else list(col) for col in self.text_cols]))]
            else:
                text_cols = self.text_cols

            text = {}
            for col in text_cols:
                if not isinstance(col,str):
                    colstr = '+'.join(col)
                    coltext = ' '.join([example_row[col] for col in col])
                else:
                    colstr = col
                    coltext = example_row[col]

                text.update({colstr: coltext})

        else:
            ### retrieve the idx's specific to column
            idx_startlist = list(self.idx_starts)
            text_col = idx_startlist[0]
            if len(idx_startlist) > 1:
                for col in idx_startlist[1:]:
                    if self.idx_starts[col] > idx:
                        break
                    else:
                        text_col = col

            if self.aggregate_by == 'column':
                idx -= self.idx_starts[text_col]
                text = ' '.join(self.df[text_col.split('+')].iloc[idx])

            else:
                df_row = self.df.query(f"{text_col}_IDX_Start <= {idx} < {len(self)}").iloc[-1]
                text = {text_col: df_row[text_col], 'df_row_idx': list(self.df.index).index(df_row.name)}
        return text

    def change_pipeline(self, active_pipes=None, return_values=None, ent_dictionary=None):
        if active_pipes is not None:
            self.active_pipes = active_pipes
        if return_values is not None:
            self.return_values = return_values
        if ent_dictionary is not None:
            self.ent_dictionary = ent_dictionary
        self.doc_processor = SpacyDocProcessor(self.nlp, return_values=self.return_values, ent_dictionary=ent_dictionary)

    def __getitem__(self, idx):
        if self.already_tokenized:
            return self.toked_df[['input_ids', 'row_id', 'sentence_id', 'text_field']].iloc[idx]

        else:
            ### First retrieve the text block relative to the index.
            example = self.retrieve_text(idx) # returns dict of text_col names to the text... concatenation of multiple cols also returned
            if not self.separate_cols_on_return:
                text_cols = [tuple(chain(*[[col] if isinstance(col,str) else list(col) for col in self.text_cols]))]
            else:
                text_cols = self.text_cols
            for text_col in {k:v for k,v in example.items() if k in text_cols}:
                example.update(self.doc_processor(text_col))

                ### If extracting by specific sentence
                if self.aggregate_by == 'sentence':
                    ##!!!!!! NEED TO FIND THE GLOBAL IDX FOR THE SENTENCE
                    label = list(example.keys())[0].replace('_input_ids','')
                    sents = example[text_col+'_input_ids'].split(' + ')
                    text_col_idx_start = self.idx_starts[label]
                    df_idx = example['df_row_idx']
                    idx_start = self.df.iloc[df_idx][label+'_IDX_Start']
                    sent_ind = int(idx - idx_start)
                    #example = {'input_ids': example[text+'_input_ids'][sent_ind], 'row_id': df_idx, 'sentence_id': idx, 'text_field': label}
                    example = {'input_ids': sents[sent_ind], 'row_id': df_idx, 'sentence_id': idx, 'text_field': label}
            return example

    """
    def __iter__(self):
        #for index, line in pd.DataFrame(self.df[self.text_col]).iterrows():
        if self.active_pipes != 'all':
            with self.nlp.select_pipes(enable=self.active_pipes):  # if wanted basic tokenization/sentencization, set active_pipes to ['tok2vec','parser']
                for doc in self.nlp.pipe(self.df[self.text_col], batch_size=self.pipeline_batch_size):
                    return_vals = self.doc_processor(doc)
                    if self.ent_dict_train and 'ents' in return_vals:
                        self.ent_dictionary.add_documents(return_vals['ents'])
                    if self.token_dict_train and 'tokens' in return_vals:
                        self.dictionary.add_documents(return_vals['tokens'])
                    yield return_vals

        else:
            for doc in self.nlp.pipe(self.df[self.text_col], batch_size=self.pipeline_batch_size):
                return_vals = self.doc_processor(doc)
                if self.ent_dict_train and 'ents' in return_vals:
                    self.ent_dictionary.add_documents([return_vals['ents']])
                if self.token_dict_train and 'tokens' in return_vals:
                    self.dictionary.add_documents([return_vals['tokens']])
                yield return_vals

    def __iter___OLD(self):
        for idx in range(len(self)):
            toked = self[idx]
            yield toked
    """

    def setup(self):
        #pipe_results = pd.DataFrame(iter(self))
        #self.toked_df = pipe_results
        if self.processed_file in os.listdir():
            self.already_tokenized = True
            self.toked_df = pd.read_csv(self.processed_file, delimiter='\t', quotechar='|', quoting=csv.QUOTE_MINIMAL)
            self.toked_df['input_ids'] = self.toked_df['input_ids'].transform(lambda x: x.split())
            return
        header = True
        with open(self.processed_file, 'w') as csvfile:
            writer_ = csv.writer(csvfile, delimiter='\t',
                                quotechar='|', quoting=csv.QUOTE_MINIMAL)
            for toked_data in iter(self.process_pipeline):  #!!! 1/3/23 add in the SpacyProcessPipeline to this class
                if header:
                    writer_.writerow(list(toked_data.keys()))
                    print(len(list(toked_data.values())))
                    header = False
                writer_.writerow(list(toked_data.values()))

        self.already_tokenized = True
        return


    def __len__(self):
        if self.aggregate_by == 'row':
            return len(self.df)
        else:
            last_col = self.text_cols[-1]
            max_idx_start = int(self.df[[last_col+'_IDX_Start']].max())
            if self.aggregate_by == 'column':
                return max_idx_start
            else:
                # count number of sentences in last text column
                return max_idx_start + int(self.df[[last_col+'_NumSentences']].iloc[-1])


    def index_text_samples(self, text_cols):
        # if we are using multiple text columns separately, or if we are returning individual sentences as samples,
        # we'll need to figure out where to start the dataset index for each text column
        if self.aggregate_by == 'column':
            # if each column is a separate sample, and not return sentences separately, just take len of dataframe for index spacing
            idx_starts = {k: text_cols.index(k)*len(self.df) for k in text_cols}
            return idx_starts

        else:  # for aggregating by sentences
            idx_starts = {}
            idx_start = 0

            # iterate through the text columns, calculate number of sentences based on sent tokenizer, generate idx's from sentence counts
            for col in text_cols:
                strcol = col if isinstance(col,str) else '+'.join(col) # for cases where a tuple or list of columns is supplied, such that their text will be grouped together
                print(f"... Indexing {strcol} ...")
                idx_starts[strcol] = idx_start
                if strcol+'_NumSentences' not in self.df.columns:
                    pipeline_results = self.preprocess_pipeline(self.df, col, tokenizer=self.tokenizer, pipeline_tasks=['num_sentences'])
                    #self.df = self.calculate_num_sentences_textcol(self.df, col, self.tokenizer)
                    self.df[strcol+'_NumSentences'] = pipeline_results['num_sentences']
                self.df[strcol + '_IDX_Start'] = self.df[strcol + '_NumSentences'].cumsum() + idx_start
                #else:
                #    self.df[col + '_IDX_Start'] = self.df[col].cum_sum() + idx_start
                next_idx_start = self.df[strcol + '_IDX_Start'].max()
                self.df[strcol+'_IDX_Start'] = self.df[strcol+'_IDX_Start'].shift().fillna(idx_start)
                idx_start = next_idx_start
        return idx_starts


    @staticmethod
    def preprocess_pipeline(df, text_col, tokenizer=None,
                            pipeline_tasks=['num_words', 'num_sentences'], batch_size=500):
        strcol = text_col if isinstance(text_col, str) else '+'.join(text_col)
        if tokenizer is not None:
            tokenizer_tasks = list(set(pipeline_tasks).intersection(set(tokenizer.pipeline_tasks)))
            pipe_results = tokenizer.preprocess_pipeline(NLPDataset.concat_text_cols(df, text_col),
                                                         pipeline_tasks=tokenizer_tasks, batch_size=batch_size)
            pipeline_tasks = list(set(pipeline_tasks) - set(tokenizer_tasks)) # if a tokenizer is missing specific tasks, process in the NLPDataset class

        if len(pipeline_tasks) > 0:  # for is the tokenizer does not have
            raise NotImplementedError("""Need to go back and add processing using self.calculate_num_sentences and self.calculate_num_words.
                The issue is I need a way to add in global_word_count as well.""")
        return pipe_results

    @staticmethod
    def calculate_num_sentences(texts):
        ### default function if a tokenizer without a method to calculate_num_sentences is used.
        return [len(set(filter(lambda x: x != '', re.split('[.!?]', text)))) for text in texts] # just count by splitting by punctuation

    @staticmethod
    def calculate_num_words(texts):
        ### default function if a tokenizer without a method to calculate_num_sentences is used.
        return [len(set(filter(lambda x: x != '', text.split()))) for text in texts]

    @staticmethod
    def calculate_num_sentences_textcol(df, text_col, tokenizer=None):
        """text_col can be either a column in the df with text, or a tuple or list of text columns.
            If a tuple or list, the number of sentences in the text concatenation will be returned """
        strcol = text_col if isinstance(text_col,str) else '+'.join(text_col)
        df[strcol+'_NumSentences'] = NLPDataset.calculate_num_sentences(NLPDataset.concat_text_cols(df,text_col), tokenizer)
        print(f"Number of sentences calculated for {strcol}: {df[strcol+'_NumSentences'].sum()}")
        return df

    @staticmethod
    def concat_text_cols(df, text_cols, delimiter=' '):
        return concat_text_cols(df, text_cols, delimiter)

    ### UNUSED ###
    def count_sentences(self, text_cols):
        #idx_start = 0
        #idx_starts = {}

        #if not self.sentence_separation and self.aggregate_by == 'row':
        #    idx_starts[text_cols[0]] = idx_start
        #    return idx_starts

        for col in text_cols:
            if col+'NumSentences' not in self.df.columns:
                self.df = self.calculate_num_sentences_df(self.df, self.tokenizer, col)
        return



class NLPDataset_DEPRECATED(Dataset):
    def __init__(self, df, tokenizer, text_cols=[], sentence_len=128, negative_pads=False,
                 sentence_separation=True, aggregate_by='row', separate_cols_on_return=True,
                 processed_file='processed_squad_train.csv'):
        """
        Run through a dataframe with one or multiple columns of text. For each included column, return the tokenized text
        as a separate item.

        Inputs...
        df : the dataframe with text samples
        tokenizer : the tokenizer to use. Typically spaCy.
            NOTE ... will check the tokenizer for "sentence_separation", a boolean expressing whether to break text apart
                into multiple sentences.
                -If multiple sentences, each item in the dataset will correspond to one unique sentence. Otherwise,
                     the entire block of text will be returned
        text_cols : the name of the df columns containing text. When indexing each dataset item, will maintain the order
                    provided by the text_cols list. Can optionally supply a list of tuples, each of which contains
                    a set of text columns to be aggregated together.
        sentence_len : the length of the sentence to return. If the number of tokens < sentence_len, will pad
        sentence_separation : if True, will split apart sentences, such that each document will be a list of sentenecs.
        aggregate_by : will determine what to consider as one "item" for the dataset.
            OPTIONS :
                Default: 'row'. If row, then will return all sentences of all text columns for the entire row together.
                'column'. If column, then will return all sentences of a text column.
                'sentence'. If sentence, will return each sentences of each text column separately.

        """

        self.df = df
        self.tokenizer = tokenizer
        self.text_cols = text_cols
        self.sentence_len = sentence_len
        self.negative_pads = negative_pads
        self.sentence_separation = sentence_separation
        self.tokenizer.sentence_separation = sentence_separation
        self.aggregate_by = aggregate_by
        self.separate_cols_on_return = separate_cols_on_return #only applicable if aggregate_by == 'row'... will return each text col separately instead of concatenating them
        self.run_tokenizer_on_output = True

        #if self.sentence_separation:
        #    self.count_sentences(self.text_cols)

        self.idx_starts = self.index_text_samples(text_cols)
        self.already_tokenized = False
        self.processed_file = processed_file
        self.setup()


    def retrieve_text(self, idx):
        if self.aggregate_by == 'row':
            example_row = self.df.iloc[idx]
            if isinstance(self.text_cols,str):
                return {self.text_cols: example_row[self.text_cols]}

            if not self.separate_cols_on_return:
                text_cols = [tuple(chain(*[[col] if isinstance(col,str) else list(col) for col in self.text_cols]))]
            else:
                text_cols = self.text_cols

            text = {}
            for col in self.text_cols:
                if not isinstance(col,str):
                    colstr = '+'.join(col)
                    coltext = ' '.join([example_row[col] for col in col])
                else:
                    colstr = col
                    coltext = example_row[col]

                text.update({colstr: coltext})

        else:
            ### retrieve the idx's specific to column
            for col in self.idx_starts:
                if self.idx_starts[col] > idx:
                    break
                else:
                    text_col = col

            if self.aggregate_by == 'column':
                idx -= self.idx_starts[text_col]
                text = ' '.join(self.df[text_col.split('+')].iloc[idx])

            else:
                df_row = self.df.query(f"{text_col}_IDX_Start <= {idx} < {len(self)}").iloc[-1]
                text = {text_col: df_row[text_col], 'df_row_idx': list(self.df.index).index(df_row.name)}
        return text


    def __getitem__(self, idx):
        if self.already_tokenized:
            return self.toked_df[['input_ids', 'row_id', 'sentence_id', 'text_field']].iloc[idx]

        else:
            ### First retrieve the text block relative to the index.
            example = self.retrieve_text(idx) # returns dict of text_col names to the text... concatenation of multiple cols also returned

            ### TOKENIZE
            if self.run_tokenizer_on_output:
                text_fields = [i for i in example.keys() if i in self.text_cols]
                for text in text_fields:
                    toked = tokenize_to_dict(self.tokenizer, example[text], self.sentence_len,
                                            sentence_separation=True, text_label=text,
                                             make_pad_negative=self.negative_pads, return_tokens_as_str=True)
                                            # return_tokens_as_str will output the result as space-delimited integer tokens

                    example.update(toked)

                ### If extracting by specific sentence
                if self.aggregate_by == 'sentence':
                    ##!!!!!! NEED TO FIND THE GLOBAL IDX FOR THE SENTENCE
                    label = list(example.keys())[0].replace('_input_ids','')
                    sents = example[text+'_input_ids'].split(' + ')
                    text_col_idx_start = self.idx_starts[label]
                    df_idx = example['df_row_idx']
                    idx_start = self.df.iloc[df_idx][label+'_IDX_Start']
                    sent_ind = int(idx - idx_start)
                    #example = {'input_ids': example[text+'_input_ids'][sent_ind], 'row_id': df_idx, 'sentence_id': idx, 'text_field': label}
                    example = {'input_ids': sents[sent_ind], 'row_id': df_idx, 'sentence_id': idx, 'text_field': label}
            return example

    def __iter__(self):
        for idx in range(len(self)):
            toked = self[idx]
            yield toked

    def setup(self):
        #pipe_results = pd.DataFrame(iter(self))
        #self.toked_df = pipe_results
        if self.processed_file in os.listdir():
            self.already_tokenized = True
            self.toked_df = pd.read_csv(self.processed_file, delimiter='\t', quotechar='|', quoting=csv.QUOTE_MINIMAL)
            self.toked_df['input_ids'] = self.toked_df['input_ids'].transform(lambda x: x.split())
            return
        header = True
        with open(self.processed_file, 'w') as csvfile:
            writer_ = csv.writer(csvfile, delimiter='\t',
                                quotechar='|', quoting=csv.QUOTE_MINIMAL)
            for toked_data in iter(self):
                if header:
                    writer_.writerow(list(toked_data.keys()))
                    print(len(list(toked_data.values())))
                    header = False
                writer_.writerow(list(toked_data.values()))

        self.already_tokenized = True
        return


    def __len__(self):
        if self.aggregate_by == 'row':
            return len(self.df)
        else:
            last_col = self.text_cols[-1]
            max_idx_start = int(self.df[[last_col+'_IDX_Start']].max())
            if self.aggregate_by == 'column':
                return max_idx_start
            else:
                # count number of sentences in last text column
                return max_idx_start + int(self.df[[last_col+'_NumSentences']].iloc[-1])


    def index_text_samples(self, text_cols):
        # if we are using multiple text columns separately, or if we are returning individual sentences as samples,
        # we'll need to figure out where to start the dataset index for each text column
        if self.aggregate_by == 'column':
            # if each column is a separate sample, and not return sentences separately, just take len of dataframe for index spacing
            idx_starts = {k: text_cols.index(k)*len(self.df) for k in text_cols}
            return idx_starts

        else:  # for aggregating by sentences
            idx_starts = {}
            idx_start = 0

            # iterate through the text columns, calculate number of sentences based on sent tokenizer, generate idx's from sentence counts
            for col in text_cols:
                strcol = col if isinstance(col,str) else '+'.join(col) # for cases where a tuple or list of columns is supplied, such that their text will be grouped together
                print(f"... Indexing {strcol} ...")
                idx_starts[strcol] = idx_start
                if strcol+'_NumSentences' not in self.df.columns:
                    pipeline_results = self.preprocess_pipeline(self.df, col, tokenizer=self.tokenizer, pipeline_tasks=['num_sentences'])
                    #self.df = self.calculate_num_sentences_textcol(self.df, col, self.tokenizer)
                    self.df[strcol+'_NumSentences'] = pipeline_results['num_sentences']
                self.df[strcol + '_IDX_Start'] = self.df[strcol + '_NumSentences'].cumsum() + idx_start
                #else:
                #    self.df[col + '_IDX_Start'] = self.df[col].cum_sum() + idx_start
                next_idx_start = self.df[strcol + '_IDX_Start'].max()
                self.df[strcol+'_IDX_Start'] = self.df[strcol+'_IDX_Start'].shift().fillna(idx_start)
                idx_start = next_idx_start
        return idx_starts


    @staticmethod
    def preprocess_pipeline(df, text_col, tokenizer=None,
                            pipeline_tasks=['num_words', 'num_sentences'], batch_size=500):
        strcol = text_col if isinstance(text_col, str) else '+'.join(text_col)
        if tokenizer is not None:
            tokenizer_tasks = list(set(pipeline_tasks).intersection(set(tokenizer.pipeline_tasks)))
            pipe_results = tokenizer.preprocess_pipeline(NLPDataset.concat_text_cols(df, text_col),
                                                         pipeline_tasks=tokenizer_tasks, batch_size=batch_size)
            pipeline_tasks = list(set(pipeline_tasks) - set(tokenizer_tasks)) # if a tokenizer is missing specific tasks, process in the NLPDataset class

        if len(pipeline_tasks) > 0:  # for is the tokenizer does not have
            raise NotImplementedError("""Need to go back and add processing using self.calculate_num_sentences and self.calculate_num_words.
                The issue is I need a way to add in global_word_count as well.""")
        return pipe_results

    @staticmethod
    def calculate_num_sentences(texts):
        ### default function if a tokenizer without a method to calculate_num_sentences is used.
        return [len(set(filter(lambda x: x != '', re.split('[.!?]', text)))) for text in texts] # just count by splitting by punctuation

    @staticmethod
    def calculate_num_words(texts):
        ### default function if a tokenizer without a method to calculate_num_sentences is used.
        return [len(set(filter(lambda x: x != '', text.split()))) for text in texts]

    @staticmethod
    def calculate_num_sentences_textcol(df, text_col, tokenizer=None):
        """text_col can be either a column in the df with text, or a tuple or list of text columns.
            If a tuple or list, the number of sentences in the text concatenation will be returned """
        strcol = text_col if isinstance(text_col,str) else '+'.join(text_col)
        df[strcol+'_NumSentences'] = NLPDataset.calculate_num_sentences(NLPDataset.concat_text_cols(df,text_col), tokenizer)
        print(f"Number of sentences calculated for {strcol}: {df[strcol+'_NumSentences'].sum()}")
        return df

    @staticmethod
    def concat_text_cols(df, text_cols, delimiter=' '):
        if isinstance(text_cols,str):
            return df[text_cols]
        elif len(text_cols) == 1:
            return df[text_cols[0]]
        else:
            return_col = df[text_cols[0]] + delimiter + df[text_cols[1]]
            for col in text_cols[2:]:
                return_col += df[col]
            return return_col

    ### UNUSED ###
    def count_sentences(self, text_cols):
        #idx_start = 0
        #idx_starts = {}

        #if not self.sentence_separation and self.aggregate_by == 'row':
        #    idx_starts[text_cols[0]] = idx_start
        #    return idx_starts

        for col in text_cols:
            if col+'NumSentences' not in self.df.columns:
                self.df = self.calculate_num_sentences_df(self.df, self.tokenizer, col)
        return


class SkipGramDataset(Dataset):
    def __init__(self, df, already_toked=False, args=None):
        self.args = args if args is not None else SkipGramArgs()
        self.__dict__.update(self.args.__dict__)
        self.df = df.copy()
        self.already_toked = already_toked  ### !!! NOT IMPLEMENTED: Assumed True at the moment

        ### Load in the codes, and collect frequency information/form a dictionary from the df
        self.codes = Codes(self.df, code_types=self.code_types, dx_dict=self.dx_dict, \
                           proc_dict=self.proc_dict, rm_bottom_n=self.rm_bottom_n, \
                           code_dict_file=self.code_dict_file, conn=self.conn)
        self.__dict__.update(self.codes.__dict__)

        ### Get a count of the DX's per row, as wells the cumulative count (for indexing inter-row)
        # self.df['Context_Count'] = min(self.df['DX_Count'] * 2 * self.args.window_size, \
        #                                  (self.df['DX_Count'] - 1) * self.df['DX_Count']) # Number of dx -context pairs contained in the row for the given window size

        self.df['Count_PreSubSample'] = self.df['Codes'].transform(len)
        self.df = self.df[self.df['Count_PreSubSample'] > 0]

        self.df['Context_Count_PreSubSample'] = (self.df['Count_PreSubSample'] * \
                                                 (self.df['Count_PreSubSample'] - 1)) / 2
        self.df['IDX_Range_PreSubSample'] = self.df['Context_Count_PreSubSample'].cumsum()

        print("Performing Sub-sampling")
        # Perform subsampling to reduce highly frequent codes
        self.df['Codes'] = self.df['Codes'].transform(lambda row: \
                                                          [code for code in row if self.subsample_frequent(code)])
        self.df['Word_Count'] = self.df['Codes'].transform(len)
        self.df = self.df[self.df['Word_Count'] > 0]

        self.df['Context_Count'] = (self.df['Word_Count'] * (self.df['Word_Count'] - 1)) / 2
        mean_contexts_lost = (self.df['Context_Count_PreSubSample'] - self.df['Context_Count']).mean()
        print("Average of {} contexts lost per record after Sub-sampling.". \
              format(str(np.round(mean_contexts_lost))))

        self.df['IDX_Range'] = self.df['Context_Count'].cumsum().astype(int)

        if self.shuffle_codes:
            print("Shuffling codes inside each row.")
            self.df['Codes'].transform(random.shuffle)

        self.generate_examples_serial()

    def __len__(self):
        return self.df.iloc[-1]['IDX_Range']

    def __getitem__(self, index):
        return self._example_to_tensor(*self.examples[index])

    def generate_examples_serial(self):
        """
        Generates examples with no multiprocessing - straight through!
        :return: None - updates class properties
        """
        self.examples = []
        for row in tqdm(self.df['Codes'],
                        desc="Generating Examples (serial)"):  ### If adding Proc codes, should make into iterrows, so can access both columns
            self.examples.extend(self._generate_examples_from_row(row))

    def subsample_frequent(self, code):
        f = self.dictionary.dfs[self.dictionary.token2id[code]] / self.freq_sum
        subsamp_freq = 1 - np.sqrt(0.001 / f)
        return subsamp_freq < random.random()

    def _generate_examples_from_row(self, row):
        """
        Generate all examples from a file within window size
        :param file: File from self.files
        :returns: List of examples
        """
        combs = list(combinations([self.dictionary.token2id[i] for i in row], 2))
        return combs

    def _example_to_tensor(self, center, target):
        """
        Takes raw example and turns it into tensor values
        :params example: Tuple of form: (center word, document id)
        :params target: String of the target word
        :returns: A tuple of tensors
        """
        center, target = torch.tensor([int(center)]), torch.tensor([int(target)])
        return center, target


class SquadDataset(Dataset):
    def __init__(self, df, tokenizer, max_input_len=[5,128], max_question_len=128, max_label_len=50,
                 separate_context_question=False, negative_pads=False):
        self.df = df
        self.tokenizer = tokenizer
        if isinstance(max_input_len,list):  # for contexts to be cast into individual sentences... first value is the number of sentences, second the sentence length
            self.max_input_len = max_input_len[1]
            self.max_sentences = max_input_len[0]
        else: # for single blocks of text, not split into sentence
            self.max_input_len = max_input_len
            self.max_sentences = None
        self.max_label_len = max_label_len
        self.max_question_len = max_question_len
        self.separate_context_question = separate_context_question
        self.negative_pads = negative_pads
        self.sentence_separation = tokenizer.sentence_separation

    def __getitem__(self, idx):
        example_row = self.df.iloc[idx]
        ex_row_index = example_row.name

        if not self.separate_context_question:
            example = tokenize_to_dict(self.tokenizer, example_row['question']+example_row['context'],
                                       self.max_input_len, max_sentences=self.max_sentences, make_pad_negative=self.negative_pads)
            context_label = 'input_ids'
        else:
            ### Tokenize the questions and contexts... separately
            example = tokenize_to_dict(self.tokenizer, example_row['question'], self.max_question_len,
                                       sentence_separation=False, text_label='question',
                                       make_pad_negative=self.negative_pads)
            input_ids = tokenize_to_dict(self.tokenizer, example_row['context'], self.max_input_len,
                                         max_sentences=self.max_sentences,
                                        text_label='context', make_pad_negative=self.negative_pads)
            example.update(input_ids)
            context_label = 'context_input_ids'

        # to ensure all data samples have same dimension, add sentences of only padding
        if self.sentence_separation and self.max_sentences is not None:
            doc_shape = example[context_label].shape
            if len(doc_shape) == 1:  # prevent flattening when only one sentence is present
                example[context_label] = example[context_label].reshape(1,-1)
                doc_shape = (1, doc_shape[0])
            print("EX ", example[context_label])
            pad_sentence_ct = self.max_sentences - doc_shape[0]
            if pad_sentence_ct > 0:
                pad_sentences = torch.empty((pad_sentence_ct, self.max_input_len))
                pad_sentences.fill_(self.tokenizer.padding_value)
                example[context_label] = torch.cat((example[context_label], pad_sentences))

        ### Tokenize the answers
        answer = tokenize_to_dict(self.tokenizer, example_row['answer'], self.max_label_len,
                                        sentence_separation=False,
                                        text_label='answer', make_pad_negative=self.negative_pads)
        example.update(answer)

        example = {k: v.squeeze() for k, v in example.items()}
        example.update({'qas_id': example_row['qas_id'], 'Row_ID': example_row.name})
        return example

    def __len__(self):
        return len(self.df)



class SquadDataModule(pl.LightningDataModule):
    def __init__(
            self,
            train_df,
            test_df,
            tokenizer,
            max_input_len=[5,128],
            max_question_len=50,
            max_label_len=50,
            separate_context_question=False,
            negative_pads=False,
            batch_size=64,
            num_workers=0
            ):
        super().__init__()
        self.batch_size=batch_size
        self.tokenizer = tokenizer
        self.train_df = train_df
        self.test_df = test_df
        #self.test_df['Ans_Predict'] = ''
        
        self.max_input_len = max_input_len
        self.max_question_len = max_question_len
        self.max_label_len = max_label_len
        self.separate_context_question = separate_context_question
        self.negative_pads = negative_pads
        self.num_workers = num_workers

    def setup(self):
        self.train_dataset = SquadDataset(
            self.train_df,
            self.tokenizer,
            self.max_input_len,
            self.max_label_len,
            separate_context_question = self.separate_context_question,
            negative_pads= self.negative_pads)
        
        self.test_dataset = SquadDataset(
            self.test_df,
            self.tokenizer,
            self.max_input_len,
            self.max_label_len,
            separate_context_question = self.separate_context_question,
            negative_pads = self.negative_pads)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers
            )

    def val_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=4,
            shuffle=False,
            num_workers=self.num_workers
            )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=4,
            shuffle=False,
            num_workers=self.num_workers
            )


class VAEDataModule_Sentence(pl.LightningDataModule):
    def __init__(
            self,
            train_df,
            test_df,
            tokenizer,
            batch_size=64,
            num_workers=0
    ):
        super().__init__()
        self.batch_size = batch_size
        self.tokenizer = tokenizer
        self.train_df = train_df
        self.test_df = test_df
        self.num_workers = num_workers

    def setup(self):
        self.train_dataset = NLPDataset(self.train_df, self.tokenizer, text_cols=['question', 'context', 'answer'], aggregate_by='sentence')
        self.test_dataset = NLPDataset(self.test_df, self.tokenizer, text_cols=['question', 'context', 'answer'], aggregate_by='sentence')

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers
        )

    def val_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=4,
            shuffle=False,
            num_workers=self.num_workers
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=4,
            shuffle=False,
            num_workers=self.num_workers
        )

