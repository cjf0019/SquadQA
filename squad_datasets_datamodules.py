#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 16 15:38:19 2021

@author: newsjunkie345
"""

import torch
import transformers
import datasets
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader
from squad_utilities import tokenize_to_dict, SpacyTokenizer
from itertools import chain
import re

class NLPDataset(Dataset):
    def __init__(self, df, tokenizer, text_cols=[], sentence_len=128, negative_pads=False,
                 sentence_separation=True, aggregate_by='row', separate_cols_on_return=True):
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
                text = {text_col: self.df.query(f"{text_col}_IDX_Start < {idx}").iloc[0]}
        return text


    def __getitem__(self, idx):
        ### First retrieve the text block relative to the index.
        example = self.retrieve_text(idx) # returns dict of text_col names to the text... concatenation of multiple cols also returned

        ### TOKENIZE
        if self.run_tokenizer_on_output:
            text_fields = [i for i in example.keys()]
            for text in text_fields:
                example.update(tokenize_to_dict(self.tokenizer, example[text], self.sentence_len,
                                                 sentence_separation=True, text_label=text,
                                                 make_pad_negative=self.negative_pads))

            ### If extracting by specific sentence
            if self.aggregate_by == 'sentence':
                ##!!!!!! NEED TO FIND THE GLOBAL IDX FOR THE SENTENCE
                label = list(example.keys())[0]
                text_col_idx_start = self.idx_starts[label]
                df_idx = idx - text_col_idx_start
                idx_start = self.df.iloc[df_idx][list(example.keys())[0]+'_IDX_Start']
                sent_ind = idx - idx_start
                example = {text+'Sent_'+str(sent_ind): example[text][sent_ind]}

        return example


    def __len__(self):
        return len(self.df)


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
                if '_NumSentences' not in self.df.columns:
                    self.df = self.calculate_num_sentences_textcol(self.df, col, self.tokenizer)
                self.df[strcol + '_IDX_Start'] = self.df[strcol + '_NumSentences'].cumsum() + idx_start
                #else:
                #    self.df[col + '_IDX_Start'] = self.df[col].cum_sum() + idx_start
                next_idx_start = self.df[strcol + '_IDX_Start'].max()
                self.df[strcol+'_IDX_Start'] = self.df[strcol+'_IDX_Start'].shift().fillna(idx_start)
                idx_start = next_idx_start
        return idx_starts


    @staticmethod
    def calculate_num_sentences(text, tokenizer=None):
        if tokenizer is not None and isinstance(tokenizer,SpacyTokenizer):
            return tokenizer.calculate_num_sentences(text)
        else:
            return len(set(filter(lambda x: x != '', re.split('[.!?]', text))))  # if no tokenizer present, just count by splitting by a period

    @staticmethod
    def calculate_num_sentences_textcol(df, text_col, tokenizer=None):
        """text_col can be either a column in the df with text, or a tuple or list of text columns.
            If a tuple or list, the number of sentences in the text concatenation will be returned """
        strcol = text_col if isinstance(text_col,str) else '+'.join(text_col)
        df[strcol+'_NumSentences'] = NLPDataset.concat_text_cols(df,text_col).transform(lambda x:
                                                                        NLPDataset.calculate_num_sentences(x,tokenizer))
        print(f"Number of sentences calculated for {strcol}")
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



class NLPDataset_Deprecated(Dataset):
    def __init__(self, df, tokenizer, text_cols=[], sentence_len=128, negative_pads=False,
                 sentence_separation=True, aggregate_by='row', separate_cols_on_return=True):
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
                text = {text_col: self.df.query(f"{text_col}_IDX_Start < {idx}").iloc[0]}
        return text


    def __getitem__(self, idx):
        ### First retrieve the text block relative to the index.
        example = self.retrieve_text(idx) # returns dict of text_col names to the text... concatenation of multiple cols also returned

        ### TOKENIZE
        if self.run_tokenizer_on_output:
            text_fields = [i for i in example.keys()]
            for text in text_fields:
                example.update(tokenize_to_dict(self.tokenizer, example[text], self.sentence_len,
                                                 sentence_separation=True, text_label=text,
                                                 make_pad_negative=self.negative_pads))

            ### If extracting by specific sentence
            if self.aggregate_by == 'sentence':
                ##!!!!!! NEED TO FIND THE GLOBAL IDX FOR THE SENTENCE
                label = list(example.keys())[0]
                text_col_idx_start = self.idx_starts[label]
                df_idx = idx - text_col_idx_start
                idx_start = self.df.iloc[df_idx][list(example.keys())[0]+'_IDX_Start']
                sent_ind = idx - idx_start
                example = {text+'Sent_'+str(sent_ind): example[text][sent_ind]}

        return example


    def __len__(self):
        return len(self.df)

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
                if '_NumSentences' not in self.df.columns:
                    self.df = self.calculate_num_sentences_textcol(self.df, self.tokenizer, col)
                self.df[strcol + '_IDX_Start'] = self.df[strcol + '_NumSentences'].cumsum() + idx_start
                #else:
                #    self.df[col + '_IDX_Start'] = self.df[col].cum_sum() + idx_start
                next_idx_start = self.df[strcol + '_IDX_Start'].max()
                self.df[strcol+'_IDX_Start'] = self.df[strcol+'_IDX_Start'].shift().fillna(idx_start)
                idx_start = next_idx_start
        return idx_starts


    @staticmethod
    def calculate_num_sentences(text, tokenizer):
        return len([i for i in tokenizer.nlp_model(text).sents])

    @staticmethod
    def calculate_num_sentences_textcol(df, tokenizer, text_col):
        """text_col can be either a column in the df with text, or a tuple or list of text columns.
            If a tuple or list, the number of sentences in the text concatenation will be returned """
        strcol = text_col if isinstance(text_col,str) else '+'.join(text_col)
        df[strcol+'_NumSentences'] = NLPDataset.concat_text_cols(df,text_col).transform(lambda x:
                                                                        NLPDataset.calculate_num_sentences(x,tokenizer))
        print(f"Number of sentences calculated for {strcol}")
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