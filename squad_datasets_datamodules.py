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
from squad_utilities import tokenize_to_dict


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


