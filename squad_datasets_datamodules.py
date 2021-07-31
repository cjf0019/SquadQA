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
    def __init__(self, df, tokenizer, max_input_len=512, max_question_len=100, max_label_len=50,
                 separate_context_question=False, negative_pads=False):
        self.df = df
        self.tokenizer = tokenizer
        self.max_input_len = max_input_len
        self.max_label_len = max_label_len
        self.max_question_len = max_question_len
        self.separate_context_question = separate_context_question
        self.negative_pads = negative_pads

    def __getitem__(self, idx):
        example_row = self.df.iloc[idx]
        ex_row_index = example_row.name

        if not self.separate_context_question:
            example = tokenize_to_dict(self.tokenizer, example_row['question']+example_row['context'],
                                       self.max_input_len, make_pad_negative=self.negative_pads)

        else:
            ### Tokenize the questions and contexts... separately
            example = tokenize_to_dict(self.tokenizer, example_row['question'], self.max_question_len,
                                       text_label='question', make_pad_negative=self.negative_pads)
            input_ids = tokenize_to_dict(self.tokenizer, example_row['context'], self.max_input_len,
                                         text_label='context', make_pad_negative=self.negative_pads)
            example.update(input_ids)

        ### Tokenize the answers
        answer = tokenize_to_dict(self.tokenizer, example_row['answer'], self.max_label_len)
        example.update(answer)

        example = {k: v.flatten() for k, v in example.items()}
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
            max_input_len=512,
            max_label_len=50,
            batch_size=64
            ):
        super().__init__()
        self.batch_size=batch_size
        self.tokenizer = tokenizer
        self.train_df = train_df
        self.test_df = test_df
        #self.test_df['Ans_Predict'] = ''
        
        self.max_input_len = max_input_len
        self.max_label_len = max_label_len
        
    def setup(self):
        self.train_dataset = SquadDataset(
            self.train_df,
            self.tokenizer,
            self.max_input_len,
            self.max_label_len)
        
        self.test_dataset = SquadDataset(
            self.test_df,
            self.tokenizer,
            self.max_input_len,
            self.max_label_len)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=4
            )

    def val_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=64,
            shuffle=False,
            num_workers=4
            )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=64,
            shuffle=False,
            num_workers=1
            )


