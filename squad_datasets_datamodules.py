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


class SquadDataset(Dataset):
    def __init__(self, df, tokenizer, max_input_len=512, max_label_len=50):
        self.df = df
        self.tokenizer = tokenizer
        self.max_input_len = max_input_len
        self.max_label_len = max_label_len
        

    def __getitem__(self, idx):
        example_row = self.df.iloc[idx] 
        ex_row_index = example_row.name
        ### Tokenize the questions and contexts... will concatenate them
        input_encodings = self.tokenizer(example_row['question'], \
                                    example_row['context'], \
                                    truncation=True, \
                                    max_length = self.max_input_len, \
                                    padding="max_length",\
                                    add_special_tokens = True,\
                                    return_tensors='pt')
        #val_encodings = tokenizer(val_contexts, val_questions, truncation=True, padding=True)
        
        ### Tokenize the answers
        answer_encodings = self.tokenizer(example_row['answer'], 
                                    truncation=True, \
                                    max_length = self.max_label_len, \
                                    padding="max_length",\
                                    add_special_tokens = True,\
                                    return_tensors='pt')
            
        ### T5 requires disregarded tokens to be < 0, so change pads to -100
        labels = answer_encodings['input_ids']
        labels[labels==0] = -100
        
        answer_encodings['labels'] = answer_encodings.pop('input_ids')
        answer_encodings['label_attention_mask'] = answer_encodings.pop('attention_mask')

        #print(input_encodings['input_ids'].size())
        example = input_encodings
        example.update(answer_encodings)
        example = {k: v.flatten() for k, v in example.items()}
        example.update({'qas_id':example_row['qas_id'],'Row_ID':example_row.name})
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


class SquadDataset_SeparateContextQuestion(Dataset):
    def __init__(self, df, tokenizer, max_input_len=512, max_question_len=100, max_label_len=50, separate_context_question=False):
        self.df = df
        self.tokenizer = tokenizer
        self.max_input_len = max_input_len
        self.max_label_len = max_label_len
        self.max_question_len = max_question_len
        self.separate_context_question = separate_context_question

    def __getitem__(self, idx):
        example_row = self.df.iloc[idx]
        ex_row_index = example_row.name

        if not self.separate_context_question:
            ### Tokenize the questions and contexts... will concatenate them
            input_encodings = self.tokenizer(example_row['question'], \
                                             example_row['context'], \
                                             truncation=True, \
                                             max_length=self.max_input_len, \
                                             padding="max_length", \
                                             add_special_tokens=True, \
                                             return_tensors='pt')

        else:
            ### Tokenize the questions and contexts... separately
            question_encodings = self.tokenizer(example_row['question'], \
                                             truncation=True, \
                                             max_length=self.max_question_len, \
                                             padding="max_length", \
                                             add_special_tokens=True, \
                                             return_tensors='pt')

            input_encodings = self.tokenizer(example_row['context'], \
                                             truncation=True, \
                                             max_length=self.max_input_len, \
                                             padding="max_length", \
                                             add_special_tokens=True, \
                                             return_tensors='pt')

        # val_encodings = tokenizer(val_contexts, val_questions, truncation=True, padding=True)

        ### Tokenize the answers
        answer_encodings = self.tokenizer(example_row['answer'],
                                          truncation=True, \
                                          max_length=self.max_label_len, \
                                          padding="max_length", \
                                          add_special_tokens=True, \
                                          return_tensors='pt')


        input_ids = input_encodings['input_ids']
        input_ids[input_ids == 0] = -100
        input_encodings['input_ids'] = input_ids

        example = input_encodings

        questions = question_encodings['input_ids']
        questions[questions == 0] = -100
        question_encodings['question'] = question_encodings.pop('input_ids')
        question_encodings['question_attention_mask'] = question_encodings.pop('attention_mask')
        example.update(question_encodings)

        ### T5 requires disregarded tokens to be < 0, so change pads to -100
        labels = answer_encodings['input_ids']
        labels[labels == 0] = -100
        answer_encodings['labels'] = answer_encodings.pop('input_ids')
        answer_encodings['label_attention_mask'] = answer_encodings.pop('attention_mask')
        example.update(answer_encodings)

        example = {k: v.flatten() for k, v in example.items()}
        example.update({'qas_id': example_row['qas_id'], 'Row_ID': example_row.name})
        return example

    def __len__(self):
        return len(self.df)



