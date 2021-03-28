#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 27 16:21:00 2021

@author: Connor Favreau
"""


import os
import torch
import transformers
import datasets
import pytorch_lightning as pl

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"   # dont use GPU for now
input_dir = ".//"

import json
from pathlib import Path


def read_squad(path):
    path = Path(path)
    with open(path, 'rb') as f:
        squad_dict = json.load(f)

    contexts = []
    questions = []
    answers = []
    for group in squad_dict['data']:
        for passage in group['paragraphs']:
            context = passage['context']
            for qa in passage['qas']:
                question = qa['question']
                for answer in qa['answers']:
                    contexts.append(context)
                    questions.append(question)
                    answers.append(answer)

    return contexts, questions, answers


train_contexts, train_questions, train_answers = read_squad(input_dir+'squad/train-v2.0.json')
val_contexts, val_questions, val_answers = read_squad(input_dir+'squad/dev-v2.0.json')


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

add_end_idx(train_answers, train_contexts)
add_end_idx(val_answers, val_contexts)

def get_max_ans_len(answers):
    maxlen = 0
    for ans in answers:
        length = len(ans['text'].split())
        if length > maxlen:
            maxlen = length
    return maxlen

maxlength = get_max_ans_len(train_answers)
### Max answer length is 43, so set to 50, and set Q-context to 512


from transformers import T5Tokenizer, T5Model
MODEL_NAME = 't5-small'
tokenizer = T5Tokenizer.from_pretrained('t5-small')
#model = T5Model.from_pretrained('t5-small')


def setup_df(contexts,questions,answers):
    df = pd.DataFrame()
    df['context'] = contexts
    df['question'] = questions
    df['answer'] = [answer['text'] for answer in answers]
    df['answer_start'] = [answer['answer_start'] for answer in answers]
    df['answer_end'] = [answer['answer_end'] for answer in answers]
    return df
    
train_df = setup_df(train_contexts,train_questions,train_answers)
val_df = setup_df(val_contexts,val_questions,val_answers)


from torch.utils.data import DataLoader
from transformers import AdamW
    
class SquadDataset(torch.utils.data.Dataset):
    def __init__(self, df, max_input_len=512, max_label_len=50):
        self.df = df
        self.tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME)
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
        example.update({'Row_ID':ex_row_index})
        return example

    def __len__(self):
        return len(self.df)



class SquadDataset_DEPRECATED(torch.utils.data.Dataset):
    def __init__(self, encodings,labels, max_input_len=512, max_label_len=50):
        self.encodings = encodings
        self.labels = labels
        self.max_input_len = max_input_len
        self.max_label_len = max_label_len

    def __getitem__(self, idx):
        example = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        example.update({key: torch.tensor(val[idx]) for key, val in self.labels.items()})
        return example

    def __len__(self):
        return len(self.encodings.input_ids)



class SquadDataModule(pl.LightningDataModule):
    def __init__(
            self,
            train_df,
            test_df,
            max_input_len=512,
            max_label_len=50,
            batch_size=32
            ):
        super().__init__()
        self.batch_size=batch_size
        self.train_df = train_df
        self.test_df = test_df
        #self.test_df['Ans_Predict'] = ''
        
        self.max_input_len = max_input_len
        self.max_label_len = max_label_len
        
    def setup(self):
        self.train_dataset = SquadDataset(
            self.train_df,
            self.max_input_len,
            self.max_label_len)
        
        self.test_dataset = SquadDataset(
            self.test_df,
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
            batch_size=4,
            shuffle=False,
            num_workers=4           
            )
    
    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=4
            )


BATCH_SIZE = 32
N_EPOCHS = 3

data_module = SquadDataModule(
            train_df,\
            val_df,\
            batch_size = BATCH_SIZE
            )

data_module.setup()

from transformers import T5ForConditionalGeneration

#### ADD IN THE RETURN OF WHAT THE OUTPUTS ARE PREDICTING
#### ... TO COMPARE SENTENCE TO SENTENCE
output_agg = []
### Currently getting 4 different arrays of size (50, VOCAB_SiZE) that are all equal

       
def generate_answers(inputs,seq_len,model,input_col='input_ids',output_col='Ans_Predict',tokenize_input=False):
    if tokenize_input:      
         print("TEXT",inputs)
         inputs = tokenizer(inputs,padding=True,return_tensors="pt")
         inputs = inputs['input_ids']

    generation_output = model.generate(inputs,max_length=seq_len)
    answers = [tokenizer.decode(i) for i in generation_output.detach().numpy()]
    return answers


class SquadModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME,return_dict=True)

    def forward(self, input_ids, attention_mask, labels=None):
        output = self.model(
            input_ids = input_ids,
            attention_mask = attention_mask,
            labels = labels            
            )
        
        return output.loss, output.logits
    
    def training_step(self, batch, batch_idx):
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['labels']
        loss, outputs = self(input_ids, attention_mask, labels)
        self.log('train_loss', loss, prog_bar=True, logger=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        ans_pred_col = "val_ans_pred_epoch_{}".format(str(self.current_epoch))
        if ans_pred_col not in data_module.test_df.columns:
            data_module.test_df[ans_pred_col] = ''
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['labels']
        row_ids = batch['Row_ID'].detach().numpy()
        loss, outputs = self(input_ids, attention_mask, labels)
        
        self.log('val_loss', loss, prog_bar=True, logger=True)
        #self.generate_answers(input_ids,batch['Row_ID'].values,len(labels))
        data_module.test_df.loc[row_ids,ans_pred_col] = \
                            generate_answers(input_ids,\
                            len(labels),self.model,input_col='context',tokenize_input=False)
        
        return loss
    
    def test_step(self, batch, batch_idx):
        ans_pred_col = "test_ans_pred_epoch_{}".format(str(self.current_epoch))
        if ans_pred_col not in data_module.test_df.columns:
            data_module.test_df[ans_pred_col] = ''
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['labels']
        row_ids = batch['Row_ID'].detach().numpy()
        loss, outputs = self(input_ids, attention_mask, labels)
        self.log('test_loss', loss, prog_bar=True, logger=True)
        
        data_module.test_df.loc[row_ids,ans_pred_col] = \
                            generate_answers(input_ids,\
                            len(labels),self.model,input_col='context',tokenize_input=False)
        #self.generate_answers(input_ids,batch['Row_ID'].values,len(labels))

    def configure_optimizers(self):
        return AdamW(self.parameters(), lr=0.001)
   

model = SquadModel()

import datetime
CHKPTPATH = str(datetime.date.today())+'/checkpoints'

checkpoint_callback = pl.callbacks.ModelCheckpoint(
    dirpath=CHKPTPATH,
    filename='best-checkpoint',
    save_top_k=3,
    verbose=True,
    monitor='val_loss',
    mode='min'
    )

trainer = pl.Trainer(
    checkpoint_callback = checkpoint_callback,
    max_epochs = N_EPOCHS,
    logger=pl.loggers.TensorBoardLogger('logs/'),
    progress_bar_refresh_rate = 30,
    check_val_every_n_epoch=1
    )

trainer.fit(model,data_module)


def trained_to_pretrained_weight_check(model,CKPTPATH):
    # load a copy of the model from checkpiont
    pretrained_model = SquadModel.load_from_checkpoint(CHKPTPATH+'/best-checkpoint.ckpt')
    pretrained_model.freeze()
    
    def get_softmax_weights_from_checkpoint(pl_model):
        for array in pl_model.model.parameters():
            softmaxlayer = array   #will ultimately return the last layer (softmax)
        return softmaxlayer.detach().numpy()
    
    trained = get_softmax_weights_from_checkpoint(model)
    pretrained = get_softmax_weights_from_checkpoint(pretrained_model)
    
    

### to remove checkpoints, do rm -rf ./dir/

#trainer.test(test_dataloaders=data_module.test_dataloader())




