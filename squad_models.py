#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 16 16:09:02 2021

@author: newsjunkie345
"""

MODEL_NAME = 't5-small'
import torch
import pytorch_lightning as pl
from transformers import T5ForConditionalGeneration
from transformers import AdamW
from squad_utilities import generate_answers

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
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        ans_pred_col = "val_ans_pred_epoch_{}".format(str(self.current_epoch))
        if ans_pred_col not in self.data_module.test_df.columns:
            self.data_module.test_df[ans_pred_col] = ''
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['labels']
        row_ids = batch['Row_ID'].detach().numpy()
        loss, outputs = self(input_ids, attention_mask, labels)
        
        self.log('val_loss', loss, prog_bar=True, logger=True)
        #self.generate_answers(input_ids,batch['Row_ID'].values,len(labels))
        if self.generate_text:
            generation_output = self.model.generate(torch.tensor(input_ids),max_length=self.max_label_len)
            answers = [self.tokenizer.decode(i) for i in generation_output.detach().numpy()]
            self.data_module.test_df.loc[row_ids,ans_pred_col] = answers
        
        #self.data_module.test_df.loc[row_ids,ans_pred_col] = \
        #                    generate_answers(input_ids,\
        #                    50,self.model,input_col='context',tokenize_input=False)
        
        return loss
    
    def test_step(self, batch, batch_idx):
        ans_pred_col = "test_ans_pred_epoch_{}".format(str(self.current_epoch))
        if ans_pred_col not in self.data_module.test_df.columns:
            self.data_module.test_df[ans_pred_col] = ''
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['labels']
        row_ids = batch['Row_ID'].detach().numpy()
        loss, outputs = self(input_ids, attention_mask, labels)
        self.log('test_loss', loss, prog_bar=True, logger=True)
        
        if self.generate_text:
            generation_output = self.model.generate(torch.tensor(input_ids),max_length=self.max_label_len)
            answers = [self.tokenizer.decode(i) for i in generation_output.detach().numpy()]
            self.data_module.test_df.loc[row_ids,ans_pred_col] = answers
        #generate_answers(input_ids,\
            #50,self.model,input_col='context',tokenize_input=False)
        #self.generate_answers(input_ids,batch['Row_ID'].values,len(labels))

    def prepare_for_text_generation(self,data_module):
        self.data_module = data_module
        self.tokenizer = data_module.tokenizer
        self.max_label_len = data_module.max_label_len
        self.generate_text = True
        

    def configure_optimizers(self):
        return AdamW(self.parameters(), lr=0.001)