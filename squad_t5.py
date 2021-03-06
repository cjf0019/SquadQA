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
from pl_bolts.callbacks import ModuleDataMonitor,TrainingDataMonitor,\
    BatchGradientVerificationCallback,PrintTableMetricsCallback
from squad_utilities import process_generated_answer, stage_squad_data_pytorch
from squad_datasets_datamodules import SquadDataset, SquadDataModule
from squad_models import SquadModel

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import datetime



os.environ["CUDA_VISIBLE_DEVICES"] = "-1"   # dont use GPU for now
input_dir = ".//"


from transformers import T5Tokenizer, T5Model, T5ForConditionalGeneration


from torch.utils.data import DataLoader
from transformers import AdamW


BATCH_SIZE = 64
N_EPOCHS = 3
MODEL_NAME = 't5-small'
SEED = 13
CHKPTPATH = str(datetime.date.today())+'/checkpoints'
MODE = 'train'
LOAD_PRETRAINED_FILE = False
OUTPUT_PREDICTION_FILE = True


pl.seed_everything(SEED)


def run():
    train_df, val_df = stage_squad_data_pytorch()

    data_module = SquadDataModule(
        train_df,
        val_df,
        tokenizer=T5Tokenizer.from_pretrained(MODEL_NAME),
        batch_size=BATCH_SIZE
    )

    data_module.setup()

    if CHKPTPATH is not None and LOAD_PRETRAINED_FILE:
        model = SquadModel.load_from_checkpoint(CHKPTPATH+'/best-checkpoint-v2.ckpt')
    else:
        model = SquadModel()
        
    #model.data_module = data_module
    
    for name, param in model.named_parameters():
        #if 'decoder.final_layer_norm' not in name:
        if 'encoder' in name:
            param.requires_grad = False

    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath=CHKPTPATH,
        filename='best-checkpoint',
        save_top_k=3,
        verbose=True,
        monitor='val_loss',
        mode='min')
            

    #verification = BatchGradientVerificationCallback()
    monitor = ModuleDataMonitor(submodules=['model.lm_head'],log_every_n_steps=1000)
    #monitor = TrainingDataMonitor(log_every_n_steps=1)
    #printcallback = PrintTableMetricsCallback()
    trainer = pl.Trainer(
        checkpoint_callback = checkpoint_callback,
        max_epochs = N_EPOCHS,
        #logger=pl.loggers.TensorBoardLogger('logs/'),
        progress_bar_refresh_rate = 30,
        check_val_every_n_epoch=1,
        deterministic=True,\
        callbacks=[monitor]
        )
    
    model.prepare_for_text_generation(data_module)
        
    if MODE == 'train':   
        model.generate_text = False
        trainer.fit(model,data_module)

    elif MODE == 'test':
        model.freeze()
        pred =trainer.test(model,test_dataloaders=data_module.test_dataloader())

        if OUTPUT_PREDICTION_FILE:
            process_generated_answer(data_module)

### to remove checkpoints, do rm -rf ./dir/

