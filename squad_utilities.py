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


def tokenize_to_dict(tokenizer, text, output_len, text_label=None, make_pad_negative=False):
    text_label = '' if text_label is None else text_label
    encodings = tokenizer(text, truncation=True, max_length=output_len, padding="max_length", add_special_tokens=True, return_tensors='pt')
    if make_pad_negative:
        input_ids = encodings['input_ids']
        input_ids[input_ids == 0] = -100
        encodings['input_ids'] = input_ids

    encodings[(text_label+'_input_ids').lstrip('_')] = encodings.pop('input_ids')
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







