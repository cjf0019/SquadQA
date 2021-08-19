#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 16 16:09:02 2021

@author: newsjunkie345
"""

MODEL_NAME = 't5-small'
import torch
from torch import nn
import pytorch_lightning as pl
from transformers import T5ForConditionalGeneration
from transformers import AdamW
from squad_utilities import generate_answers
device = 'CPU'

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


class InputProcessor(nn.Module):
    def __init__(self, x_dim=512, vocab_dim=512, vocab_size=32128, embedding_dim=100,
                 one_hot=True, negative_padding=False):
        super(InputProcessor, self).__init__()

        self.x_dim = x_dim
        self.vocab_dim = vocab_dim
        self.vocab_size = vocab_size
        self.x_non_vocab_size = x_dim - vocab_size # inputs that aren't in the vocabulary
        self.negative_padding = negative_padding
        pad_idx = -100 if negative_padding else 0
        self.one_hot = one_hot
        if not self.one_hot:
            self.embed = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        else:
            embedding_dim = vocab_size
        self.embedding_dim = embedding_dim

    def forward(self, x, y=None):
        if self.x_non_vocab_size > 0:
            x, x_nv = torch.split(x, [self.x_dim-self.x_non_vocab_size, self.x_non_vocab_size], -1) # assumes vocab feats are before non vocab feats

        if self.one_hot:
            x = nn.functional.one_hot(x,self.vocab_size).float()
            print(x.shape, x)
        else:
            x = self.embed(x)

        x = torch.sum(x, 1)
        if self.x_non_vocab_size > 0:
            x = torch.cat((x, x_nv))
        return x



class BasicEncoder(nn.Module):
    def __init__(self, x_dim=512, vocab_dim=512, vocab_size=32128, embedding_dim=100,
                 nhid=16, ncond=0, output_dim=16, one_hot=True, negative_padding=False):
        super(BasicEncoder, self).__init__()

        self.x_dim = x_dim
        self.vocab_dim = vocab_dim
        self.vocab_size = vocab_size
        self.x_non_vocab_size = x_dim - vocab_size # inputs that aren't in the vocabulary
        self.negative_padding = negative_padding
        pad_idx = -100 if negative_padding else 0
        self.one_hot = one_hot
        if not self.one_hot:
            self.embed = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        else:
            embedding_dim = vocab_size

        self.enc1 = nn.Linear(embedding_dim, nhid)
        self.relu = nn.ReLU()
        self.calc_mean = nn.Linear(nhid + ncond, output_dim)
        self.calc_logvar = nn.Linear(nhid + ncond, output_dim)

    def forward(self, x, y=None):
        if self.x_non_vocab_size > 0:
            x, x_nv = torch.split(x, [self.x_dim-self.x_non_vocab_size, self.x_non_vocab_size], -1) # assumes vocab feats are before non vocab feats

        if self.one_hot:
            x = nn.functional.one_hot(x,self.vocab_size).float()
            print(x.shape, x)
        else:
            x = self.embed(x)

        x = torch.sum(x, 1)
        if self.x_non_vocab_size > 0:
            x = torch.cat((x, x_nv))

        x = self.relu(self.enc1(x))
        print(x.shape)
        if y is None:
            return self.calc_mean(x), self.calc_logvar(x)
        else:
            return self.calc_mean(torch.cat((x, y))), self.calc_logvar(torch.cat((x, y)))


class BasicDecoder(nn.Module):
    def __init__(self, x_dim=512, nhid=16, ncond=0):
        super(BasicDecoder, self).__init__()

        self.dec1 = nn.Linear(nhid, x_dim)
        self.relu = nn.ReLU()

    def forward(self, x, y=None):
        x = self.relu(self.dec1(x))
        if (y is None):
            return torch.sigmoid(x)
        else:
            return torch.sigmoid(torch.cat((x, y)))


class VAE(nn.Module):
    def __init__(self, shape, nhid=16):
        super(VAE, self).__init__()
        self.dim = nhid
        self.encoder = BasicEncoder(shape, nhid)
        self.decoder = BasicDecoder(shape, nhid)

    def sampling(self, mean, logvar):
        eps = torch.randn(mean.shape).to(device)
        sigma = 0.5 * torch.exp(logvar)
        return mean + eps * sigma

    def forward(self, x):
        mean, logvar = self.encoder(x)
        z = self.sampling(mean, logvar)
        return self.decoder(z), mean, logvar

    def generate(self, batch_size=None):
        z = torch.randn((batch_size, self.dim)).to(device) if batch_size else torch.randn((1, self.dim)).to(device)
        res = self.decoder(z)
        if not batch_size:
            res = res.squeeze(0)
        return res


class cVAE(nn.Module):
    def __init__(self, shape, nclass, nhid=16, ncond=16):
        super(cVAE, self).__init__()
        self.dim = nhid
        self.encoder = Encoder(shape, nhid, ncond=ncond)
        self.decoder = Decoder(shape, nhid, ncond=ncond)
        self.label_embedding = nn.Embedding(nclass, ncond)

    def sampling(self, mean, logvar):
        eps = torch.randn(mean.shape).to(device)
        sigma = 0.5 * torch.exp(logvar)
        return mean + eps * sigma

    def forward(self, x, y):
        y = self.label_embedding(y)
        mean, logvar = self.encoder(x, y)
        z = self.sampling(mean, logvar)
        return self.decoder(z, y), mean, logvar

    def generate(self, class_idx):
        if (type(class_idx) is int):
            class_idx = torch.tensor(class_idx)
        class_idx = class_idx.to(device)
        if (len(class_idx.shape) == 0):
            batch_size = None
            class_idx = class_idx.unsqueeze(0)
            z = torch.randn((1, self.dim)).to(device)
        else:
            batch_size = class_idx.shape[0]
            z = torch.randn((batch_size, self.dim)).to(device)
        y = self.label_embedding(class_idx)
        res = self.decoder(z, y)
        if not batch_size:
            res = res.squeeze(0)
        return res


# BCE_loss = nn.BCELoss(reduction = "sum")
def vae_loss(X, X_hat, mean, logvar):
    reconstruction_loss = nn.BCE_loss(X_hat, X)
    KL_divergence = 0.5 * torch.sum(-1 - logvar + torch.exp(logvar) + mean ** 2)
    return reconstruction_loss + KL_divergence


import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence
import torch.nn.functional as F
import torchtext
import nltk
import time
from datetime import timedelta
import numpy as np
from sklearn import metrics

NEG_INF = -10000
TINY_FLOAT = 1e-6


def mask_softmax(matrix, mask=None):
    """Perform softmax on length dimension with masking.

    Parameters
    ----------
    matrix: torch.float, shape [batch_size, .., max_len]
    mask: torch.long, shape [batch_size, max_len]
        Mask tensor for sequence.

    Returns
    -------
    output: torch.float, shape [batch_size, .., max_len]
        Normalized output in length dimension.
    """

    if mask is None:
        result = F.softmax(matrix, dim=-1)
    else:
        mask_norm = ((1 - mask) * NEG_INF).to(matrix)
        for i in range(matrix.dim() - mask_norm.dim()):
            mask_norm = mask_norm.unsqueeze(1)
        result = F.softmax(matrix + mask_norm, dim=-1)

    return result


def mask_mean(seq, mask=None):
    """Compute mask average on length dimension.

    Parameters
    ----------
    seq : torch.float, size [batch, max_seq_len, n_channels],
        Sequence vector.
    mask : torch.long, size [batch, max_seq_len],
        Mask vector, with 0 for mask.

    Returns
    -------
    mask_mean : torch.float, size [batch, n_channels]
        Mask mean of sequence.
    """

    if mask is None:
        return torch.mean(seq, dim=1)

    mask_sum = torch.sum(  # [b,msl,nc]->[b,nc]
        seq * mask.unsqueeze(-1).float(), dim=1)
    seq_len = torch.sum(mask, dim=-1)  # [b]
    mask_mean = mask_sum / (seq_len.unsqueeze(-1).float() + TINY_FLOAT)

    return mask_mean


def mask_max(seq, mask=None):
    """Compute mask max on length dimension.

    Parameters
    ----------
    seq : torch.float, size [batch, max_seq_len, n_channels],
        Sequence vector.
    mask : torch.long, size [batch, max_seq_len],
        Mask vector, with 0 for mask.

    Returns
    -------
    mask_max : torch.float, size [batch, n_channels]
        Mask max of sequence.
    """

    if mask is None:
        return torch.mean(seq, dim=1)

    mask_max, _ = torch.max(  # [b,msl,nc]->[b,nc]
        seq + (1 - mask.unsqueeze(-1).float()) * NEG_INF,
        dim=1)

    return mask_max


def seq_mask(seq_len, max_len):
    """Create sequence mask.

    Parameters
    ----------
    seq_len: torch.long, shape [batch_size],
        Lengths of sequences in a batch.
    max_len: int
        The maximum sequence length in a batch.

    Returns
    -------
    mask: torch.long, shape [batch_size, max_len]
        Mask tensor for sequence.
    """

    idx = torch.arange(max_len).to(seq_len).repeat(seq_len.size(0), 1)
    mask = torch.gt(seq_len.unsqueeze(1), idx).to(seq_len)

    return mask


class DynamicLSTM(nn.Module):
    """
    Dynamic LSTM module, which can handle variable length input sequence.

    Parameters
    ----------
    input_size : input size
    hidden_size : hidden size
    num_layers : number of hidden layers. Default: 1
    dropout : dropout rate. Default: 0.5
    bidirectional : If True, becomes a bidirectional RNN. Default: False.

    Inputs
    ------
    input: tensor, shaped [batch, max_step, input_size]
    seq_lens: tensor, shaped [batch], sequence lengths of batch

    Outputs
    -------
    output: tensor, shaped [batch, max_step, num_directions * hidden_size],
         tensor containing the output features (h_t) from the last layer
         of the LSTM, for each t.
    """

    def __init__(self, input_size, hidden_size=100,
                 num_layers=1, dropout=0., bidirectional=False):
        super(DynamicLSTM, self).__init__()

        self.lstm = nn.LSTM(
            input_size, hidden_size, num_layers, bias=True,
            batch_first=True, dropout=dropout, bidirectional=bidirectional)

    def forward(self, x, seq_lens):
        # pack input
        x_packed = pack_padded_sequence(
            x, seq_lens, batch_first=True, enforce_sorted=False)

        # pass through rnn
        y_packed, _ = self.lstm(x_packed)

        # unpack output
        y, length = pad_packed_sequence(y_packed, batch_first=True)
        return y


class Attention(torch.nn.Module):

    def __init__(self, encoder_dim: int, decoder_dim: int):
        super().__init__()
        self.encoder_dim = encoder_dim
        self.decoder_dim = decoder_dim

    def forward(self,
                query: torch.Tensor,  # [decoder_dim]
                values: torch.Tensor,  # [seq_length, encoder_dim]
                ):
        weights = self._get_weights(query, values)  # [seq_length]
        weights = torch.nn.functional.softmax(weights, dim=0)
        return weights @ values  # [encoder_dim]


class AdditiveAttention(Attention):

    def __init__(self, encoder_dim, decoder_dim):
        super().__init__(encoder_dim, decoder_dim)
        self.v = torch.nn.Parameter(
            torch.FloatTensor(self.decoder_dim).uniform_(-0.1, 0.1))
        self.W_1 = torch.nn.Linear(self.decoder_dim, self.decoder_dim)
        self.W_2 = torch.nn.Linear(self.encoder_dim, self.decoder_dim)

    def _get_weights(self,
                     query: torch.Tensor,  # [decoder_dim]
                     values: torch.Tensor,  # [seq_length, encoder_dim]
                     ):
        query = query.repeat(values.size(0), 1)  # [seq_length, decoder_dim]
        weights = self.W_1(query) + self.W_2(values)  # [seq_length, decoder_dim]
        return torch.tanh(weights) @ self.v  # [seq_length]


class MultiplicativeAttention(Attention):

    def __init__(self, encoder_dim: int, decoder_dim: int):
        super().__init__(encoder_dim, decoder_dim)
        self.W = torch.nn.Parameter(torch.FloatTensor(
            self.decoder_dim, self.encoder_dim).uniform_(-0.1, 0.1))

    def _get_weights(self,
                     query: torch.Tensor,  # [decoder_dim]
                     values: torch.Tensor,  # [seq_length, encoder_dim]
                     ):
        weights = query @ self.W @ values.T  # [seq_length]
        return weights / np.sqrt(self.decoder_dim)  # [seq_length]


class Attention(nn.Module):
    """
    Inputs:
        last_hidden: (batch_size, hidden_size)
        encoder_outputs: (batch_size, max_time, hidden_size)
    Returns:
        attention_weights: (batch_size, max_time)
    """

    def __init__(self, batch_size, hidden_size, method="dot", mlp=False):
        super(Attention, self).__init__()
        self.method = method
        self.hidden_size = hidden_size
        if method == 'dot':
            pass
        elif method == 'general':  ### I think this might be the location based attention
            self.Wa = nn.Linear(hidden_size, hidden_size, bias=False)
        elif method == "concat":
            self.Wa = nn.Linear(hidden_size, hidden_size, bias=False)
            self.va = nn.Parameter(torch.FloatTensor(batch_size, hidden_size))
        elif method == 'bahdanau':
            self.Wa = nn.Linear(hidden_size, hidden_size, bias=False)
            self.Ua = nn.Linear(hidden_size, hidden_size, bias=False)
            self.va = nn.Parameter(torch.FloatTensor(batch_size, hidden_size))
        else:
            raise NotImplementedError

        self.mlp = mlp
        if mlp:
            self.phi = nn.Linear(hidden_size, hidden_size, bias=False)
            self.psi = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, last_hidden, encoder_outputs, seq_len=None):
        batch_size, seq_lens, _ = encoder_outputs.size()
        if self.mlp:
            last_hidden = self.phi(last_hidden)
            encoder_outputs = self.psi(encoder_outputs)

        attention_energies = self._score(last_hidden, encoder_outputs, self.method)

        if seq_len is not None:
            attention_energies = mask_3d(attention_energies, seq_len, -float('inf'))

        return F.softmax(attention_energies, -1)

    def _score(self, last_hidden, encoder_outputs, method):
        """
        Computes an attention score
        :param last_hidden: (batch_size, hidden_dim)
        :param encoder_outputs: (batch_size, max_time, hidden_dim)
        :param method: str (`dot`, `general`, `concat`, `bahdanau`)
        :return: a score (batch_size, max_time)
        """

        assert encoder_outputs.size()[-1] == self.hidden_size

        if method == 'dot':
            last_hidden = last_hidden.unsqueeze(-1)
            return encoder_outputs.bmm(last_hidden).squeeze(-1)

        elif method == 'general':
            x = self.Wa(last_hidden)
            x = x.unsqueeze(-1)
            return encoder_outputs.bmm(x).squeeze(-1)

        elif method == "concat":
            x = last_hidden.unsqueeze(1)
            x = F.tanh(self.Wa(torch.cat((x, encoder_outputs), 1)))
            return x.bmm(self.va.unsqueeze(2)).squeeze(-1)

        elif method == "bahdanau":
            x = last_hidden.unsqueeze(1)
            out = F.tanh(self.Wa(x) + self.Ua(encoder_outputs))
            return out.bmm(self.va.unsqueeze(2)).squeeze(-1)

        else:
            raise NotImplementedError

    def extra_repr(self):
        return 'score={}, mlp_preprocessing={}'.format(
            self.method, self.mlp)
