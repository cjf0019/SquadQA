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
        """
        Switch back and forth between a integerized inputs (corresponding to vocab)
        and embeddings/one-hot.

        FOR ENCODING :
            INPUT : integer sequences corresponding to sequences of codes/words
            OUTPUT : embeddings/one-hot vectors (dim of embedding_dim for former, vocab_size for latter)

        FOR DECODING :
            INPUT : If one_hot, should just be the result of a softmax or sigmoid, of dimension vocab_size,
                    corresponding to the log-prob of the output being each vocab word.
                    If embedding, should correspond to a vector in the embedding space, such that
                        nearest neighbors can be performed with the vector and each vocab embedding
            OUTPUT : The vocab codes/words corresponding to 1) the most probable for one_hot
                        or 2) the nearest neighbor embedding for embeddings

        NOTE : Assumes the same embedding weight matrix used for both encoding and decoding
        """

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
        self.mode = 'encode'

    def forward(self, x):
        if self.mode == 'encode':
            return self.encode_(x)
        elif self.mode == 'decode':
            return self.decode_(x)
        else:
            raise Exception("Received mode of {}. Will only accept 'encode' or 'decode'".format(self.mode))

    def encode_(self, x):
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

    def decode_(self,x):
        if self.one_hot:
            x = torch.round(x).int()
            x = torch.argmax(x,dim=-1)
            return x    # Returns the batch of integerized sequences

        else:
            distance = torch.norm(self.embed.weight.data - x, dim=1)
            nearest = torch.argmin(distance)


class BasicEncoder(nn.Module):
    def __init__(self, x_dim=512, nhid=16, ncond=0, output_dim=16):
        super(BasicEncoder, self).__init__()

        self.x_dim = x_dim
        self.enc1 = nn.Linear(x_dim, nhid)
        self.relu = nn.ReLU()
        self.calc_mean = nn.Linear(nhid + ncond, output_dim)
        self.calc_logvar = nn.Linear(nhid + ncond, output_dim)

    def forward(self, x, y=None):
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


class ConvolutionalEncoder(nn.Module):
    def __init__(self, x_dim=(512,32128), output_dim=100, ncond=0):
        super(ConvolutionalEncoder, self).__init__()

        self.kernel_size = 5
        self.stride = 2
        self.dilation = 1
        self.padding = 0
        self.x_dim = x_dim  # seq_len x vocab_dim
        #self.enc1 = nn.Linear(x_dim[1], nhid)
        self.relu = nn.ReLU()
        self.conv1 = nn.Conv1d(x_dim[1], 200, self.kernel_size, stride=self.stride)
        self.conv2 = nn.Conv1d(200, 300, self.kernel_size, stride=self.stride)
        self.conv3 = nn.Conv1d(300, output_dim, self.kernel_size, stride=self.stride)
        #self.max_pool2 = nn.MaxPool1d(512)

        linear_dim = self.calc_linear_dim()
        self.calc_z = nn.Linear(int(linear_dim)+ncond, 2)

    def forward(self, x, y=None):
        x = x.permute(0, 2, 1)  # batch_size x seq_len x vocab/embed_dim
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        #x = self.max_pool2(x)
        #x = self.relu(x)
        print(x.shape)
        if y is not None:
            x = torch.cat((x, y))

        mean_var = self.relu(self.calc_z(x))
        mean, var = mean_var.split(1, dim=-1)
        return mean.squeeze(), var.squeeze()

    def calc_linear_dim(self):
        """ calculate the dimension required from last conv layer to use in the subsequent MLP layer"""
        def calc_lout(l,padding,dilation,ksize,stride):
            return (l+2*padding-dilation*(ksize-1)-1)/stride + 1

        conv1dim = calc_lout(512,self.padding,self.dilation,self.kernel_size,self.stride)
        conv2dim = calc_lout(conv1dim,self.padding,self.dilation,self.kernel_size,self.stride)
        conv3dim = calc_lout(conv2dim, self.padding, self.dilation, self.kernel_size, self.stride)
        return conv3dim


class ConvolutionalDecoder(nn.Module):
    def __init__(self, output_dim=(512,32128), nhid=100, ncond=0, softmax_temp=0.01):
        super(ConvolutionalDecoder, self).__init__()

        self.kernel_size = 5
        self.stride = 2
        self.dilation = 1
        self.padding = 0
        self.output_padding = 0
        self.output_dim = output_dim  # seq_len x vocab_dim
        #self.enc1 = nn.Linear(x_dim[1], nhid)
        self.relu = nn.ReLU()
        self.linear = nn.Linear(1, 61+ncond)
        self.conv1 = nn.ConvTranspose1d(nhid, 300, self.kernel_size, stride=self.stride,
                                        padding=self.padding, output_padding=self.output_padding)
        self.conv2 = nn.ConvTranspose1d(300, 200, self.kernel_size, stride=self.stride,
                                        padding=self.padding, output_padding=1)  #getting issues syncing with encoder convs, so adding 1 output padding
        self.conv3 = nn.ConvTranspose1d(200, output_dim[1], self.kernel_size, stride=self.stride,
                                        padding=self.padding, output_padding=1)
        #self.max_pool2 = nn.MaxPool1d(512)

        self.softmax_temp = softmax_temp
        linear_dim = self.calc_linear_dim()
        #self.calc_mean = nn.Linear(nhid + ncond, output_dim)
        #self.calc_logvar = nn.Linear(nhid + ncond, output_dim)
        #self.calc_z = nn.Linear(int(linear_dim)+ncond, 2)

    def forward(self, x, y=None):
        #x = x.permute(0, 2, 1)  # batch_size x seq_len x vocab/embed_dim
        x = x.view((x.shape[0],x.shape[1],1))
        x = self.linear(x)
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        #x = self.max_pool2(x)
        #x = self.relu(x)
        if y is not None:
            x = torch.cat((x, y))

        # the following softmax assumes one-hot embedding
        x = x.permute(0,2,1)  # batch_size x vocab/embed_dim x seq_len
        #x = torch.div(x,torch.norm(x, dim=1)) #normalize per sequence index
        x = torch.softmax(x/self.softmax_temp, axis=-1)
        return x

    def calc_linear_dim(self):
        """ calculate the dimension required from last conv layer to use in the subsequent MLP layer"""
        def calc_lout(l,padding,dilation,ksize,stride,output_padding):
            return (l-1)*stride - 2*padding + dilation*(ksize-1) + output_padding + 1

        conv1dim = calc_lout(61,self.padding,self.dilation,self.kernel_size,self.stride,self.output_padding)
        conv2dim = calc_lout(conv1dim,self.padding,self.dilation,self.kernel_size,self.stride,self.output_padding)
        conv3dim = calc_lout(conv2dim, self.padding, self.dilation, self.kernel_size, self.stride,self.output_padding)
        return conv3dim


class ConvolutionalEncoderDecoder(nn.Module):
    def __init__(self, x_dim=(512,32128), nhid=100, ncond=0, decoder_softmax_temp=0.01, mode='encode'):
        super(ConvolutionalEncoderDecoder, self).__init__()

        self.mode = mode
        # general convolutional parameters
        self.kernel_size = 5
        self.stride = 2
        self.dilation = 1
        self.padding = 0
        self.x_dim = x_dim  # seq_len x vocab_dim
        self.linear_dims = self.calc_linear_dims()
        #self.output_padding = 0  # only used for decoder... added to right side of convtranspose
        #self.enc1 = nn.Linear(x_dim[1], nhid)
        self.relu = nn.ReLU()

        # encoder params
        self.conv1 = nn.Conv1d(x.shape[2], 200, self.kernel_size, stride=self.stride)
        self.conv2 = nn.Conv1d(200, 300, self.kernel_size, stride=self.stride)
        self.conv3 = nn.Conv1d(300, nhid, self.kernel_size, stride=self.stride)
        self.calc_z = nn.Linear(int(self.linear_dims[0])+ncond, 2)

        # decoder params
        self.decode_z = nn.Linear(1, self.linear_dims[0] + ncond)
        self.conv1t = nn.ConvTranspose1d(nhid, 300, self.kernel_size, stride=self.stride,
                                        padding=self.padding)
        self.conv2t = nn.ConvTranspose1d(300, 200, self.kernel_size, stride=self.stride,
                                        padding=self.padding)  # getting issues syncing with encoder convs, so adding 1 output padding
        self.conv3t = nn.ConvTranspose1d(200, x_dim[1], self.kernel_size, stride=self.stride,
                                        padding=self.padding, output_padding=self.out_pad)

        self.decoder_softmax_temp = decoder_softmax_temp

    def forward(self, x, y=None):
        if self.mode == 'encode':
            return self._encode(x,y)
        elif self.mode == 'decode':
            return self._decode(x,y)
        else:
            raise Exception("Received mode of {}. Will only accept 'encode' or 'decode'".format(self.mode))

    def _encode(self, x, y=None):
        x = x.permute(0, 2, 1)  # batch_size x seq_len x vocab/embed_dim
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        if y is not None:
            x = torch.cat((x, y))

        mean_var = self.relu(self.calc_z(x))
        mean, var = mean_var.split(1, dim=-1)
        return mean.squeeze(), var.squeeze()

    def _decode(self, x, y=None):
        x = x.view((x.shape[0], x.shape[1], 1))  # unflatten hidden layer
        x = self.decode_z(x)
        x = self.relu(self.conv1t(x))
        x = self.relu(self.conv2t(x))
        x = self.relu(self.conv3t(x))
        if y is not None:
            x = torch.cat((x, y))

        # the following softmax assumes one-hot embedding
        x = x.permute(0,2,1)  # batch_size x vocab/embed_dim x seq_len
        #x = torch.div(x,torch.norm(x, dim=1)) #normalize per sequence index
        x = torch.softmax(x/self.decoder_softmax_temp, axis=-1)
        return x

    def calc_lout_encoder(self, l, padding, dilation, ksize, stride):
        return (l+2*padding-dilation*(ksize-1)-1)/stride + 1

    def calc_lout_decoder(self, l, padding, dilation, ksize, stride, output_padding):
        return (l - 1) * stride - 2 * padding + dilation * (ksize - 1) + output_padding + 1

    def calc_linear_dims(self):
        """ calculate the dimensions required for
        1) encoder's last conv layer to use in the subsequent MLP layer
        2) decoders last conv_transpose layer (should be equal to the original input seq length
        3) output padding (out_pad) to apply to last decoder conv_transpose layer to ensure equality to original seq len
        """
        ### Encoder convolutional layers length dimension
        self.conv1dim = self.calc_lout_encoder(self.x_dim[0],self.padding,self.dilation,self.kernel_size,self.stride)
        self.conv2dim = self.calc_lout_encoder(self.conv1dim,self.padding,self.dilation,self.kernel_size,self.stride)
        self.conv3dim = self.calc_lout_encoder(self.conv2dim, self.padding, self.dilation, self.kernel_size, self.stride)

        ### Decoder convolutional layers length dimension
        self.conv1tdim = self.calc_lout_decoder(int(round(self.conv3dim)),self.padding,self.dilation,self.kernel_size,self.stride,0)
        self.conv2tdim = self.calc_lout_decoder(self.conv1tdim,self.padding,self.dilation,self.kernel_size,self.stride,0)
        self.out_pad = int(round(self.conv1dim)) - int(round(self.conv2tdim)) #ensure the resulting dimension mirrors that from the encoder conv operation
        self.conv3tdim = self.calc_lout_decoder(self.conv2tdim,self.padding,self.dilation,self.kernel_size,self.stride,self.out_pad)
        return (int(round(self.conv3dim)), int(round(self.conv3tdim)))


class VAE(nn.Module):
    def __init__(self, shape=(512,32128), nhid=100, device='cpu'):
        super(VAE, self).__init__()
        self.dim = nhid
        #self.encoder = BasicEncoder(shape, nhid)
        #self.decoder = BasicDecoder(shape[1], nhid)
        self.encoder = ConvolutionalEncoder(x_dim=shape,nhid=nhid)
        self.decoder = ConvolutionalDecoder(output_dim=shape,nhid=nhid)
        self.device = device

    def sampling(self, mean, logvar):
        eps = torch.randn(mean.shape).to(self.device)
        sigma = 0.5 * torch.exp(logvar)
        return mean + eps * sigma

    def forward(self, x):
        mean, logvar = self.encoder(x)
        z = self.sampling(mean, logvar)
        return self.decoder(z), mean, logvar

    def generate(self, batch_size=None):
        z = torch.randn((batch_size, self.dim)).to(self.device) if batch_size else torch.randn((1, self.dim)).to(self.device)
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
