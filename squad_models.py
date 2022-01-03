#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 16 16:09:02 2021

@author: newsjunkie345
"""

MODEL_NAME = 't5-small'
NEG_INF = -10000
TINY_FLOAT = 1e-6

import torch
from torch import nn
import pytorch_lightning as pl
from transformers import T5ForConditionalGeneration
from transformers import AdamW
from squad_utilities import generate_answers
device = 'CPU'

from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence
import torch.nn.functional as F
import numpy as np


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


class SquadModel_VAE(pl.LightningModule):
    def __init__(self, tokenizer, x_dim, beta=1.0, neg_samp_embed_loc=None, neg_samp_embed_cov=None):
        super().__init__()
        self.tokenizer = tokenizer
        self.seq_len = x_dim[-2]
        self.vocab_dim = x_dim[-1]

        if tokenizer is not None:
            self.inpproc = InputProcessor(tokenizer, seq_len=self.seq_len, one_hot=False)
        self.model = VAE(x_dim=(128,300), nhid=100, dec_softmax_temp=0.01)   # x_dim is of seq_len x embed_dim... might want to make more clear
        self.loss_model = NegativeSamplingLoss(tokenizer, neg_samp_embed_loc=neg_samp_embed_loc, neg_samp_embed_cov=neg_samp_embed_cov)
        self.beta = beta # mult constant on the kl divergence loss

    def forward(self, x):
        if self.tokenizer is not None:
            x = self.inpproc(x) # convert input_ids to word embeddings

        output, mean, logvar = self.model(x)

        ### !!! The loss model takes one word (batched) at a time... need to modify either this or the loss model to do sentences
        loss = self.loss_model(output, x)
        kl_loss = self.kl_divergence(mean, logvar)
        loss += kl_loss

        if self.tokenizer is not None:  #convert back into tok integers
            self.inpproc.mode = 'decode'
            output = self.inpproc(output)
            self.inpproc.mode = 'encode'
        return output, loss

    def training_step(self, batch, batch_idx):
        input_ids = batch['input_ids']
        outputs, loss = self(input_ids)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        #ans_pred_col = "val_ans_pred_epoch_{}".format(str(self.current_epoch))
        #if ans_pred_col not in self.data_module.test_df.columns:
        #    self.data_module.test_df[ans_pred_col] = ''
        input_ids = batch['input_ids']
        #row_ids = batch['Row_ID'].detach().numpy()
        outputs, loss = self(input_ids)

        self.log('val_loss', loss, prog_bar=True, logger=True)
        # self.generate_answers(input_ids,batch['Row_ID'].values,len(labels))
        #if self.generate_text:
        #    generation_output = self.model.generate(torch.tensor(input_ids), max_length=self.max_label_len)
        #    answers = [self.tokenizer.decode(i) for i in generation_output.detach().numpy()]
        #    self.data_module.test_df.loc[row_ids, ans_pred_col] = answers

        # self.data_module.test_df.loc[row_ids,ans_pred_col] = \
        #                    generate_answers(input_ids,\
        #                    50,self.model,input_col='context',tokenize_input=False)

        return loss

    def test_step(self, batch, batch_idx):
        #ans_pred_col = "test_ans_pred_epoch_{}".format(str(self.current_epoch))
        #if ans_pred_col not in self.data_module.test_df.columns:
        #    self.data_module.test_df[ans_pred_col] = ''
        input_ids = batch['input_ids']
        #row_ids = batch['Row_ID'].detach().numpy()
        loss, outputs = self(input_ids)
        self.log('test_loss', loss, prog_bar=True, logger=True)

        #if self.generate_text:
        #    generation_output = self.model.generate(torch.tensor(input_ids), max_length=self.max_label_len)
        #    answers = [self.tokenizer.decode(i) for i in generation_output.detach().numpy()]
        #    self.data_module.test_df.loc[row_ids, ans_pred_col] = answers
        # generate_answers(input_ids,\
        # 50,self.model,input_col='context',tokenize_input=False)
        # self.generate_answers(input_ids,batch['Row_ID'].values,len(labels))

    def prepare_for_text_generation(self, data_module):
        self.data_module = data_module
        self.tokenizer = data_module.tokenizer
        self.max_label_len = data_module.max_label_len
        self.generate_text = True

    def configure_optimizers(self):
        return AdamW(self.parameters(), lr=0.001)

    @staticmethod
    def kl_divergence(mean, logvar, beta=1.0):
        KL_divergence = 0.5 * torch.sum(-1 - logvar + torch.exp(logvar) + mean ** 2)
        return beta * KL_divergence


class InputProcessor(nn.Module):
    def __init__(self, tokenizer, x_dim=512, seq_len=512, one_hot=True, pad_idx=20000, agg_seq=False):
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
        #self.batch_size = batch_size
        self.seq_len = seq_len
        self.tokenizer = tokenizer
        #self.vocab_dim = vocab_dim
        self.vocab_size = self.tokenizer.vocab_size
        self.x_non_vocab_size = x_dim - self.vocab_size # inputs that aren't in the vocabulary
        self.pad_idx = pad_idx
        self.one_hot = one_hot
        #self.calculate_loss = calculate_loss
        if not self.one_hot:
            #if self.embed_model is None:
            #    self.embed_model = gensim.downloader.load("glove-wiki-gigaword-300")
            #    weights = torch.FloatTensor(self.embed_model.wv.vectors)
            #else:
            #    self.embed_model = embed_model
            #    ### !!! For non gensim models, the weights might be from a different mechanism... modify in the future

            # Build nn.Embedding() layer
            self.weights = self.tokenizer.weights
            self.embed = nn.Embedding.from_pretrained(self.weights,padding_idx=self.pad_idx)
            self.embed.requires_grad = False
            #self.embed = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
            self.embedding_dim = self.weights.shape[-1]

        else:
            self.embedding_dim = self.vocab_size
        self.mode = 'encode'
        self.agg_seq = agg_seq # optionally reduce the sequence down through summing ('sum') or mean ('mean')

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
        else:
            x = self.embed(x.int())

        if self.x_non_vocab_size > 0:  # add non-vocab features
            x = torch.cat((x, x_nv))

        if self.agg_seq == 'sum' or self.agg_seq == True:
            x = torch.sum(x, -2)  #sum the second to last dimension, corresponding to per sentence
        elif self.agg_seq == 'mean':
            x = torch.mean(x, -2)
        return x

    def decode_(self,x):
        if self.one_hot:
            #x = torch.round(x).int()
            #x = torch.argmax(x,dim=-1)

            # to make differentiable, use gumbel softmax... acts on the last dimension, which should be the vocab logits
            x = torch.nn.functional.gumbel_softmax(x,hard=True)

            if self.calculate_loss:
                pass

            return x    # Returns the batch of integerized sequences

        else:
            #distance = torch.norm(self.embed.weight.data - x, dim=1)
            to_flatten = [dim for dim in x.shape[:-1]]
            decode_dims = (np.prod(to_flatten), x.shape[-1]) # flatten so each row corresponds to an embedding in a sentence/example/batch

            cos_sim = torch.matmul(x.view(decode_dims) / (torch.norm(x.view(decode_dims), dim=0) + 1e-8),
                    self.embed.weight.data.T.float() / (torch.norm(self.embed.weight.data.T.float(), dim=0) + 1e-8)).view(
                        tuple(to_flatten + [self.vocab_size]))

            #distance = torch.matmul(x.view((self.batch_size * self.seq_len, self.embedding_dim)) /
            #                        (torch.norm(x.view((self.batch_size * self.seq_len, self.embedding_dim)), dim=0) + 1e-8),
            #             self.embed.weight.data.T.float() /
            #             (torch.norm(self.embed.weight.data.T.float(), dim=0) + 1e-8)).view(self.batch_size, self.seq_len, self.vocab_size)

            #nearest = torch.argmin(distance,dim=-1)
            nearest = torch.argmax(cos_sim, dim=-1)
            return nearest

    def convert_tok_to_word(self, tok_result):
        return [self.tokenizer.decode(i) for i in tok_result]


class AliasMultinomial(object):
    """
    Fast sampling from a multinomial distribution.
    https://hips.seas.harvard.edu/blog/2013/03/03/the-alias-method-efficient-sampling-with-many-discrete-outcomes/
    Code taken from: https://github.com/TropComplique/lda2vec-pytorch/blob/master/utils/alias_multinomial.py
    """

    def __init__(self, probs, device):
        """
        probs: a float tensor with shape [K].
            It represents probabilities of different outcomes.
            There are K outcomes. Probabilities sum to one.
        """
        self.device = device

        K = len(probs)
        self.q = t.zeros(K).to(device)
        self.J = t.LongTensor([0] * K).to(device)

        # sort the data into the outcomes with probabilities
        # that are larger and smaller than 1/K
        smaller = []
        larger = []
        for kk, prob in enumerate(probs):
            self.q[kk] = K * prob
            if self.q[kk] < 1.0:
                smaller.append(kk)
            else:
                larger.append(kk)

        # loop though and create little binary mixtures that
        # appropriately allocate the larger outcomes over the
        # overall uniform mixture
        while len(smaller) > 0 and len(larger) > 0:
            small = smaller.pop()
            large = larger.pop()

            self.J[small] = large
            self.q[large] = (self.q[large] - 1.0) + self.q[small]

            if self.q[large] < 1.0:
                smaller.append(large)
            else:
                larger.append(large)

        self.q.clamp(0.0, 1.0)
        self.J.clamp(0, K - 1)

    def draw(self, N):
        """Draw N samples from the distribution."""

        K = self.J.size(0)
        r = t.LongTensor(np.random.randint(0, K, size=N)).to(self.device)
        q = self.q.index_select(0, r).clamp(0.0, 1.0)
        j = self.J.index_select(0, r)
        b = t.bernoulli(q)
        oq = r.mul(b.long())
        oj = j.mul((1 - b).long())

        return oq + oj


class NegativeSamplingLoss(nn.Module):
    BETA = 0.75
    NUM_SAMPLES = 2

    def __init__(self, tokenizer, neg_samp_embed_loc=None, neg_samp_embed_cov=None, device='cpu'):
        super(NegativeSamplingLoss, self).__init__()
        self.tokenizer = tokenizer
        self.criterion = nn.BCEWithLogitsLoss()
        self.embeddings = tokenizer.weights
        self.device = device

        if 'global_embed_mean' in tokenizer.summary_statistics or neg_samp_embed_loc is not None:
            embed_mean = tokenizer.summary_statistics['global_embed_mean'] if neg_samp_embed_loc is None else neg_samp_embed_loc
            embed_cov = neg_samp_embed_cov if 'global_embed_covariance' not in tokenizer.summary_statistics \
                                            else tokenizer.summary_statistics['global_embed_covariance']

            self.multivariate_normal_sampler = torch.distributions.MultivariateNormal(embed_mean, embed_cov)
            self.sample_mode = 'multivariate_normal'

        elif 'global_word_count' in tokenizer.summary_statistics:
            self.vocab_len = len(tokenizer.summary_statistics['global_word_count'])
            self.transformed_freq_vec = torch.tensor(
                np.array(list(tokenizer.summary_statistics['global_word_count'].values())) ** self.BETA)

            self.freq_sum = torch.sum(self.transformed_freq_vec)
            # Generate table
            self.unigram_table = self.generate_unigram_table()
            self.sample_mode = 'unigram'

    def forward(self, predicted, target):
        predicted, target = predicted.squeeze(), target.squeeze()  # batch_size x embed_size

        # Compute true portion
        true_scores = (predicted * target).sum(-1)  # batch_size
        loss = self.criterion(true_scores, torch.ones_like(true_scores))

        # Compute negatively sampled portion -
        for i in range(self.NUM_SAMPLES):
            samples = self.get_samples(n_shape=tuple(predicted.shape[:-1]))
            neg_sample_scores = (predicted * samples).sum(-1)
            # Update loss
            loss += self.criterion(neg_sample_scores, torch.zeros_like(neg_sample_scores))

        return loss

    def get_samples(self, n_shape):
        if self.sample_mode == 'unigram':
            return self.get_unigram_samples(n_shape)
        else:
            return self.get_multivariate_normal_samples(n_shape)

    def get_multivariate_normal_samples(self,n_shape):
        return self.multivariate_normal_sampler.sample(n_shape)

    def get_unigram_samples(self, n_shape):
        """
        Returns a sample according to a unigram distribution
        Randomly choose a value from self.unigram_table
        """
        rand_idxs = self.unigram_table.draw(n_shape).to(self.device)
        return self.embeddings(rand_idxs).squeeze()

    def get_unigram_prob(self, token_idx):
        return (self.transformed_freq_vec[token_idx].item()) / self.freq_sum.item()

    def generate_unigram_table(self):
        # Probability at each index corresponds to probability of selecting that token
        pdf = [self.get_unigram_prob(t_idx) for t_idx in range(0, self.vocab_len)]
        # Generate the table from PDF
        return AliasMultinomial(pdf, self.device)

    @staticmethod
    def bce_loss_w_logits(x, y):
        max_val = np.clip(x, 0, None)
        loss = x - x * y + max_val + np.log(np.exp(-max_val) + np.exp((-x - max_val)))
        return loss.mean()



class VAEInputLoss(nn.Module):
    BETA = 0.75  # exponent to adjust sampling frequency
    NUM_SAMPLES = 2

    def __init__(self, dataset, embeddings, device):
        super(VAEInputLoss, self).__init__()
        self.dataset = dataset
        self.criterion = nn.BCEWithLogitsLoss()
        self.vocab_len = len(dataset.dictionary)
        self.embeddings = embeddings
        self.device = device

        # Helpful values for unigram distribution generation
        # Should use cfs instead but: https://github.com/RaRe-Technologies/gensim/issues/2574
        self.transformed_freq_vec = t.tensor(
            np.array([dataset.dictionary.dfs[i] for i in range(self.vocab_len)]) ** self.BETA)

        self.freq_sum = t.sum(self.transformed_freq_vec)
        # Generate table
        self.unigram_table = self.generate_unigram_table()

    def forward(self, center, context):
        center, context = center.squeeze(), context.squeeze()  # batch_size x embed_size

        # Compute true portion
        true_scores = (center * context).sum(-1)  # batch_size
        loss = self.criterion(true_scores, t.ones_like(true_scores))
        # test_loss = loss.detach().item()

        # Compute negatively sampled portion -
        for i in range(self.NUM_SAMPLES):
            samples = self.get_unigram_samples(n=center.shape[0])
            neg_sample_scores = (center * samples).sum(-1)
            # Update loss
            loss += self.criterion(neg_sample_scores, t.zeros_like(neg_sample_scores))

            # x3 = neg_sample_scores.clone().detach().numpy()
            # test_loss += self.bce_loss_w_logits(x3, t.zeros_like(neg_sample_scores).numpy())

        return loss  # , test_loss

    @staticmethod
    def bce_loss_w_logits(x, y):
        max_val = np.clip(x, 0, None)
        loss = x - x * y + max_val + np.log(np.exp(-max_val) + np.exp((-x - max_val)))
        return loss.mean()

    def get_unigram_samples(self, n):
        """
        Returns a sample according to a unigram distribution
        Randomly choose a value from self.unigram_table
        """
        rand_idxs = self.unigram_table.draw(n).to(self.device)
        return self.embeddings(rand_idxs).squeeze()

    def get_unigram_prob(self, token_idx):
        return (self.transformed_freq_vec[token_idx].item()) / self.freq_sum.item()

    def generate_unigram_table(self):
        # Probability at each index corresponds to probability of selecting that token
        pdf = [self.get_unigram_prob(t_idx) for t_idx in range(0, self.vocab_len)]
        # Generate the table from PDF
        return AliasMultinomial(pdf, self.device)


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
        self.x_dim = x_dim
        #if len(x_dim) > 2:  # (batch_size x !!! OPTIONAL num_sentences x seq_len x embed/vocab_size)
        #    self.batch_size = x_dim[0]

        self.seq_len = x_dim[-2]
        self.vocab_dim = x_dim[-1]

        self.linear_dims = self.calc_linear_dims()
        #self.output_padding = 0  # only used for decoder... added to right side of convtranspose
        #self.enc1 = nn.Linear(x_dim[1], nhid)
        self.relu = nn.ReLU()

        # encoder params
        self.conv1 = nn.Conv1d(self.vocab_dim, 200, self.kernel_size, stride=self.stride)
        self.conv2 = nn.Conv1d(200, 300, self.kernel_size, stride=self.stride)
        self.conv3 = nn.Conv1d(300, nhid, self.kernel_size, stride=self.stride)
        self.calc_z = nn.Linear(int(self.linear_dims[0])+ncond, 2)

        # decoder params
        self.decode_z = nn.Linear(1, self.linear_dims[0] + ncond)
        self.conv1t = nn.ConvTranspose1d(nhid, 300, self.kernel_size, stride=self.stride,
                                        padding=self.padding)
        self.conv2t = nn.ConvTranspose1d(300, 200, self.kernel_size, stride=self.stride,
                                        padding=self.padding, output_padding=1)  # getting issues syncing with encoder convs, so adding 1 output padding
        self.conv3t = nn.ConvTranspose1d(200, self.vocab_dim, self.kernel_size, stride=self.stride,
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
        x_shape = x.shape
        if len(x_shape) == 4:
            x = x.view((x.shape[0]*x.shape[1],x.shape[2],x.shape[3])) # if each sample is broken into sentences, combine batch and sentences into one dimension

        print("XHSAPE",x.shape)
        x = x.permute(0, 2, 1).float()  # batch_size*num_sentences x seq_len x vocab/embed_dim
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)

        if len(x_shape) == 4:
            x = x.view(tuple([x_shape[0]]+[x_shape[1]]+[i for i in x.shape[1:]]))
        if y is not None:
            x = torch.cat((x, y))

        mean_var = self.relu(self.calc_z(x))
        mean, var = mean_var.split(1, dim=-1)
        return mean.squeeze(), var.squeeze()

    def _decode(self, x, y=None):
        #x = x.view((x.shape[0], x.shape[1], 1))  # unflatten hidden layer
        x = x.unsqueeze(-1)
        x_shape = x.shape
        if len(x_shape) == 4:
            x = x.view((x_shape[0]*x_shape[1],x_shape[2],x_shape[3]))
        x = self.decode_z(x)
        x = self.relu(self.conv1t(x))
        x = self.relu(self.conv2t(x))
        x = self.relu(self.conv3t(x))
        x = x.permute(0,2,1)  # batch_size x vocab/embed_dim x seq_len

        if len(x_shape) == 4:
            x = x.view(tuple([x_shape[0]]+[x_shape[1]]+[i for i in x.shape[1:]]))

        if y is not None:
            x = torch.cat((x, y))

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
        self.conv1dim = self.calc_lout_encoder(self.seq_len,self.padding,self.dilation,self.kernel_size,self.stride)
        self.conv2dim = self.calc_lout_encoder(self.conv1dim,self.padding,self.dilation,self.kernel_size,self.stride)
        self.conv3dim = self.calc_lout_encoder(self.conv2dim, self.padding, self.dilation, self.kernel_size, self.stride)

        ### Decoder convolutional layers length dimension
        self.conv1tdim = int(round(self.calc_lout_decoder(int(round(self.conv3dim)),self.padding,self.dilation,self.kernel_size,self.stride,0)))
        self.conv2tdim = int(round(self.calc_lout_decoder(self.conv1tdim,self.padding,self.dilation,self.kernel_size,self.stride,1)))
        self.conv3tdim = int(round(self.calc_lout_decoder(self.conv2tdim,self.padding,self.dilation,self.kernel_size,self.stride,0)))
        self.out_pad = self.seq_len - self.conv3tdim  # using output padding, ensure the last layer reproduces the input dimension
        self.conv3tdim = self.seq_len
        return (int(round(self.conv3dim)), int(round(self.conv3tdim)))



class VAE(nn.Module):
    def __init__(self, x_dim=(512,32128), nhid=50, dec_softmax_temp=0.01, device='cpu'):
        super(VAE, self).__init__()
        self.x_dim=x_dim
        self.dim = nhid
        self.encdec = ConvolutionalEncoderDecoder(x_dim=x_dim,nhid=nhid,decoder_softmax_temp=dec_softmax_temp)
        self.device = device

    def sampling(self, mean, logvar):
        eps = torch.randn(mean.shape).to(self.device)
        sigma = 0.5 * torch.exp(logvar)
        return mean + eps * sigma

    def forward(self, x):
        mean, logvar = self.encdec(x)
        z = self.sampling(mean, logvar)
        self.encdec.mode = 'decode'
        decoded = self.encdec(z)
        self.encdec.mode = 'encode'
        return decoded, mean, logvar

    def generate(self, batch_size=None):
        z = torch.randn((batch_size, self.dim)).to(self.device) if batch_size else torch.randn((1, self.dim)).to(self.device)
        self.encdec.mode = 'decode'
        res = self.encdec(z)
        if not batch_size:
            res = res.squeeze(0)
        self.encdec.mode = 'encode'
        return res


class VAE_ProcessingIncluded(nn.Module):
    def __init__(self, tokenizer=None, x_dim=(512,32128), nhid=50, dec_softmax_temp=0.01, device='cpu'):
        super(VAE, self).__init__()
        self.x_dim=x_dim
        self.tokenizer = tokenizer
        self.seq_len = x_dim[-2]
        self.vocab_dim = x_dim[-1]

        if tokenizer is not None:
            self.inpproc = InputProcessor(tokenizer, seq_len=self.seq_len, one_hot=False)
        self.dim = nhid
        self.encdec = ConvolutionalEncoderDecoder(x_dim=x_dim,nhid=nhid,decoder_softmax_temp=dec_softmax_temp)
        self.device = device

    def sampling(self, mean, logvar):
        eps = torch.randn(mean.shape).to(self.device)
        sigma = 0.5 * torch.exp(logvar)
        return mean + eps * sigma

    def forward(self, x):
        if self.tokenizer is not None:
            x = self.inpproc(x) # convert input_ids to word embeddings

        mean, logvar = self.encdec(x)
        z = self.sampling(mean, logvar)
        self.encdec.mode = 'decode'
        decoded = self.encdec(z)
        self.encdec.mode = 'encode'
        #print(decoded.shape)
        if self.tokenizer is not None:
            self.inpproc.mode = 'decode'
            decoded = self.inpproc(decoded)
        return decoded, mean, logvar

    def generate(self, batch_size=None):
        z = torch.randn((batch_size, self.dim)).to(self.device) if batch_size else torch.randn((1, self.dim)).to(self.device)
        self.encdec.mode = 'decode'
        res = self.encdec(z)
        if not batch_size:
            res = res.squeeze(0)
        self.encdec.mode = 'encode'
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
