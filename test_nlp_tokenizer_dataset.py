from unittest import TestCase
import unittest
import pandas as pd
import os
from squad_utilities import concat_text_cols, stage_squad_data_pytorch, SpacyTokenizer, SpacyDocProcessor, SpacyProcessPipeline
from squad_datasets_datamodules import NLPDataset
import numpy as np
import spacy
import pytorch_lightning as pl

SPACY_MODEL = 'en_core_web_md'

text_col1 = pd.Series(
    ['Squirrels like to jump around. I ran into a squirrel at the park the other day. It was a massive squirrel.',
     'Foxes can be very sneaky. I saw a fox roaming in the tall grass and ran away.',
     'I have heard lions are very lazy. I have never seen a lion before. Yesterday, I watched a Nat Geographic about them. The program said they were lazy.'],
    name='Text_1')
text_col2 = pd.Series(['What do you know about squirrels?',
                       'What do you know about foxes? Have you ever seen one?',
                       'What do you know about lions? They scare me!'],
                      name='Text_2')
df = pd.DataFrame([text_col1, text_col2]).T

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # dont use GPU for now
input_dir = "C:\\Users\\cfavr\\Documents\\Python Scripts\\"
device = 'CPU'
TRAIN_FILE = input_dir + 'squad/train-v2.0.json'
TEST_FILE = input_dir + 'squad/dev-v2.0.json'
TEXT_COLS = ['question','context','answer']
AGGREGATE_BY = 'sentence'

SEED=13
pl.seed_everything(SEED)
train_df, val_df = stage_squad_data_pytorch(TRAIN_FILE, TEST_FILE)
val_df = val_df.iloc[:10]
tokenizer = SpacyTokenizer(sentence_separation=True)
nlp = spacy.load(SPACY_MODEL)

class SpacyDocProcessor_Test(TestCase):
    def setUp(self):
        self.doc_processor = SpacyDocProcessor(nlp, return_values=['tokens','token_count','sentence_count',
                'running_token_count','running_sentence_count','running_embed_mean', 'running_embed_covariance'],
                 output_format='text', ent_dictionary=None)
        self.results1 = self.doc_processor(text_col1[0])
        self.results2 = self.doc_processor(text_col1[1])
        self.nlp_results = nlp(text_col1[0])

    def test_results(self):
        self.assertEqual(self.results1['token_count'],24)
        self.assertEqual(self.results2['token_count'],19)
        self.assertEqual(self.results2['running_token_count'],43)
        self.assertEqual((self.nlp_results-self.results1['running_embed_mean']).sum(),0)

class SpacyProcessPipeline_Test(TestCase):
    def setUp(self):
        self.pipeline = SpacyProcessPipeline(doc_return_values=['tokens','token_count','sentence_count'],
                             summary_return_values=['token_count','sentence_count','embed_mean','embed_covariance'],
                             output_format='integer', ent_dictionary=None, active_pipes='all', pipeline_batch_size=500)
        self.results = self.pipeline(df['Text_1'])
        val_df['+'.join(TEXT_COLS)] = concat_text_cols(val_df,TEXT_COLS)
        self.val_results = self.pipeline(val_df['+'.join(TEXT_COLS)])

    def test_return_keys(self):
        shouldreturn=  set(['tokens', 'token_count', 'sentence_count', 'global_token_count', 'global_sentence_count',
                                'global_embed_mean','global_embed_covariance'])
        missing = shouldreturn.difference(set(self.results.keys()))
        if len(missing) > 0:
            print("Missing the following results from processing: ",missing)
        self.assertEqual(len(missing),0)



    class Test_SpacyTokenizer(TestCase):
    def setUp(self):
        self.tokenizer = tokenizer

    def test_vocab_size(self):
        self.assertEqual(self.tokenizer.vocab_size, 20001)

    def test_id2token(self):
        self.assertEqual(self.tokenizer[20000], '<PAD>')
        self.assertEqual(self.tokenizer[1535], 'figured')

    def test_token2id(self):
        self.assertEqual(self.tokenizer.token2id['computer'], 793)
        self.assertEqual(self.tokenizer.token2id['mini'], 3162)
        self.assertEqual(max(self.tokenizer.token2id.values()), 19999) # it's currently missing <PAD>... might want to add

    def test_cosine_similarity(self):
        cossim = np.round(float(self.tokenizer.cosine_similarity('king','queen').detach()),4)
        self.assertEqual(cossim, 0.7253)

    def test_top_similar(self):
        self.assertEqual(self.top_similar('garbage',10), ['trashcan','rubish','waster','junks','wastepaper','crud','crap','upcycling','toters','wind-blown'])

    def test_calculate_num_sentences_doc(self):
        self.assertEqual(5, self.tokenizer.calculate_num_sentences_doc("How many sentences are there? That was one. This is now three. This sentence has an exclamation! Wow!"))

    def test_calculate_num_sentences(self):
        self.assertEqual(sum(self.tokenizer.calculate_num_sentences(df['Text_1'])), 9)
        self.assertEqual(sum(self.tokenizer.calculate_num_sentences(df['Text_2'])), 5)



class Test_NLPDataset_SentenceAgg(TestCase):
    def setUp(self):
        self.dataset = NLPDataset(df,tokenizer,text_cols=['Text_1','Text_2'],aggregate_by='sentence')


    def test_idx_starts(self):
        self.assertEqual(self.dataset.idx_starts['Text_1'], 0)
        self.assertEqual(self.dataset.idx_starts['Text_2'], 9)


    def test_num_sentences(self):
        self.assertEqual(self.dataset.df.Text_1_NumSentences,[3,2,4])
        self.assertEqual(self.dataset.df.Text_2_NumSentences,[1,2,2])


    def test_retrieve_text(self):
        self.assertEqual(self.dataset.retrieve_text(13)['Text_2'],'What do you know about lions? They scare me!')
        self.assertEqual(self.dataset.retrieve_text(0)['Text_1'],'Squirrels like to jump around. I ran into a squirrel at the park the other day. It was a massive squirrel.')


    def test_getidx(self):
        self.assertEqual(int(self.dataset[13]['Text_2_Sent_1'].sum()), 2484833)


    def test_index_text_samples(self):
        assert False

    def test_calculate_num_sentences(self):
        assert False

    def test_calculate_num_sentences_textcol(self):
        assert False

    def test_concat_text_cols(self):
        assert False


if __name__ == '__main__':
    unittest.main()