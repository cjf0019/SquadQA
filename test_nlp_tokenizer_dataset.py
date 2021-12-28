from unittest import TestCase
import pandas as pd
from squad_utilities import SpacyTokenizer
from squad_datasets_datamodules import NLPDataset
import numpy as np

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
tokenizer = SpacyTokenizer(sentence_separation=True)


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


    def test_retrieve_text(self):
        assert False

    def test_index_text_samples(self):
        assert False

    def test_calculate_num_sentences(self):
        assert False

    def test_calculate_num_sentences_textcol(self):
        assert False

    def test_concat_text_cols(self):
        assert False



