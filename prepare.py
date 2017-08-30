from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

#python -m gensim.scripts.glove2word2vec --input  glove.twitter.27B.50d.txt --output glove.twitter.27B.50d.w2vformat.txt
#model = gensim.models.KeyedVectors.load_word2vec_format(os.path.join(os.path.dirname(__file__), 'GoogleNews-vectors-negative300.bin'), binary=True)
#model = gensim.models.KeyedVectors.load_word2vec_format('D:\\final project\\code\\learn\\glove.twitter.27B\\glove.twitter.27B.50d.w2vformat.txt')
import nltk
import collections
import itertools
import tensorflow as tf
import numpy as np
import sys
import pickle
from input import VOCAB_SIZE, TAGS_SIZE
import random
from vocabulary import Vocabulary
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import gensim
import re
sys.path.append("D:\\final_project\\project_2\\fastqa4tackbp-master\\fastqa4tackbp-master")
def load_glove(stream, vocab=None):
    """Loads GloVe file and merges it if optional vocabulary
    Args:
        stream (iterable): An opened filestream to the GloVe file.
        vocab (dict=None): Word2idx dict of existing vocabulary.
    Returns:
        return_vocab (Vocabulary), lookup (matrix); Vocabulary contains the
                     word2idx and the matrix contains the embedded words.
    """
    word2idx = {}
    first_line = stream.readline()
    dim = len(first_line.split()) - 1
    lookup = np.empty([500000, dim], dtype=np.float)
    lookup[0] = np.fromstring(first_line.split(maxsplit=1)[1], sep=' ')
    word2idx[first_line.split(maxsplit=1)[0]] = 0
    n = 1
    for line in stream:
        word, vec = line.rstrip().split(maxsplit=1)
        if word.isalpha():
            if vocab is None or word in vocab and word not in word2idx:
                idx = len(word2idx)
                word2idx[word] = idx
                if idx > np.size(lookup, axis=0) - 1:
                    lookup.resize([lookup.shape[0] + 500000, lookup.shape[1]])
                try:
                    lookup[idx] = np.fromstring(vec, sep=' ')
                except:
                    del word2idx[word]      
        n += 1
    lookup.resize([len(word2idx), dim])
    return_vocab = Vocabulary(word2idx)   
    return return_vocab, lookup


def norm_word(word,num=15):
    i=0
    chars=num*['0']
    for x in list(word):
        chars[i]=x
        i=i+1
    return chars

def char_id():
    chars=[chr(i) for i in range(97,123)]+[chr(i) for i in range(65,91)]+['0']
    char2id={}
    id2char={}
    for i in list(range(len(chars))):
        char2id[chars[i]]=i
        id2char[i]=chars[i]
    return char2id,id2char
def judge_pure_english(keyword):  
    return all(ord(c) < 128 for c in keyword)  
'''    
with open('D:\\final project\\code\\project\\glove.840B.300d.txt', 'r',encoding='utf-8') as f:
    vocab, lookup = load_glove(f)
    char2id,id2char=char_id()
    with open('D:\\final project\\code\\text8.txt',encoding='utf-8') as f:
        corp=f.read()
        corp=list(set(corp.split()))
    word_length=15
    inputs=[]
    labels=[]
    for i in list(range(len(corp))):
        try:
            idx=vocab.word2idx[corp[i]]
            label=lookup[idx]
        except:
            continue    
        if len(list(corp[i]))<=word_length:
            input_char = norm_word(corp[i],num=word_length)
            input_char_ind=[char2id[x] for x in input_char]
            inputs.append(input_char_ind)
            labels.append(label)
'''
'''
from gensim.models import word2vec
sentences = word2vec.Text8Corpus('D:\\final_project\\code\\text8.txt')
model = word2vec.Word2Vec(sentences, size=300, window=5, min_count=5, workers=4)
x=WordSegmenter(model)
x.build_iterator('D:\\final_project\\code\\text9.txt',1,100,2,1)
x.build(learn_context_weights=False)
x.train(1,'1')
char2id,id2char=char_id()
with open('D:\\final project\\code\\text8.txt',encoding='utf-8') as f:
    corp=f.read()
    corp=list(set(corp.split()))
word_length=15
inputs=[]
labels=[]
for i in list(range(len(corp))):
    try:
        label=model.wv[corp[i]]
    except:
        continue
    
    if len(list(corp[i]))<=word_length:
        input_char = norm_word(corp[i],num=word_length)
        input_char_ind=[char2id[x] for x in input_char]
        inputs.append(input_char_ind)
        labels.append(label)
'''
model = gensim.models.KeyedVectors.load_word2vec_format('D:\\final project\\code\\learn\\glove.6B\\glove.6B.50d.w2vformat.txt')
char2id,id2char=char_id()
max_word_length=15
inputs=[]
vectors=[]
w_lengths=[]

chars=[chr(i) for i in range(97,123)]+[chr(i) for i in range(65,91)]
for i in list(range(len(model.wv.vocab))):
    word=model.wv.index2word[i]
    if judge_pure_english(word):
        if word.isalpha() and len(word)>=3:
            vector=model.wv[word]
            w_length=len(list(word))
            if w_length<=max_word_length:       
                input_char = norm_word(word,num=max_word_length)
                input_char_ind=[char2id[x] for x in input_char]
                inputs.append(input_char_ind)
                vectors.append(vector)
                w_lengths.append(w_length)
inputs=np.array(inputs)
vectors=np.array(vectors)
#model.wv.similar_by_vector(x)
#inputs, labels,word_length = build_corpus(vocab,lookup)
#del vocab
#del lookup
from gensim.models import word2vec
sentences = word2vec.Text8Corpus('D:\\final project\\code\\text8.txt')