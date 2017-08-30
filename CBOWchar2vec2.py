# -*- coding: utf-8 -*-
"""
Created on Sun Aug 13 14:25:56 2017

@author: YIFAN
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import numpy
from scipy.spatial.distance import cosine
from keras.layers import merge
#from keras.engine import Model
from keras.layers.core import Reshape
from keras.engine import Model
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM
from keras.layers.core import (Flatten, Dense, Activation, Lambda)
from keras.layers import Input
from keras.layers.wrappers import TimeDistributed
import keras.backend as K


from CBOWchar2vec import (Word2VecIterator, MENEvaluator,
                           infinite_cycle, LossHistory, word_to_indices)
import re, collections
import json
###############################################################################
#spelling
def words(text):
    return re.findall('[a-z]+', text.lower())
def train(features):
    model = collections.defaultdict(lambda: 1)
    for f in features:
        model[f] += 1
    return model
NWORDS = train(words(open("big.txt").read()))
alphabet = 'abcdefghijklmnopqrstuvwxyz'

def edits1(word):
   splits = [(word[:i], word[i:]) for i in range(len(word) + 1)]
   deletes = [a + b[1:] for a, b in splits if b]
   transposes = [a + b[1] + b[0] + b[2:] for a, b in splits if len(b)>1]
   replaces = [a + c + b[1:] for a, b in splits for c in alphabet if b]
   inserts = [a + c + b for a, b in splits for c in alphabet]
   return set(deletes + transposes + replaces + inserts)

def known_edits2(word):
    return set(e2 for e1 in edits1(word) for e2 in edits1(e1) if e2 in NWORDS)

def known(words):
    return set(w for w in words if w in NWORDS)

def correct(word):
    candidates = known([word]) or known(edits1(word)) or known_edits2(word) or [word]
    return max(candidates, key=NWORDS.get)

###############################################################################
#char2vec model
word_dim=300
class WordSegmenter(object):
    def __init__(self, word2vec_model):
        self.word_vectors = None
        self.word2vec_model = word2vec_model

    def build_iterator(self, corpus, n_neg, batch_size, window, min_count):
        self.iterator = Word2VecIterator(corpus, n_neg, batch_size, window,
                                         min_count, self.word2vec_model)

    def build(self, learn_context_weights=True):
        content_forward = Input(shape=(None,), dtype='int32',
                                name='content_forward')
        content_backward = Input(shape=(None,), dtype='int32',
                                 name='content_backward')
        context = Input(shape=(None,), dtype='int32', name='context')
        words= Input(shape=(1,), dtype='int32', name='word')
        
        print(1)
        if learn_context_weights:
            context_weights = None
        else:
            context_weights = [self.word2vec_model.syn1neg]
        context_embedding = Embedding(input_dim=len(self.iterator.word_index),
                                      output_dim=word_dim,
                                      weights=context_weights)
        context_embedding2 = Embedding(input_dim=len(self.iterator.word_index),
                                      output_dim=word_dim, input_length=1,
                                      weights=context_weights)
        if not learn_context_weights:
            context_embedding.trainable = False
            context_embedding2.trainable = False
#        context_sum=Flatten()(context_sum)
        context_seq_1 = LSTM(output_dim=word_dim,return_sequences=True,
                           activation='tanh')(context_embedding(context))
        rnn_context = TimeDistributed(Dense(output_dim=word_dim))(
           context_seq_1)
        context_attention_1 = TimeDistributed(Dense(output_dim=100,
                                            activation='tanh',
                                            bias=False))(rnn_context)
        context_attention_2 = TimeDistributed(Dense(output_dim=1,
                                            activity_regularizer='activity_l2',
                                            bias=False))(context_attention_1)
        '''
        def expand_dims(x):
            context_sum = K.sum(x,axis=1)
            return K.expand_dims(context_sum,dim=0)
        def expand_dims_output_shape(input_shape):
            return (input_shape[0],input_shape[2])
        context_sum = Lambda(expand_dims, expand_dims_output_shape)(context_embedding(context))
        context_sum=Flatten()(context_sum)
        '''
        print(2)
        char_embedding = Embedding(
            input_dim=29, output_dim=64, mask_zero=True)
        print(3)
        embed_forward = char_embedding(content_forward)
        embed_backward = char_embedding(content_backward)

        rnn_forward = LSTM(output_dim=word_dim, return_sequences=True,
                           activation='tanh')(embed_forward)
        backwards_lstm = LSTM(output_dim=word_dim, return_sequences=True,
                              activation='tanh', go_backwards=True)
        print(4)
        def reverse_tensor(inputs, mask):
            return inputs[:, ::-1, :]
    
        def reverse_tensor_shape(input_shapes):
            return input_shapes

        reverse = Lambda(reverse_tensor, output_shape=reverse_tensor_shape)
        reverse.supports_masking = True
        print(5)
        rnn_backward = reverse(backwards_lstm(embed_backward))
        print(6)
        rnn_bidi = TimeDistributed(Dense(output_dim=word_dim))(
            merge([rnn_forward, rnn_backward], mode='concat'))

        attention_1 = TimeDistributed(Dense(output_dim=100,
                                            activation='tanh',
                                            bias=False))(rnn_bidi)
        attention_2 = TimeDistributed(Dense(output_dim=1,
                                            activity_regularizer='activity_l2',
                                            bias=False))(attention_1)
        print(7)

        def attn_merge(inputs, mask):
            vectors = inputs[0]
            logits = inputs[1]
            # Flatten the logits and take a softmax
            logits = K.squeeze(logits, axis=2)
            pre_softmax = K.switch(mask[0], logits, -numpy.inf)
            weights = K.expand_dims(K.softmax(pre_softmax))
            return K.sum(vectors * weights, axis=1)
            
        def attn_merge_shape(input_shapes):
            return(input_shapes[0][0], input_shapes[0][2])
        def attn_merge2(inputs):
            vectors = inputs[0]
            logits = inputs[1]
            # Flatten the logits and take a softmax
            logits = K.squeeze(logits, axis=2)
            weights = K.expand_dims(K.softmax(logits))
            return K.sum(vectors * weights, axis=1)
            
        def attn_merge_shape2(input_shapes):
            return(input_shapes[0][0], input_shapes[0][2])
        print(8)
        attn = Lambda(attn_merge, output_shape=attn_merge_shape)
        attn2= Lambda(attn_merge2, output_shape=attn_merge_shape2)
        attn.supports_masking = True
        attn.compute_mask = lambda inputs, mask: None
        context_flat=attn2([rnn_context,context_attention_2])
        content_flat = attn([rnn_bidi, attention_2])

        p_emb=Dense(output_dim=word_dim)(merge([context_flat, content_flat], mode='concat'))
        print(9)
        target_emb=Flatten()(context_embedding2(words))
        output = Activation('sigmoid', name='output')(merge([p_emb,target_emb], mode='dot',dot_axes=1))

        print(10)
        model = Model(input=[content_forward, content_backward, context,words],
                      output=output)
        model.load_weights('weights.h5')
        model.compile(loss='binary_crossentropy', optimizer='adam',
                      metrics=['accuracy'])

        inputs = [content_forward, content_backward,context]
        
        self._predict = K.function(inputs, p_emb)
        self._attention = K.function(inputs, K.squeeze(attention_2, axis=2))
        self.model = model

    def train(self, epochs, model_name):
#        history = LossHistory()
#        evaluation = MENEvaluator(self)
        self.model.fit_generator(iter(infinite_cycle(self.iterator)),
                                 len(self.iterator),
                                 epochs)
#                                 callbacks=[history, evaluation])
        self.model.save_weights('weights.h5')
        '''
        plt.figure()
        plt.plot(history.total_seen, history.losses)
        plt.ylim((0.0, 0.5))
        plt.savefig(model_name + '.png')
        '''
    def predict(self, word,context_words):
        word=word.lower()
        context_words=[correct(x.lower()) for x in context_words]
        indices = word_to_indices(word)
        forward = numpy.array([[27] + indices])
        backward = numpy.array([indices + [28]])
        context=numpy.array([self.iterator.word2index[w]+1 for w in context_words])
        embedding = self._predict([forward, backward,[context]])
        attention = self._attention([forward, backward,[context]])
        return embedding, attention
        
    def predict2(self, word,context_words):
        embedding_output=K.function([self.model.layers[0].input,self.model.layers[2].input,self.model.layers[3].input],[self.model.layers[21].output])
        word=word.lower()
        context_words=[correct(x.lower()) for x in context_words]
        indices = word_to_indices(word)
        forward = numpy.array([[27] + indices])
        backward = numpy.array([indices + [28]])
        context=numpy.array([self.iterator.word2index[w]+1 for w in context_words])
        embedding = embedding_output([forward, backward,[context]])
        return embedding

    def most_similar(self,word,context_words,n=10):
        embedding = self.predict2(word,context_words)
        return  self.word2vec_model.wv.similar_by_vector(embedding[0][0],n)    

#x.model.save_weights('weights.h5')
#model =load_model('my_model.h5')       
#char_embedding :x.model.layers[1].get_weights()[0][1]
#x.model.summary()
#get_left_first_layer_output=K.function([x.model.layers[0].input,x.model.layers[2].input],[x.model.layers[21].output])
#indices = word_to_indices(word)
#forward = numpy.array([[27] + indices])
#backward = numpy.array([indices + [28]])
#layer_output = get_left_first_layer_output([forward,backward])