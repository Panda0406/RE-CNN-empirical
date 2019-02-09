# -*- coding: utf-8 -*-
import numpy as np
import cPickle
from collections import defaultdict
import sys, re
import pandas as pd
import copy
import torch.nn.init as init

def weights_init(m):
    classname = m.__class__.__name__
    if 'Linear' in classname or 'Conv' in classname:
        init.xavier_normal(m.weight.data)
        init.constant(m.bias, 0.0)

def confine_pf(pf, max_pos):
    if pf < -max_pos:
        return -max_pos
    elif pf > max_pos:
        return max_pos
    else:
        return pf

def load_bin_vec(fname, vocab):
    """
    Loads 300x1 word embeddings w2vfile
    from https://github.com/yoonkim/CNN_sentence
    """
    word_vecs = {}
    with open(fname, "rb") as f:
        header = f.readline()
        vocab_size, layer1_size = map(int, header.split())
        binary_len = np.dtype('float32').itemsize * layer1_size
        for line in xrange(vocab_size):
            word = []
            while True:
                ch = f.read(1)
                if ch == ' ':
                    word = ''.join(word)
                    break
                if ch != '\n':
                    word.append(ch)
            if word in vocab:
               word_vecs[word] = np.fromstring(f.read(binary_len), dtype='float32')  
            else:
                f.read(binary_len)
    cPickle.dump(word_vecs, open("./data/word_vecs.pkl", "w"))
    return word_vecs

def load_word_vec(fname):
    return cPickle.load(open(fname))

def add_unknown_words(word_vecs, vocab, min_df=5000, k=300):
    """
    For words that occur in at least min_df documents, create a separate word vector.
    0.25 is chosen so the unknown vectors have (approximately) same variance as pre-trained ones
    """
    for word in vocab:
        if word not in word_vecs and vocab[word] >= min_df:
            #print word
            word_vecs[word] = np.random.uniform(-0.25,0.25,k)

def remove_UNK(instances, word_vecs):
    lengths = list()
    for num in range(len(instances)):
        sent_ = copy.deepcopy(instances[num]['text'])
        sent = list()
        for word in sent_:
            if word_vecs.has_key(word):
                sent.append(word)
        lengths.append(len(sent))
        instances[num]['text'] = " ".join(sent)
    print "SENT MAX LENGTH: ", max(lengths)

def get_W(word_vecs, vocab, k=300):
    """
    Get word matrix. W[i] is the vector for word indexed by i
    from https://github.com/yoonkim/CNN_sentence
    """
    vocab_size = len(word_vecs)
    word_idx_map = dict()
    W = np.zeros(shape=(vocab_size+2, k), dtype='float32')
    W[0] = np.zeros(k, dtype='float32') # padding vector
    i = 1
    for word in vocab:
    	if word_vecs.has_key(word):
        	W[i] = word_vecs[word]
        	word_idx_map[word] = i
        	i += 1
        else:
        	word_idx_map[word] = vocab_size+1
    W[vocab_size+1] = np.zeros(k, dtype='float32')
    return W, word_idx_map

def data2array(instances, word_idx_map, max_sent, max_pos):

    s2id_l, pf1_l, pf2_l = list(), list(), list()
    l2id_m = np.zeros(len(instances), dtype='int32')

    for num in range(len(instances)):
        sent_l = instances[num]['text'].split()
        pf1 = instances[num]['pf1']
        pf2 = instances[num]['pf2']
        l2id_m[num] = int(instances[num]['y_id'])

        sent_ids = [word_idx_map[w] for w in sent_l]

        s2id_l.append([0]*4 + sent_ids + [0]*(max_sent-len(sent_ids)))
        pf1_l.append([0]*4 + pf1 + [2*max_pos+1]*(max_sent-len(sent_ids)))
        pf2_l.append([0]*4 + pf2 + [2*max_pos+1]*(max_sent-len(sent_ids)))

    return np.array(s2id_l), l2id_m, np.array(pf1_l), np.array(pf2_l)

def clean_str(string):
    """
    from https://github.com/yoonkim/CNN_sentence
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`\.]", " ", string)     
    string = re.sub(r"\'s", " \'s", string) 
    string = re.sub(r"\'ve", " \'ve", string) 
    string = re.sub(r"n\'t", " n\'t", string) 
    string = re.sub(r"\'re", " \'re", string) 
    string = re.sub(r"\'d", " \'d", string) 
    string = re.sub(r"\'ll", " \'ll", string) 
    string = re.sub(r",", " , ", string)
    string = re.sub(r"\.", " . ", string)
    string = re.sub(r"!", " ! ", string) 
    string = re.sub(r"\(", " ( ", string) 
    string = re.sub(r"\)", " ) ", string) 
    string = re.sub(r"\?", " ? ", string) 
    string = re.sub(r"\s{2,}", " ", string)    
    return string.strip().lower() 

