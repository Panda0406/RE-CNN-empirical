# -*- coding: utf-8 -*-
import numpy as np
import cPickle
import random
from collections import defaultdict
import sys, re
import copy
from utils import *
from args import load_hyperparameters

def sent_info(sent, args, rela, rela2id, exchange_entity=False, is_Training=True):
    try:
        e1 = re.search('<e1>(.*)</e1>', sent).group(1)
        e2 = re.search('<e2>(.*)</e2>', sent).group(1)
    except:
        print '####Do not find entity!!!####'

    # Entity Tags
    if not exchange_entity:
        sent = re.sub('</e1>', ' e1end ', sent)
        sent = re.sub('</e2>', ' e2end ', sent)
        sent = re.sub('<e1>', ' e1start ', sent)
        sent = re.sub('<e2>', ' e2start ', sent)
    else:
        sent = re.sub('</e1>', ' e2end ', sent)
        sent = re.sub('</e2>', ' e1end ', sent)
        sent = re.sub('<e1>', ' e2start ', sent)
        sent = re.sub('<e2>', ' e1start ', sent)

    sent = clean_str(sent)
    sent_l = sent.split()

    if len(sent_l) > args.max_sent:
        sent_l = sent_l[:args.max_sent]

    e1_pos = sent_l.index('e1start') + 1
    e2_pos = sent_l.index('e2start') + 1

    pf1 = list()
    pf2 = list()
    
    for num in range(len(sent_l)):
        pf1.append(30+confine_pf(num-e1_pos, args.max_pos))
        pf2.append(30+confine_pf(num-e2_pos, args.max_pos))

    if len(sent_l)!=len(pf1) or len(sent_l)!=len(pf2):
        print 'pf ERROR!!'

    datum = {"y": rela,
              "y_id": int(rela2id[rela]),
              "text": " ".join(sent_l),
              "entity1": e1,
              "entity2": e2,
              "pf1": pf1,
              "pf2": pf2,
              "train": is_Training,
              "num_words": len(sent_l)}

    return datum, sent_l


def load_data(ftrain, ftest, frela, fw2v, args):
    """
    transform input dataset into dictionary structure
    """
    rela2id = dict()
    id2rela = dict()
    #rela2opposite = dict()
    
    with open(frela, "r") as f:
        for line in f:
            rela = line.strip().split('\t')[0]
            idx = int(line.strip().split('\t')[1])
            rela2id[rela] = idx
            id2rela[idx] = rela

    #for idx in range(1, 10, 1):
        #rela2opposite[id2rela[idx]] = id2rela[idx+9]
        #rela2opposite[id2rela[idx+9]] = id2rela[idx]

    instances = []
    vocab = defaultdict(float)

    with open(ftrain, "r") as f:
        f = f.readlines()
        for num in range(8000):
            sent_ = f[num*4+0].strip().split('\t')[1]
            sent = sent_[1:-1] #  because of quotation mark
            rela = f[num*4+1].strip()

            item, sent_l = sent_info(sent, args, rela, rela2id, exchange_entity=False, is_Training=True)
            instances.append(item)

            words = set(sent_l)
            for word in words:
                vocab[word] += 1
    
    with open(ftest, "r") as f:
        f = f.readlines()
        for num in range(2717):
            sent_ = f[num*4+0].strip().split('\t')[1]
            sent = sent_[1:-1]
            rela = f[num*4+1].strip()

            item, sent_l = sent_info(sent, args, rela, rela2id, exchange_entity=False, is_Training=False)
            instances.append(item)

            words = set(sent_l)
            for word in words:
                vocab[word] += 1

    #print vocab['e1end'], vocab['e2end'], vocab['e2start'], vocab['e1start']
    print "number of sentences: " + str(len(instances))
    print "vocab size: " + str(len(vocab))

    print "loading word2vec vectors..."
    #w2v = load_bin_vec(fw2v, vocab)
    w2v = load_word_vec(fw2v)
    add_unknown_words(w2v, vocab, min_df=args.min_freq)
    print "word2vec loaded!"
    print "num words already in word2vec: " + str(len(w2v))

    #remove_UNK(instances, w2v)
    cPickle.dump(instances, open("./data/instances.pkl", "wb"))

    W_matrix, word_idx_map = get_W(w2v, vocab)
    cPickle.dump(word_idx_map, open("./data/word2id.pkl", "wb"))
    np.savez("./data/embedding_weights", W_matrix)

    s2id_m, l2id_m, pf1_m, pf2_m = data2array(instances, word_idx_map, args.max_sent, args.max_pos)
    #print s2id_m.shape, l2id_m.shape, pf1_m.shape, pf2_m.shape
    #print "Other Size:", len(np.where(l2id_m == 0)[0])
    np.savez("./data/trans_id", s2id_m, l2id_m)

    #print W_matrix.shape
    #print W_matrix
    #print s2id_m[11]
    #print l2id_m[11]
    #print instances[11]
    #print instances[100]
    print "dataset created!"

    return W_matrix, s2id_m, l2id_m, rela2id, id2rela


if __name__ == "__main__":

    args = load_hyperparameters()

    print "loading data..."
    #fw2v = './GoogleNews-vectors-negative300.bin'
    fw2v = './data/word_vecs.pkl'
    ftrain = './SemEval2010_task8_all_data/SemEval2010_task8_training/TRAIN_FILE.TXT'
    ftest = './SemEval2010_task8_all_data/SemEval2010_task8_testing_keys/TEST_FILE_FULL.TXT'
    frela = './data/rela2id'
    W_matrix, s2id_m, l2id_m, rela2id, id2rela = load_data(ftrain, ftest, frela, fw2v, args)

    #rela2count = defaultdict(int)
    #for item in instances:
    #    if item["train"]:
    #        rela2count[item['y']] += 1

    #for k,v in rela2count.items():
    #    print k, v
