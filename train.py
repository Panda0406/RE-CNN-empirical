# -*- coding: utf-8 -*-

from __future__ import division
import os,sys
from copy import deepcopy
import random
import numpy as np
from args import load_hyperparameters
from gen_data import load_data
from classifier import Extractor

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.transforms as T
import torch.optim as optim

# if gpu is to be used
use_cuda = torch.cuda.is_available()
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor
ByteTensor = torch.cuda.ByteTensor if use_cuda else torch.ByteTensor

if __name__ == "__main__":
    
    args = load_hyperparameters()
    # if gpu is to be used
    if use_cuda:
        torch.cuda.set_device(args.device)
        print "GPU is available!"
    else:
        print "GPU is not available!"

    # "./models" saves the intermediate model files
    if not os.path.exists(args.model_save_path):
        os.mkdir(args.model_save_path)

    if not os.path.exists(args.result_save_path):
        os.mkdir(args.result_save_path)

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.deterministic=True 

    print "loading data..."
    #fw2v = '../GoogleNews-vectors-negative300.bin'
    fw2v = './data/word_vecs.pkl'
    ftrain = './SemEval2010_task8_all_data/SemEval2010_task8_training/TRAIN_FILE.TXT'
    ftest = './SemEval2010_task8_all_data/SemEval2010_task8_testing_keys/TEST_FILE_FULL.TXT'
    frela = './data/rela2id'
    W_matrix, s2id_m, l2id_m, rela2id, id2rela = load_data(ftrain, ftest, frela, fw2v, args)

    Classifier = Extractor(W_matrix, s2id_m, l2id_m, id2rela, args)
    Classifier.train()
