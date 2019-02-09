# -*- coding: utf-8 -*-
import random
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

# if gpu is to be used
use_cuda = torch.cuda.is_available()
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor
ByteTensor = torch.cuda.ByteTensor if use_cuda else torch.ByteTensor
Tensor = FloatTensor

class MultiWindow_CNN(nn.Module):
    
    def __init__(self, embeddings, label_num, args):
        super(MultiWindow_CNN, self).__init__()

        window_size = args.window_size
        dropout = args.dropout
        kernel_num = args.kernel_num

        self.embed = nn.Embedding(embeddings.shape[0], embeddings.shape[1])
        self.embed.weight.data.copy_(embeddings)

        self.convs1 = nn.ModuleList([nn.Conv2d(1, kernel_num, (K, embeddings.shape[1])) for K in window_size])
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(len(window_size)*kernel_num, label_num)

    def forward(self, x):
        x = self.embed(x)
        x = x.unsqueeze(1)
        x_conv = [F.relu(conv(x)).squeeze(3) for conv in self.convs1]
        x_pool = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x_conv]
        sent = torch.cat(x_pool, 1)
        sent = self.dropout(sent)
        sent = self.fc(sent)
        return sent

class MultiWindow_CNN_(nn.Module):
    """
    Add position feature
    """
    def __init__(self, embeddings, label_num, args):
        super(MultiWindow_CNN, self).__init__()

        window_size = args.window_size
        dropout = args.dropout
        kernel_num = args.kernel_num

        self.embed = nn.Embedding(embeddings.shape[0], embeddings.shape[1])
        self.embed.weight.data.copy_(embeddings)

        position_num = args.max_pos * 2 + 2
        self.embed_pf1 = nn.Embedding(position_num, args.pos_dim)
        self.embed_pf2 = nn.Embedding(position_num, args.pos_dim)

        combine_dim = embeddings.shape[1] + 2 * args.pos_dim

        self.convs1 = nn.ModuleList([nn.Conv2d(1, kernel_num, (K, combine_dim)) for K in window_size])
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(len(window_size)*kernel_num, label_num)

    def forward(self, x, pf1, pf2):
        word = self.embed(x)
        pf1 = self.embed_pf1(pf1)
        pf2 = self.embed_pf2(pf2)

        x = torch.cat((word,pf1,pf2), 2)
        x = x.unsqueeze(1)
        
        x_conv = [F.relu(conv(x)).squeeze(3) for conv in self.convs1]
        x_pool = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x_conv]

        sent = torch.cat(x_pool, 1)
        sent = self.dropout(sent)
        sent = self.fc(sent)

        return sent

