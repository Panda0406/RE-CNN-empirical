# -*- coding:utf-8 -*-
import numpy as np
import random
import cPickle as pickle
from DNN_model import MultiWindow_CNN

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.transforms as T
import torch.optim as optim
from utils import weights_init

use_cuda = torch.cuda.is_available()
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor
ByteTensor = torch.cuda.ByteTensor if use_cuda else torch.ByteTensor

class Extractor(object):

    def __init__(self, embeddings, all_sents, all_labels, id2rela, args):
        super(Extractor, self).__init__()
        for k, v in vars(args).items(): setattr(self, k, v)
        self.args = args
        self.id2rela = id2rela

        self.embeddings = torch.from_numpy(embeddings.astype(np.float64))
        self.all_sents = all_sents.astype('int64')
        self.all_labels = all_labels

        print 'all_sents size: ', all_sents.shape

        self.train_x = self.all_sents[:-2717]
        self.train_y = self.all_labels[:-2717]
        self.test_x = self.all_sents[-2717:]
        self.test_y = self.all_labels[-2717:]

        print "env finish!"

    def train(self):
        print 'Trainingset Size', self.train_x.shape

        train_data = Data.TensorDataset(torch.LongTensor(self.train_x), torch.LongTensor(self.train_y))
        trainloader = Data.DataLoader(train_data, batch_size=self.batch_size, shuffle=True)

        test_data = Data.TensorDataset(torch.LongTensor(self.test_x), torch.LongTensor(self.test_y))
        testloader = Data.DataLoader(test_data, batch_size=500, shuffle=False)

        RE_CNN = MultiWindow_CNN(self.embeddings, self.label_num, self.args).cuda()
        RE_CNN.apply(weights_init)
        parameters = filter(lambda p: p.requires_grad, RE_CNN.parameters())

        weight_p, bias_p = [],[]
        for name, p in RE_CNN.named_parameters():
            if 'bias' in name:
                bias_p += [p]
            else:
                weight_p += [p]

        # L2 Regularization
        optimizer = optim.Adadelta([{'params': weight_p, 'weight_decay':1e-6}, 
            {'params': bias_p, 'weight_decay':0}], lr=self.lr)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.05)

        weights = [5.0] * self.args.label_num
        weights[0] = 1.0  # Give Other class low weight
        weights = torch.Tensor(weights)
        weights = weights.cuda()

        for epoch in range(self.epoch_max):
            running_loss = 0.0
            RE_CNN.train()
            scheduler.step()
            for i,data in enumerate(trainloader, 0):

                inputs, labels = data
                inputs, labels = Variable(inputs).type(LongTensor), Variable(labels).type(LongTensor)
                outputs = RE_CNN(inputs)

                loss = F.cross_entropy(outputs, labels, weight=weights)
                #print "LOSS:", loss.data[0]
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            RE_CNN.eval()
            acc_train, _ = self.calculate_acc(trainloader, RE_CNN, epoch)
            print "Epoch: %d, TRAIN_acc, %.4f, Loss, %.4f" % (epoch, acc_train, loss.data[0])

        _, answers = self.calculate_acc(testloader, RE_CNN, epoch)
        self.save_answers(answers, epoch)
        torch.save(RE_CNN, './models/RE_CNN.pkl')


    def calculate_acc(self, dataLoader, model, epoch):
        correct = 0
        total = 0
        answers = list()

        for data in dataLoader:
            inputs, labels = data
            inputs = Variable(inputs).type(LongTensor)
            labels = labels.type(LongTensor)
            logits = model(inputs)
            _, predicted = torch.max(logits.data, 1)
            total += len(predicted)
            predicted_ = predicted.cpu().numpy()
            for num in range(len(predicted)):
                answers.append(predicted_[num])
                if labels[num] == predicted[num]:
                    correct += 1

        accuracy = float(correct)/float(total)
        return accuracy, answers

    def save_answers(self, answers, epoch):
        assert len(answers) == 2717
        fans = open('./Answers/proposed_answer.txt', 'w')
        #fans = open('./Answers/proposed_answer.txt', 'w')
        for num in range(len(answers)):
            idx = 8001 + num
            fans.write(str(idx)+'\t'+self.id2rela[answers[num]]+'\n')
