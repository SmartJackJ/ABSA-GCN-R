# -*- coding: utf-8 -*-
import logging
import os
import math
import argparse
import random
import sys
import time
from time import strftime, localtime

import numpy
import torch
import torch.nn as nn
from sklearn import metrics
from torch.utils.data import random_split, DataLoader

from models import ASGCN, ASRGCN, LSTM, TD_LSTM, ATAE_LSTM, MemNet, AOA, IAN
from utils import word_util
from utils.data_reader import ABSADataset

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler(sys.stdout))


class Instructor:
    def __init__(self, opt):
        self.opt = opt
        tokenizer = word_util.build_tokenizer(
            dataset_names=[opt.dataset_file['train'], opt.dataset_file['test']],
            max_seq_len=opt.max_seq_len,
            dat_file_name='{0}_tokenizer.dat'.format(opt.input_normalize_data_path + opt.dataset))
        embedding_matrix = word_util.build_embedding_matrix(
            word2idx=tokenizer.word2idx,
            embed_dim=opt.embed_dim,
            dat_file_path=opt.input_normalize_data_path,
            pre_vector_type=opt.pre_vector_type,
            refine_vector_type=opt.refine_vector_type,
            dataset_name=opt.dataset)
        self.model = opt.model_class(embedding_matrix, opt).to(opt.device)
        # 通过数据集读取器和分词器读取数据
        self.train_set = ABSADataset(opt.dataset_file['train'], tokenizer)
        self.test_set = ABSADataset(opt.dataset_file['test'], tokenizer)
        # 确认是否需要验证集,否则将训练集当验证集
        assert 0 <= opt.val_set_ratio < 1
        if opt.val_set_ratio > 0:
            val_set_len = int(len(self.train_set) * opt.val_set_ratio)
            self.train_set, self.val_set = random_split(self.train_set,
                                                        (len(self.train_set) - val_set_len, val_set_len))
        else:
            self.val_set = self.test_set
        # 打印各种参数
        if opt.device.type == 'cuda':
            logger.info('cuda memory allocated: {}'.format(torch.cuda.memory_allocated(device=opt.device.index)))
        self.global_f1 = 0.
        self._print_args()

    def _print_args(self):
        n_trainable_params, n_non_trainable_params = 0, 0
        for p in self.model.parameters():
            n_params = torch.prod(torch.tensor(p.shape))
            if p.requires_grad:
                n_trainable_params += n_params
            else:
                n_non_trainable_params += n_params
        logger.info(
            '> 训练参数数量: {0}, 未训练参数数量: {1}'.format(n_trainable_params, n_non_trainable_params))
        logger.info('> 训练参数:')
        for arg in vars(self.opt):
            logger.info('>>> {0}: {1}'.format(arg, getattr(self.opt, arg)))

    def _reset_params(self):
        for child in self.model.children():
            # if type(child) != BertModel:  # skip bert params
            for p in child.parameters():
                if p.requires_grad:
                    if len(p.shape) > 1:
                        self.opt.initializer(p)
                    else:
                        stdv = 1. / math.sqrt(p.shape[0])
                        torch.nn.init.uniform_(p, a=-stdv, b=stdv)

    def _train(self, criterion, optimizer, train_data_loader, val_data_loader):
        max_val_acc = 0
        relation_val_f1 = 0
        max_val_f1 = 0
        relation_val_acc = 0
        global_step = 0
        continue_not_increase = 0
        for i_epoch in range(self.opt.num_epoch):
            logger.info('>' * 100)
            logger.info('epoch: {}'.format(i_epoch))
            n_correct, n_total, loss_total = 0, 0, 0
            increase_flag = False
            for i_batch, batch in enumerate(train_data_loader):
                # switch model to training mode
                self.model.train()
                global_step += 1
                # clear gradient accumulators
                optimizer.zero_grad()
                inputs = [batch[col].to(self.opt.device) for col in self.opt.input_controller]
                outputs = self.model(inputs)
                targets = batch['polarity'].to(self.opt.device)

                loss = criterion(outputs, targets)
                loss.backward()

                optimizer.step()

                if global_step % self.opt.log_step == 0:
                    n_correct += (torch.argmax(outputs, -1) == targets).sum().item()
                    n_total += len(outputs)
                    loss_total += loss.item() * len(outputs)
                    train_acc = n_correct / n_total
                    train_loss = loss_total / n_total
                    val_acc, val_f1 = self._evaluate_acc_f1(val_data_loader)
                    logger.info(
                        'loss: {:.4f}, acc: {:.4f}, val_acc:{:.4f}, val_f1: {:.4f}'.format(train_loss, train_acc,
                                                                                           val_acc, val_f1))
                    if val_acc > max_val_acc:
                        max_val_acc = val_acc
                        relation_val_f1 = val_f1
                    if val_f1 > max_val_f1:
                        increase_flag = True
                        max_val_f1 = val_f1
                        relation_val_acc = val_acc
                        if val_f1 > self.global_f1:
                            self.global_f1 = val_f1
                            if not os.path.exists('result'):
                                os.mkdir('result')
                            path = 'result/{0}_{1}_val_acc_{2}_f1_{3}'.format(self.opt.model_name, self.opt.dataset,
                                                                              round(val_acc, 4), round(val_f1, 4))
                            # torch.save(self.model.state_dict(), path)
                            # logger.info('>> saved: {}'.format(path))

            logger.info('> 最大验证集准确率: {:.4f}, 对应的宏平均: {:.4f}'.format(max_val_acc, relation_val_f1))
            logger.info('> 最大验证集宏平均: {:.4f}, 对应的准确率: {:.4f}'.format(max_val_f1, relation_val_acc))
            if not increase_flag:
                continue_not_increase += 1
                if continue_not_increase >= self.opt.patience:
                    logger.info('early stop.')
                    break
            else:
                continue_not_increase = 0

        return max_val_acc, max_val_f1

    def _evaluate_acc_f1(self, data_loader):
        n_correct, n_total = 0, 0
        t_targets_all, t_outputs_all = None, None
        # switch model to evaluation mode
        self.model.eval()
        with torch.no_grad():
            for i_batch, t_batch in enumerate(data_loader):
                t_inputs = [t_batch[col].to(self.opt.device) for col in self.opt.input_controller]
                t_targets = t_batch['polarity'].to(self.opt.device)
                t_outputs = self.model(t_inputs)
                n_correct += (torch.argmax(t_outputs, -1) == t_targets).sum().item()
                n_total += len(t_outputs)

                if t_targets_all is None:
                    t_targets_all = t_targets
                    t_outputs_all = t_outputs
                else:
                    t_targets_all = torch.cat((t_targets_all, t_targets), dim=0)
                    t_outputs_all = torch.cat((t_outputs_all, t_outputs), dim=0)
        acc = n_correct / n_total
        f1 = metrics.f1_score(t_targets_all.cpu(), torch.argmax(t_outputs_all, -1).cpu(), labels=[0, 1, 2],
                              average='macro')
        return acc, f1

    def run(self, repeats=3):
        # Loss and Optimizer
        criterion = nn.CrossEntropyLoss()
        max_test_acc_avg = 0
        max_test_f1_avg = 0
        for i in range(repeats):
            logger.info('repeat:{0}'.format(i + 1))
            self._reset_params()
            _params = filter(lambda p: p.requires_grad, self.model.parameters())
            optimizer = self.opt.optimizer(_params, lr=self.opt.lr, weight_decay=self.opt.l2reg)

            train_data_loader = DataLoader(dataset=self.train_set, batch_size=self.opt.batch_size, shuffle=True)
            test_data_loader = DataLoader(dataset=self.test_set, batch_size=self.opt.batch_size, shuffle=False)
            val_data_loader = DataLoader(dataset=self.val_set, batch_size=self.opt.batch_size, shuffle=False)
            max_val_acc, max_val_f1 = self._train(criterion, optimizer, train_data_loader, val_data_loader)
            logger.info('max_test_acc: {0}     max_test_f1: {1}'.format(max_val_acc, max_val_f1))
            max_test_acc_avg += max_val_acc
            max_test_f1_avg += max_val_f1
            logger.info('#' * 100)
        logger.info(
            '>> 最大_测试集平均准确率: {:.4f}, 最大_测试集平均宏平均: {:.4f}'.format(max_test_acc_avg / repeats, max_test_f1_avg / repeats))


def main():
    # 聚合参数
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', default='asrgcn', type=str)
    parser.add_argument('--dataset', default='laptop', type=str, help='twitter, restaurant, laptop')
    parser.add_argument('--input_normalize_data_path', default='./input_normalize_data/')  # 中间数据保存地址
    parser.add_argument('--pre_vector_type', default='glove')  # 词向量类型
    parser.add_argument('--refine_vector_type', default='general', help='normal,general,twitter, restaurant, laptop')
    parser.add_argument('--optimizer', default='adam', type=str)
    parser.add_argument('--initializer', default='xavier_uniform_', type=str)
    parser.add_argument('--lr', default=0.001, type=float, help='try 5e-5, 2e-5 for BERT, 1e-3 for others')
    parser.add_argument('--dropout', default=0.1, type=float)
    parser.add_argument('--l2reg', default=0.00001, type=float)
    parser.add_argument('--num_epoch', default=30, type=int, help='try larger number for non-BERT models')
    parser.add_argument('--batch_size', default=16, type=int, help='try 16, 32, 64 for BERT models')
    parser.add_argument('--log_step', default=5, type=int)
    parser.add_argument('--embed_dim', default=300, type=int)
    parser.add_argument('--hidden_dim', default=300, type=int)
    parser.add_argument('--max_seq_len', default=85, type=int)
    parser.add_argument('--polarities_dim', default=3, type=int)
    parser.add_argument('--hops', default=3, type=int)
    parser.add_argument('--patience', default=6, type=int)
    parser.add_argument('--device', default="cuda:0", type=str, help='e.g. cuda:0')
    parser.add_argument('--seed', default=1314, type=int, help='set seed for reproducibility')
    parser.add_argument('--val_set_ratio', default=0, type=float, help='ratio between 0 and 1 for validation support')

    opt = parser.parse_args()

    if opt.seed is not None:
        random.seed(opt.seed)
        numpy.random.seed(opt.seed)
        torch.manual_seed(opt.seed)
        torch.cuda.manual_seed(opt.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        os.environ['PYTHONHASHSEED'] = str(opt.seed)

    model_classes = {
        'lstm': LSTM,
        'td_lstm': TD_LSTM,
        'atae_lstm': ATAE_LSTM,
        'aoa': AOA,
        'ian': IAN,
        'memnet': MemNet,
        'asgcn': ASGCN,
        'asrgcn': ASRGCN,
    }
    dataset_files = {
        'twitter': {
            'train': './datasets/acl-14-short-data/train.raw',
            'test': './datasets/acl-14-short-data/test.raw'
        },
        'restaurant': {
            'train': './datasets/semeval14/restaurant_train.raw',
            'test': './datasets/semeval14/restaurant_test.raw'
        },
        'laptop': {
            'train': './datasets/semeval14/laptop_train.raw',
            'test': './datasets/semeval14/laptop_test.raw'
        }
    }
    input_controller = {
        'lstm': ['text_indices'],
        'td_lstm': ['left_with_aspect_indices', 'right_with_aspect_indices'],
        'atae_lstm': ['text_indices', 'aspect_indices'],
        'memnet': ['context_indices', 'aspect_indices'],
        'aoa': ['text_indices', 'aspect_indices'],
        'ian': ['text_indices', 'aspect_indices'],
        'asgcn': ['text_indices', 'aspect_indices', 'left_indices', 'dependency_graph'],
        'asrgcn': ['text_indices', 'aspect_indices', 'left_indices', 'dependency_graph', 'head_vector',
                   'behead_vector', 'relation_vector']
    }
    initializers = {
        'xavier_uniform_': torch.nn.init.xavier_uniform_,
        'xavier_normal_': torch.nn.init.xavier_normal_,
        'orthogonal_': torch.nn.init.orthogonal_,
    }
    optimizers = {
        'adadelta': torch.optim.Adadelta,  # default lr=1.0
        'adagrad': torch.optim.Adagrad,  # default lr=0.01
        'adam': torch.optim.Adam,  # default lr=0.001
        'adamax': torch.optim.Adamax,  # default lr=0.002
        'asgd': torch.optim.ASGD,  # default lr=0.01
        'rmsprop': torch.optim.RMSprop,  # default lr=0.01
        'sgd': torch.optim.SGD,
    }

    # 参数载入

    opt.model_class = model_classes[opt.model_name]
    opt.dataset_file = dataset_files[opt.dataset]
    opt.input_controller = input_controller[opt.model_name]
    opt.initializer = initializers[opt.initializer]

    opt.optimizer = optimizers[opt.optimizer]
    opt.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # log_file = './result/log/{}/{}-{}-{}.log'.format(opt.model_name, opt.model_name, opt.dataset,
    #                                                  strftime("%y%m%d-%H%M", localtime()))
    # logger.addHandler(logging.FileHandler(log_file, encoding='utf-8'))

    ins = Instructor(opt)
    ins.run()


if __name__ == '__main__':
    main()
