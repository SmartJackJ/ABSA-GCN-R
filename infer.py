# -*- coding: utf-8 -*-

import os
import pickle
import torch
import torch.nn.functional as F
import argparse
import numpy as np
from utils import data_reader, word_util
from models import ASGCN, ASRGCN
from dependency_graph import dependency_adj_matrix, to_relation_word_message


class Inferer:
    """A simple inference example"""

    def __init__(self, opt):
        self.opt = opt
        self.tokenizer = word_util.build_tokenizer(
            dataset_names=[opt.dataset_file['train'], opt.dataset_file['test']],
            max_seq_len=opt.max_seq_len,
            dat_file_name='{0}_tokenizer.dat'.format(opt.input_normalize_data_path + opt.dataset))
        embedding_matrix = word_util.build_embedding_matrix(
            word2idx=self.tokenizer.word2idx,
            embed_dim=opt.embed_dim,
            dat_file_path=opt.input_normalize_data_path,
            pre_vector_type=opt.pre_vector_type,
            refine_vector_type=opt.refine_vector_type,
            dataset_name=opt.dataset)
        self.model = opt.model_class(embedding_matrix, opt)
        print('loading model {0} ...'.format(opt.model_name))
        self.model.load_state_dict(torch.load(opt.state_dict_path))
        self.model = self.model.to(opt.device)
        # 设置为评估模式
        self.model.eval()
        torch.autograd.set_grad_enabled(False)

    def evaluate(self, text, aspect):
        aspect = aspect.lower().strip()
        text_left, _, text_right = [s.strip() for s in text.lower().partition(aspect)]

        text_indices = self.tokenizer.text_to_sequence(text_left + " " + aspect + " " + text_right)
        context_indices = self.tokenizer.text_to_sequence(text_left + " " + text_right)
        left_indices = self.tokenizer.text_to_sequence(text_left)
        left_with_aspect_indices = self.tokenizer.text_to_sequence(text_left + " " + aspect)
        right_indices = self.tokenizer.text_to_sequence(text_right, reverse=True)
        right_with_aspect_indices = self.tokenizer.text_to_sequence(aspect + " " + text_right, reverse=True)
        aspect_indices = self.tokenizer.text_to_sequence(aspect)
        left_len = np.sum(left_indices != 0)
        aspect_len = np.sum(aspect_indices != 0)
        aspect_boundary = np.asarray([left_len, left_len + aspect_len - 1], dtype=np.int64)

        text_len = np.sum(text_indices != 0)
        concat_segments_indices = [0] * (text_len + 2) + [1] * (aspect_len + 1)
        concat_segments_indices = word_util.pad_and_truncate(concat_segments_indices, self.tokenizer.max_seq_len)

        dependency_graph = dependency_adj_matrix(text)[0]
        head_vector, behead_vector, relation_vector = to_relation_word_message(dependency_graph)

        data = {
            'concat_segments_indices': concat_segments_indices,
            'text_indices': text_indices,
            'context_indices': context_indices,
            'left_indices': left_indices,
            'left_with_aspect_indices': left_with_aspect_indices,
            'right_indices': right_indices,
            'right_with_aspect_indices': right_with_aspect_indices,
            'aspect_indices': aspect_indices,
            'aspect_boundary': aspect_boundary,
            'dependency_graph': dependency_graph,
            'head_vector': head_vector,
            'behead_vector': behead_vector,
            'relation_vector': relation_vector
        }

        t_inputs = [torch.tensor([data[col]], device=self.opt.device) for col in self.opt.input_controller]
        t_outputs = self.model(t_inputs)
        t_probs = F.softmax(t_outputs, dim=-1).cpu().numpy()

        return t_probs


if __name__ == '__main__':
    model_classes = {
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
        'asgcn': ['text_indices', 'aspect_indices', 'left_indices', 'dependency_graph'],
        'asrgcn': ['text_indices', 'aspect_indices', 'left_indices', 'dependency_graph', 'head_vector',
                   'behead_vector', 'relation_vector']
    }


    class Option(object): pass


    opt = Option()
    opt.model_name = 'asrgcn'
    opt.dataset = 'laptop'
    opt.pre_vector_type = 'glove'
    opt.refine_vector_type = 'normal'
    opt.input_normalize_data_path = "./input_normalize_data/"
    opt.model_class = model_classes[opt.model_name]
    opt.dataset_file = dataset_files[opt.dataset]
    opt.input_controller = input_controller[opt.model_name]
    # set your trained models here
    opt.state_dict_path = './result/asrgcn/asrgcn_laptop_val_acc_0.7712_f1_0.7312'
    opt.embed_dim = 300
    opt.hidden_dim = 300
    opt.max_seq_len = 8
    opt.polarities_dim = 3
    opt.hops = 3
    opt.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    opt.local_context_focus = 'cdm'
    opt.SRD = 3

    inf = Inferer(opt)
    t_probs = inf.evaluate(
        'I charge it at night and skip taking the cord with me because of the good battery life .',
        'battery life')
    print(t_probs.argmax(axis=-1) - 1)
