# -*- coding: utf-8 -*-

import numpy as np
import spacy
import pickle
from spacy.tokens import Doc

semantic_relation_dic = {'self': 1, 'aux': 2, 'amod': 3, 'acl': 4, 'quantmod': 5, 'case': 6, 'nsubj': 7,
                         'neg': 8, 'intj': 9, 'cc': 10, 'nmod': 11, 'dep': 12, 'agent': 13, 'punct': 14,
                         'pcomp': 15, 'parataxis': 16, 'meta': 17, 'det': 18, 'nsubjpass': 19, 'oprd': 20,
                         'appos': 21, 'expl': 22, 'relcl': 23, 'mark': 24, 'csubj': 25, 'conj': 26, 'acomp': 27,
                         'advmod': 28, 'xcomp': 29, 'auxpass': 30, 'csubjpass': 31, 'predet': 32, 'preconj': 33,
                         'compound': 34, 'poss': 35, 'dobj': 36, 'prt': 37, 'pobj': 38, 'prep': 39, 'nummod': 40,
                         'advcl': 41, 'attr': 42, 'ccomp': 43, 'dative': 44, 'npadvmod': 45}


class WhitespaceTokenizer(object):
    def __init__(self, vocab):
        self.vocab = vocab

    def __call__(self, text):
        words = text.split()
        # All tokens 'own' a subsequent space character in this tokenizer
        spaces = [True] * len(words)
        return Doc(self.vocab, words=words, spaces=spaces)


nlp = spacy.load('en_core_web_md')
nlp.tokenizer = WhitespaceTokenizer(nlp.vocab)


def to_relation_word_message(matrix):
    # 分别返回,词头位置,词尾位置,依存关系向量
    head_position = []
    be_head_position = []
    vector = []
    for i in range(len(matrix)):
        for j in range(len(matrix[i])):
            if matrix[i][j] != 0:
                vector.append(matrix[i][j])
                if matrix[i][j] > 0:
                    head_position.append(i)
                    be_head_position.append(j)
                else:
                    head_position.append(j)
                    be_head_position.append(i)
    return np.array(head_position), np.array(be_head_position), np.abs(
        np.array(vector))


def dependency_adj_matrix(text):
    tokens = nlp(text)
    words = text.split()
    matrix = np.zeros((len(words), len(words))).astype('float32')
    assert len(words) == len(list(tokens))
    for token in tokens:
        # 自己和自己的关系
        matrix[token.i][token.i] = semantic_relation_dic['self']
        for child in token.children:
            # 支配者
            matrix[token.i][child.i] = semantic_relation_dic[child.dep_]
            # 被支配
            matrix[child.i][token.i] = -semantic_relation_dic[child.dep_]
    scale_matrix = np.where(matrix != 0, 1, matrix)
    return scale_matrix, matrix


def process(filename):
    fin = open(filename, 'r', encoding='utf-8', newline='\n', errors='ignore')
    lines = fin.readlines()
    fin.close()
    idx2graph = {}
    idx2relation = {}
    idx2head = {}
    idx2be_head = {}
    for i in range(0, len(lines), 3):
        text_left, _, text_right = [s.strip() for s in lines[i].partition("$T$")]
        aspect = lines[i + 1].strip()
        adj_matrix, relation_matrix = dependency_adj_matrix(text_left + ' ' + aspect + ' ' + text_right)
        head, be_head, relation = to_relation_word_message(relation_matrix)
        assert len(head) == len(be_head) == len(relation)
        idx2head[i] = head
        idx2be_head[i] = be_head
        idx2relation[i] = relation
        idx2graph[i] = adj_matrix
    fout = open(filename + '.graph', 'wb')
    pickle.dump(idx2graph, fout)
    fout.close()

    fout1 = open(filename + '.head', 'wb')
    pickle.dump(idx2head, fout1)
    fout1.close()

    fout2 = open(filename + '.behead', 'wb')
    pickle.dump(idx2be_head, fout2)
    fout2.close()

    fout3 = open(filename + '.relation', 'wb')
    pickle.dump(idx2relation, fout3)
    fout3.close()


if __name__ == '__main__':
    process('./datasets/acl-14-short-data/train.raw')
    process('./datasets/acl-14-short-data/test.raw')
    process('./datasets/semeval14/restaurant_train.raw')
    process('./datasets/semeval14/restaurant_test.raw')
    process('./datasets/semeval14/laptop_train.raw')
    process('./datasets/semeval14/laptop_test.raw')
    print("保存完毕")
