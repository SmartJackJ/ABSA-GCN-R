import os
import pickle
import numpy as np

pre_vector_data_name = {'glove': {'normal': 'glove.840B.300d.normal.txt',
                                  'general': 'glove.42B.300d.general.txt',
                                  'laptop': 'glove.42B.300d.laptops.txt',
                                  'restaurant': 'glove.42B.300d.restaurant.txt',
                                  'twitter': 'glove.42B.300d.twitter.txt'},
                        'word2vec': {'normal': 'word2vec.300d.normal.txt',
                                     'general': 'word2vec.300d.general.txt',
                                     'laptop': 'word2vec.300d.laptops.txt',
                                     'restaurant': 'word2vec.300d.restaurant.txt',
                                     'twitter': 'word2vec.300d.twitter.txt'}, }


def build_tokenizer(dataset_names, max_seq_len, dat_file_name):
    """
    分词结果
    读取或者生成分词结果
    分词结果本身是已经补齐或者截断的数据
    """
    if os.path.exists(dat_file_name):
        print('读取分词结果:{0}'.format(dat_file_name))
        tokenizer = pickle.load(open(dat_file_name, 'rb'))
    else:
        print("创建分词结果中.....")
        text = ''
        for dataset_name in dataset_names:
            fin = open(dataset_name, 'r', encoding='utf-8', newline='\n', errors='ignore')
            lines = fin.readlines()
            fin.close()
            for i in range(0, len(lines), 3):
                text_left, _, text_right = [s.lower().strip() for s in lines[i].partition("$T$")]
                aspect = lines[i + 1].lower().strip()
                text_raw = text_left + " " + aspect + " " + text_right
                text += text_raw + " "

        tokenizer = Tokenizer(max_seq_len)
        tokenizer.fit_on_text(text)
        pickle.dump(tokenizer, open(dat_file_name, 'wb'))
        print("分词结果创建完毕:{0}".format(dat_file_name))
    return tokenizer


def build_embedding_matrix(word2idx,
                           embed_dim,
                           dat_file_path='./input_normalize_data/',
                           pre_vector_type='glove',
                           refine_vector_type='normal',
                           dataset_name='laptop'):
    assert pre_vector_type in pre_vector_data_name.keys()
    assert refine_vector_type in pre_vector_data_name[pre_vector_type].keys()
    pre_vector_data_file = pre_vector_data_name[pre_vector_type][refine_vector_type]
    print("预训练词向量源文件:", pre_vector_data_file)
    dat_file_name = '{0}{1}_{2}_{3}_embedding_matrix.dat'.format(dat_file_path, pre_vector_type, refine_vector_type,
                                                                 dataset_name)
    print(dat_file_name)
    if os.path.exists(dat_file_name):
        print('读取嵌入层:', dat_file_name)
        embedding_matrix = pickle.load(open(dat_file_name, 'rb'))
    else:
        print('读取预训练词向量...')
        embedding_matrix = np.zeros((len(word2idx), embed_dim))  # idx 0是填充值,0向量 idx1是新词,随机初始化
        embedding_matrix[1, :] = np.random.uniform(-1 / np.sqrt(embed_dim), 1 / np.sqrt(embed_dim), (1, embed_dim))
        f_path = './datasets/word-vector/glove.300d/' \
            if pre_vector_type == 'glove' else './datasets/word-vector/word2vec.300d/'
        f_name = f_path + pre_vector_data_file
        word_vec = _load_word_vec(f_name, word2idx=word2idx, embed_dim=embed_dim)
        print('创建嵌入层矩阵:', dat_file_name)
        for word, i in word2idx.items():
            vec = word_vec.get(word)
            if vec is not None:
                # words not found in embedding index will be all-zeros.
                embedding_matrix[i] = vec
        pickle.dump(embedding_matrix, open(dat_file_name, 'wb'))
    return embedding_matrix


def _load_word_vec(path, word2idx=None, embed_dim=300):
    """
    读取预训练词向量
    :param path:
    :param word2idx:
    :param embed_dim:
    :return: {词:向量}
    """
    fin = open(path, 'r', encoding='utf-8', newline='\n', errors='ignore')
    word_vec = {}
    for line in fin:
        tokens = line.rstrip().split()
        word, vec = ' '.join(tokens[:-embed_dim]), tokens[-embed_dim:]
        if word in word2idx.keys():
            word_vec[word] = np.asarray(vec, dtype='float32')
    return word_vec


def pad_and_truncate(sequence, max_len, d_type='int64', padding='post', truncating='post', value=0):
    """
    补齐和截断
    :param sequence:
    :param max_len:
    :param d_type:
    :param padding: 补齐的方向 pre向前  post向后
    :param truncating: 截断的方向
    :param value: 填充值
    :return:返回掩码
    """
    x = (np.ones(max_len) * value).astype(d_type)
    if truncating == 'pre':
        trunc = sequence[-max_len:]
    else:
        trunc = sequence[:max_len]
    trunc = np.asarray(trunc, dtype=d_type)
    if padding == 'post':
        x[:len(trunc)] = trunc
    else:
        x[-len(trunc):] = trunc
    return x


class Tokenizer(object):
    def __init__(self, max_seq_len, lower=True):
        self.lower = lower
        self.max_seq_len = max_seq_len
        self.word2idx = {}
        self.idx2word = {}
        self.idx = 0
        self.word2idx['<pad>'] = self.idx
        self.idx2word[self.idx] = '<pad>'
        self.idx += 1
        self.word2idx['<unk>'] = self.idx
        self.idx2word[self.idx] = '<unk>'
        self.idx += 1

    def fit_on_text(self, text):
        if self.lower:
            text = text.lower()
        words = text.split()
        for word in words:
            if word not in self.word2idx:
                self.word2idx[word] = self.idx
                self.idx2word[self.idx] = word
                self.idx += 1

    def text_to_sequence(self, text, reverse=False, padding='post', truncating='post'):
        if self.lower:
            text = text.lower()
        words = text.split()
        unknown_idx = 1
        sequence = [self.word2idx[w] if w in self.word2idx else unknown_idx for w in words]
        if len(sequence) == 0:
            sequence = [0]
        if reverse:
            sequence = sequence[::-1]
        return pad_and_truncate(sequence, self.max_seq_len, padding=padding, truncating=truncating)
