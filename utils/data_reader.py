import pickle

import numpy as np
from torch.utils.data import Dataset

from utils.word_util import pad_and_truncate


class ABSADataset(Dataset):
    def __init__(self, f_name, tokenizer):
        fin = open(f_name, 'r', encoding='utf-8', newline='\n', errors='ignore')
        lines = fin.readlines()
        fin.close()

        fin = open(f_name + '.graph', 'rb')
        idx2graph = pickle.load(fin)
        fin.close()
        fin = open(f_name + '.head', 'rb')
        idx2head_vector = pickle.load(fin)
        fin.close()
        fin = open(f_name + '.behead', 'rb')
        idx2behead_vector = pickle.load(fin)
        fin.close()
        fin = open(f_name + '.relation', 'rb')
        idx2relation_vector = pickle.load(fin)
        fin.close()

        all_data = []
        for i in range(0, len(lines), 3):
            text_left, _, text_right = [s.lower().strip() for s in lines[i].partition("$T$")]
            aspect = lines[i + 1].lower().strip()
            polarity = lines[i + 2].strip()
            text_indices = tokenizer.text_to_sequence(text_left + " " + aspect + " " + text_right)
            context_indices = tokenizer.text_to_sequence(text_left + " " + text_right)
            left_indices = tokenizer.text_to_sequence(text_left)
            left_with_aspect_indices = tokenizer.text_to_sequence(text_left + " " + aspect)
            right_indices = tokenizer.text_to_sequence(text_right, reverse=True)
            right_with_aspect_indices = tokenizer.text_to_sequence(aspect + " " + text_right, reverse=True)
            aspect_indices = tokenizer.text_to_sequence(aspect)
            left_len = np.sum(left_indices != 0)
            aspect_len = np.sum(aspect_indices != 0)
            aspect_boundary = np.asarray([left_len, left_len + aspect_len - 1], dtype=np.int64)
            polarity = int(polarity) + 1
            text_len = np.sum(text_indices != 0)
            concat_segments_indices = [0] * (text_len + 2) + [1] * (aspect_len + 1)
            concat_segments_indices = pad_and_truncate(concat_segments_indices, tokenizer.max_seq_len)

            dependency_graph = np.pad(idx2graph[i],
                                      ((0, tokenizer.max_seq_len - idx2graph[i].shape[0]),
                                       (0, tokenizer.max_seq_len - idx2graph[i].shape[0])), 'constant')
            head_vector = pad_and_truncate(idx2head_vector[i], 250)
            behead_vector = pad_and_truncate(idx2behead_vector[i], 250)
            relation_vector = pad_and_truncate(idx2relation_vector[i], 250)

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

                'dependency_graph': dependency_graph,  # 依存图
                'head_vector': head_vector,
                'behead_vector': behead_vector,
                'relation_vector': relation_vector,

                'polarity': polarity,
            }

            all_data.append(data)
        self.data = all_data

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)
