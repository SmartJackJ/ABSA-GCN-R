import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.layers.dynamic_rnn import DynamicLSTM


class GraphConvolution(nn.Module):
    """
    GCN网络
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)

    def forward(self, text, adj):
        hidden = torch.matmul(text, self.weight)
        denom = torch.sum(adj, dim=2, keepdim=True) + 1
        #
        output = torch.matmul(adj, hidden) / denom
        if self.bias is not None:
            return output + self.bias
        else:
            return output


class ASRGCN(nn.Module):
    def __init__(self, embedding_matrix, opt):
        super(ASRGCN, self).__init__()
        self.opt = opt
        self.embed = nn.Embedding.from_pretrained(torch.tensor(embedding_matrix, dtype=torch.float))
        self.text_lstm = DynamicLSTM(opt.embed_dim, opt.hidden_dim, num_layers=1, batch_first=True,
                                     bidirectional=True)
        self.gc1 = GraphConvolution(2 * opt.hidden_dim, 2 * opt.hidden_dim)
        self.gc2 = GraphConvolution(2 * opt.hidden_dim, 2 * opt.hidden_dim)
        # 根据依存方向、类型 给边设置权重
        # 依存关系嵌入层  投影至词向量空间
        # 1.将两个词向量 拼接->1200维
        # 2.将依存关系嵌入 -> 应该是46维
        # 3.输出一个得分
        self.relation_embed = nn.Embedding(46, 50, padding_idx=0)
        self.bilinear = nn.Bilinear(4 * opt.hidden_dim, 50, 1)
        self.relation_embed_dropout = nn.Dropout(0.1)
        #
        self.fc = nn.Linear(2 * opt.hidden_dim, opt.polarities_dim)
        self.text_embed_dropout = nn.Dropout(0.3)

    def position_weight(self, x, aspect_double_idx, text_len, aspect_len):
        batch_size = x.shape[0]
        seq_len = x.shape[1]
        aspect_double_idx = aspect_double_idx.cpu().numpy()
        text_len = text_len.cpu().numpy()
        aspect_len = aspect_len.cpu().numpy()
        weight = [[] for i in range(batch_size)]
        for i in range(batch_size):
            context_len = text_len[i] - aspect_len[i]
            for j in range(aspect_double_idx[i, 0]):
                weight[i].append(1 - (aspect_double_idx[i, 0] - j) / context_len)
            for j in range(aspect_double_idx[i, 0], aspect_double_idx[i, 1] + 1):
                weight[i].append(0)
            for j in range(aspect_double_idx[i, 1] + 1, text_len[i]):
                weight[i].append(1 - (j - aspect_double_idx[i, 1]) / context_len)
            for j in range(text_len[i], seq_len):
                weight[i].append(0)
        weight = torch.tensor(weight, dtype=torch.float).unsqueeze(2).to(self.opt.device)
        return weight * x

    def mask(self, x, aspect_double_idx):
        batch_size, seq_len = x.shape[0], x.shape[1]
        aspect_double_idx = aspect_double_idx.cpu().numpy()
        mask = [[] for i in range(batch_size)]
        for i in range(batch_size):
            for j in range(aspect_double_idx[i, 0]):
                mask[i].append(0)
            for j in range(aspect_double_idx[i, 0], aspect_double_idx[i, 1] + 1):
                mask[i].append(1)
            for j in range(aspect_double_idx[i, 1] + 1, seq_len):
                mask[i].append(0)
        mask = torch.tensor(mask, dtype=torch.float).unsqueeze(2).to(self.opt.device)
        return mask * x

    def word_relation_word(self, text_out, adj, head_vector, behead_vector, relation_vector, indicator):
        # 拼接
        # 支配词-修饰词
        node1 = torch.gather(text_out, 1, head_vector.long())
        node1 = torch.where(indicator != 0, node1, indicator)
        node2 = torch.gather(text_out, 1, behead_vector.long())
        node2 = torch.where(indicator != 0, node2, indicator)
        word_word_vec = torch.cat((node1, node2), dim=-1)
        # 计算边权重
        edge_feature = self.relation_embed_dropout(self.relation_embed(relation_vector.long()))
        edge_score = torch.sigmoid(self.bilinear(word_word_vec, edge_feature)).squeeze(-1)
        edge_score = torch.flatten(edge_score, edge_score.dim() - 2).gather(0, index=torch.nonzero(
            torch.flatten(relation_vector, relation_vector.dim() - 2)).T.squeeze(dim=0))
        # 边权重更新
        adj_flatten = adj.flatten(adj.dim() - 2)
        non_zero_index = torch.split(torch.nonzero(adj_flatten).T, 1, dim=0)
        adj_flatten.index_put_(non_zero_index, edge_score)
        adj = adj_flatten.reshape(adj.shape)
        return adj

    def forward(self, inputs):
        text_indices, aspect_indices, left_indices, adj, head_vector, behead_vector, relation_vector = inputs
        text_len = torch.sum(text_indices != 0, dim=-1)
        aspect_len = torch.sum(aspect_indices != 0, dim=-1)
        left_len = torch.sum(left_indices != 0, dim=-1)
        aspect_double_idx = torch.cat([left_len.unsqueeze(1), (left_len + aspect_len - 1).unsqueeze(1)], dim=1)
        text = self.embed(text_indices)
        text = self.text_embed_dropout(text)
        """
        双向lstm编码上下文
        """
        text_out, (_, _) = self.text_lstm(text, text_len)
        """
        边权重计算
        节点的边权重在0-1之间
        """
        seq_len = text_out.shape[1]
        adj = adj[:, :seq_len, :seq_len]
        relation_len = max(torch.count_nonzero(relation_vector, dim=1))
        head_vector = head_vector[:, :relation_len].unsqueeze(dim=-1).expand(-1, -1, 2 * self.opt.hidden_dim)
        behead_vector = behead_vector[:, :relation_len].unsqueeze(dim=-1).expand(-1, -1, 2 * self.opt.hidden_dim)
        relation_vector = relation_vector[:, :relation_len]
        # 指示器
        indicator = torch.where(relation_vector != 0, 1.0, 0.0).unsqueeze(dim=-1).expand(-1, -1, 2 * self.opt.hidden_dim)
        adj = self.word_relation_word(text_out, adj, head_vector, behead_vector, relation_vector, indicator)
        # print(adj)
        """
        图卷积
        输入 ： 文本特征,依存句法图矩阵
        输出 :  聚合邻居节点后的高阶节点特征
        """
        x = F.relu(self.gc1(self.position_weight(text_out, aspect_double_idx, text_len, aspect_len), adj), inplace=True)
        x = F.relu(self.gc2(self.position_weight(x, aspect_double_idx, text_len, aspect_len), adj), inplace=True)
        """
        注意力机制
        将属性的高阶节点特征与上下文做注意力机制,点积
        """
        x = self.mask(x, aspect_double_idx)
        alpha_mat = torch.matmul(x, text_out.transpose(1, 2))
        b = alpha_mat.sum(1, keepdim=True)  # 直接求和
        alpha = F.softmax(b, dim=2)
        # print("注意力得分：",alpha)
        x = torch.matmul(alpha, text_out).squeeze(1)  # batch_size x 2*hidden_dim
        output = self.fc(x)
        return output
