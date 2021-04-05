import torch
import torch.nn as nn

"""
循环神经网络配置类

循环神经网络能训练变长数据,按批次训练的过程中,各批次在填充至相同长度时
是按照数据集中最大序列长度填充0的(张量保存必须要所有数据长度一直)
如果填充了大量的0,训练效果会下降,因此实际过程中必须以每批次中的最大长度消除0元素
这个类除了构造相应的循环神经网络外,还自动化的按批次完成上述需求
"""


class DynamicLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1, bias=True, batch_first=True, dropout=0,
                 bidirectional=False, only_use_last_hidden_state=False, rnn_type='LSTM'):
        """
        LSTM 训练变长序列
        :param input_size: x 输入维度
        :param hidden_size: h 隐藏层维度
        :param num_layers: 层数
        :param bias: 是否使用偏置项 b_ih 和 b_hh
        :param batch_first: 输入输出的形式是否批次信息在前 若是,输入输出的torch的组成形式为(批次,序列,特征) (16,65,300)
        :param dropout: 除最后一层外,每一层的dropout概率
        :param bidirectional: 是否双向
        :param only_use_last_hidden_state: 是否只需要最后一层隐藏层特征
        :param rnn_type:{LSTM,GRU,RNN}
        """
        super(DynamicLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bias = bias
        self.batch_first = batch_first
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.only_use_last_hidden_state = only_use_last_hidden_state
        self.rnn_type = rnn_type
        self.RNN = self.__get_rnn(rnn_type,
                                  {'input_size': input_size, 'hidden_size': hidden_size, 'num_layers': num_layers,
                                   'bias': bias, 'batch_first': batch_first, 'dropout': dropout,
                                   'bidirectional': bidirectional})

    @staticmethod
    def __get_rnn(rnn_type_name, args):
        rnn_type_dic = ['LSTM', 'GRU', 'RNN']
        assert rnn_type_name in rnn_type_dic
        if rnn_type_name == 'LSTM':
            return nn.LSTM(**args)
        elif rnn_type_name == 'GRU':
            return nn.GRU(**args)
        elif rnn_type_name == 'RNN':
            return nn.RNN(**args)

    def forward(self, x, x_len, h0=None):
        """
        序列 -> 排序 -> 填充并打包 -> 使用RNN -> 解包 -> 解除排序
        :param x: 各个序列组成的向量  [3,16,300]
        :param x_len: 各序列长度  [13,15,14]
        :param h0: (h0,c0) 如果你想给每批次的隐藏状态以及记忆细胞赋予初始值,那么手动组建张量
        :return:
        """
        """ sort """
        x_sort_idx = torch.argsort(-x_len)  # 各长度从长到短分配的id
        x_unsort_idx = torch.argsort(x_sort_idx).long()
        x_len = x_len[x_sort_idx].to('cpu')  # 从长到短排序
        x = x[x_sort_idx.long()]
        x_emb_p = torch.nn.utils.rnn.pack_padded_sequence(x, x_len, batch_first=self.batch_first)
        # process using the selected RNN
        if self.rnn_type == 'LSTM':
            if h0 is None:
                out_pack, (ht, ct) = self.RNN(x_emb_p, None)
            else:
                out_pack, (ht, ct) = self.RNN(x_emb_p, (h0, h0))
        else:
            if h0 is None:
                out_pack, ht = self.RNN(x_emb_p, None)
            else:
                out_pack, ht = self.RNN(x_emb_p, h0)
            ct = None
        """unsort: h"""
        ht = torch.transpose(ht, 0, 1)[
            x_unsort_idx]  # (num_layers * num_directions, batch, hidden_size) -> (batch, ...)
        ht = torch.transpose(ht, 0, 1)

        if self.only_use_last_hidden_state:
            return ht
        else:
            """unpack: out"""
            out = torch.nn.utils.rnn.pad_packed_sequence(out_pack, batch_first=self.batch_first)  # (sequence, lengths)
            out = out[0]  #
            out = out[x_unsort_idx]
            """unsort: out c"""
            if self.rnn_type == 'LSTM':
                ct = torch.transpose(ct, 0, 1)[
                    x_unsort_idx]  # (num_layers * num_directions, batch, hidden_size) -> (batch, ...)
                ct = torch.transpose(ct, 0, 1)

            return out, (ht, ct)


if __name__ == '__main__':
    lstm_test = DynamicLSTM(input_size=10, hidden_size=5, num_layers=1)
    input_test = torch.randn(2, 3, 10)  # 批次数 长度 向量维度
    x_len_test = torch.tensor([3, 3])
    out, (ht, ct) = lstm_test(input_test, x_len_test)  # 输出,隐藏状态,记忆细胞状态
    print(out)
