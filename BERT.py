import os
import random
import torch
from torch import nn
from d2l import torch as d2l
from visdom import Visdom


def get_tokens_and_segments(token_a, token_b=None):
    tokens = ['<cls>'] + token_a + ['<sep>']
    segments = [0] * (len(token_a) + 2)
    if token_b is not None:
        tokens += token_b + ['<sep>']
        segments += [1] * (len(token_b) + 1)
    return tokens, segments


# 包括token_embedding, segment_embedding, position_embedding和Encoder模块
# num_hiddens = norm_shape = ffn_num_input = key_size = value_size = query_size in practice
class BERTEncoder(nn.Module):
    def __init__(self, vocab_size, num_hiddens, norm_shape, ffn_num_input, ffn_num_hiddens, num_heads,
                 num_layers, dropout, max_len=1000, key_size=768, query_size=768, value_size=768, **kwargs):
        super(BERTEncoder, self).__init__(**kwargs)
        self.token_embedding = nn.Embedding(vocab_size, num_hiddens)
        self.segment_embedding = nn.Embedding(2, num_hiddens)
        self.position_embedding = nn.Parameter(torch.rand(1, max_len, num_hiddens))  # 返回正太分布
        self.blks = nn.Sequential()
        for i in range(num_layers):
            self.blks.add_module(f'{i}', EncoderBlock(
                key_size, query_size, value_size, num_hiddens, norm_shape,
                ffn_num_input, ffn_num_hiddens, num_heads=num_heads, dropout=dropout, use_bias=True
            ))

    def forward(self, tokens, segments, valid_lens):
        x = self.token_embedding(tokens) + self.segment_embedding(segments)
        x += self.position_embedding.data[:, :x.shape[1], :]
        for blk in self.blks:
            x = blk(x, valid_lens)
        return x


# 包括MultiAttention, PositionWiseFFN和AddNorm
class EncoderBlock(nn.Module):
    def __init__(self, key_size, query_size, value_size, num_hiddens, norm_shape,
                 ffn_num_input, ffn_num_hiddens, num_heads, dropout, use_bias=False, **kwargs):
        super(EncoderBlock, self).__init__(**kwargs)
        self.attention = MultiHeadAttention(key_size, query_size, value_size, num_hiddens, num_heads, dropout, use_bias)
        self.addnorm1 = AddNorm(norm_shape, dropout)
        self.ffn = PositionWiseFFN(ffn_num_input, ffn_num_hiddens, num_hiddens)
        self.addnorm2 = AddNorm(norm_shape, dropout)

    def forward(self, X, valid_len):
        Y = self.addnorm1(X, self.attention(X, X, X, valid_len))
        return self.addnorm2(Y, self.ffn(Y))


class PositionWiseFFN(nn.Module):
    def __init__(self, ffn_num_input, ffn_num_hiddens, ffn_num_output, **kwargs):
        super(PositionWiseFFN, self).__init__(**kwargs)
        self.dense1 = nn.Linear(ffn_num_input, ffn_num_hiddens)
        self.relu = nn.ReLU()
        self.dense2 = nn.Linear(ffn_num_hiddens, ffn_num_output)

    def forward(self, X):
        return self.dense2(self.relu(self.dense1(X)))


class AddNorm(nn.Module):
    def __init__(self, normalized_shape, dropout, **kwargs):
        super(AddNorm, self).__init__(**kwargs)
        self.dropout = nn.Dropout(dropout)
        self.ln = nn.LayerNorm(normalized_shape)  # 层归一化， 输入参数为需要归一化的data_shape->均值0方差1

    def forward(self, X, Y):
        return self.ln(self.dropout(Y) + X)


class MultiHeadAttention(nn.Module):
    def __init__(self, key_size, query_size, value_size, num_hiddens, num_heads, droput, bias=False, **kwargs):
        super(MultiHeadAttention, self).__init__(**kwargs)
        self.num_heads = num_heads
        self.attention = d2l.DotProductAttention(droput)
        self.W_q = nn.Linear(query_size, num_hiddens, bias=bias)
        self.W_k = nn.Linear(key_size, num_hiddens, bias=bias)
        self.W_v = nn.Linear(value_size, num_hiddens, bias=bias)
        self.W_o = nn.Linear(num_hiddens, num_hiddens, bias=bias)

    def forward(self, queries, keys, values, valid_lens):
        queries = transpose_qkv(self.W_q(queries), self.num_heads)
        keys = transpose_qkv(self.W_q(keys), self.num_heads)
        values = transpose_qkv(self.W_q(values), self.num_heads)

        # input_shape = (batch_size, len), output_shape = (batch_size * num_head, len)
        if valid_lens is not None:
            valid_lens = torch.repeat_interleave(valid_lens, repeats=self.num_heads, dim=0)

        output = self.attention(queries, keys, values, valid_lens)
        output = transpose_output(output, self.num_heads)
        return self.W_o(output)


# input_shape = (batch_size, len, num_hiddens)
# output_shape = (batch_size * num_heads, len, num_hiddens / num_heads)
def transpose_qkv(X, num_heads):
    X = X.reshape(X.shape[0], X.shape[1], num_heads, -1)
    X = X.permute(0, 2, 1, 3)
    return X.reshape(-1, X.shape[2], X.shape[3])


# reverse of transpose_qkv
def transpose_output(X, num_heads):
    X = X.reshape(-1, num_heads, X.shape[1], X.shape[2])
    X = X.permute(0, 2, 1, 3)
    return X.reshape(X.shape[0], X.shape[1], -1)


class MaskLM(nn.Module):
    def __init__(self, vocab_size, num_hiddens, num_inputs=768, **kwargs):
        super(MaskLM, self).__init__(**kwargs)
        self.mlp = nn.Sequential(nn.Linear(num_inputs, num_hiddens),
                                 nn.ReLU(),
                                 nn.LayerNorm(num_hiddens),
                                 nn.Linear(num_hiddens, vocab_size))

    # pred_position input = (batch_size, num_pred_per_sentence)
    # output shape=(batch_size, num_pred, vocab_size)
    def forward(self, X, pred_positions):
        num_pred = pred_positions.shape[1]
        batch_size = pred_positions.shape[0]
        pred_positions = pred_positions.reshape(-1)
        batch_idx = torch.arange(0, batch_size)
        batch_idx = torch.repeat_interleave(batch_idx, num_pred)
        # 对X进行元素索引，得到需要预测的所有mask_X shape=(num_pred, num_hiddens)
        masked_x = X[batch_idx, pred_positions]
        masked_x = masked_x.reshape((batch_size, num_pred, -1))
        mlm_Y_hat = self.mlp(masked_x)
        return mlm_Y_hat


class NextSentencePred(nn.Module):
    def __init__(self, num_inputs, **kwargs):
        super(NextSentencePred, self).__init__(**kwargs)
        self.output = nn.Linear(num_inputs, 2)

    def forward(self, X):
        return self.output(X)


class BERTModel(nn.Module):
    def __init__(self, vocab_size, num_hiddens, norm_shape, ffn_num_input, ffn_num_hiddens, num_heads,
                 num_layers, dropout, max_len=1000, key_size=768, query_size=768, value_size=768,
                 hid_in_features=768, mlm_in_features=768, nsp_in_feature=768):
        super(BERTModel, self).__init__()
        self.encoder = BERTEncoder(vocab_size, num_hiddens, norm_shape, ffn_num_input, ffn_num_hiddens, num_heads,
                                   num_layers, dropout, max_len=max_len, key_size=key_size, query_size=query_size, value_size=value_size)
        self.hiddens = nn.Sequential(nn.Linear(hid_in_features, num_hiddens), nn.Tanh())
        self.mlm = MaskLM(vocab_size, num_hiddens, mlm_in_features)
        self.nsp = NextSentencePred(nsp_in_feature)

    # tokens is sentences
    def forward(self, tokens, segments, valid_len=None, pred_position=None):
        encoded_X = self.encoder(tokens, segments, valid_len)
        if pred_position is not None:
            mlm_Y_hat = self.mlm(encoded_X, pred_position)
        else:
            mlm_Y_hat = None
        nsp_Y_hat = self.nsp(self.hiddens(encoded_X[:, 0, :]))
        return encoded_X, mlm_Y_hat, nsp_Y_hat


def get_loss(net, loss, vocab_size, tokens_X, segments_X, valid_lens_X, pred_positions_X,
             mlm_weights, mlm_Y, nsp_Y):
    _, mlm_Y_hat, nsp_Y_hat = net(tokens_X, segments_X, valid_lens_X.reshape(-1), pred_positions_X)
    # 填充词元<pad>的预测将通过乘以权重0过滤掉
    mlm_l = loss(mlm_Y_hat.reshape(-1, vocab_size), mlm_Y.reshape(-1)) * mlm_weights.reshape(-1, 1)
    mlm_l = mlm_l.sum() / (mlm_weights.sum() + 1e-8)
    nsp_l = loss(nsp_Y_hat, nsp_Y)
    l = mlm_l + nsp_l
    return mlm_l, nsp_l, l


def train_bert(train_iter, net, loss, vocab_size, num_steps):
    trainer = torch.optim.Adam(net.parameters(), lr=0.1)
    step = 0
    timer = d2l.Timer()
    metric = d2l.Accumulator(4)
    vis = Visdom(env='train')
    vis.line([[0., 0.]], [0.], win='train_org', opts=dict(showlegend=True, legend=['mlm_l', 'nsp_l']))
    num_step_reached = False

    while step < num_steps and not num_step_reached:
        for tokens_X, segments_X, valid_lens_X, pred_positions_X,\
                 mlm_weights_X, mlm_Y, nsp_Y in train_iter:
            trainer.zero_grad()
            timer.start()
            mlm_l, nsp_l, l = get_loss(net, loss, vocab_size, tokens_X, segments_X, valid_lens_X,
                                       pred_positions_X, mlm_weights_X, mlm_Y, nsp_Y)
            l.backward()
            trainer.step()
            metric.add(mlm_l.detach().numpy(), nsp_l.detach().numpy(), tokens_X.shape[0], 1)
            vis.line([[metric[0] / metric[3], metric[1] / metric[3]]], [step], win='train_org', update='append')
            timer.stop()
            step += 1
            if step == num_steps:
                num_step_reached = True
                break

    print('MLM loss {0}'.format(metric[0] / metric[3]))
    print('NSP loss {0}'.format(metric[1] / metric[3]))
    print("{} sentence pairs/sec".format(metric[2] / metric[3]))


def load_data_wiki(batch_size, max_len):
    data_dir = d2l.download_extract('wikitext-2', 'wikitext-2')
    paragraphs = _read_wiki(data_dir)
    train_set = d2l._WikiTextDataset(paragraphs, max_len)
    train_iter = torch.utils.data.DataLoader(train_set, batch_size, shuffle=True)
    return train_iter, train_set.vocab


def _read_wiki(data_dir):
    file_name = os.path.join(data_dir, 'wiki.train.tokens')
    with open(file_name, 'r', encoding='gb18030', errors='ignore') as f:
        lines = f.readlines()
    paragraphs = [line.strip().lower().split('.') for line in lines if len(line.split('.')) >= 2]
    random.shuffle(paragraphs)
    return paragraphs


if __name__ == "__main__":
    batch_size = 512
    max_len = 64
    train_iter, vocab = load_data_wiki(batch_size, max_len)
    net = BERTModel(len(vocab), num_hiddens=128, norm_shape=[128],
                    ffn_num_input=128, ffn_num_hiddens=256, num_heads=4, max_len=max_len,
                    num_layers=6, dropout=0.2, key_size=128, query_size=128, value_size=128,
                    hid_in_features=128, mlm_in_features=128, nsp_in_feature=128)
    device = d2l.try_all_gpus()
    loss = nn.CrossEntropyLoss()

    train_bert(train_iter, net, loss, len(vocab), 200)

    torch.save(net.state_dict(), 'bert.pt')