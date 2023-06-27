import collections
import re
from d2l import torch as d2l
import torch
import random
import os
from torch import nn
from visdom import Visdom

class Vocab:
    def __init__(self, tokens=None, min_freq=0, reserved_tokens=None):
        # reserved token : <pad> <bos> ...
        if tokens is None:
            tokens = []
        if reserved_tokens is None:
            reserved_tokens = []
        # counter: token_to_freq mapping
        counter = count_corpus(tokens)
        # _token_freq: 2d list [[token, freq], [...], ...]
        self._token_freqs = sorted(counter.items(), key=lambda x: x[1], reverse=True)
        self.idx_to_token = ['<unk>'] + reserved_tokens
        self.token_to_idx = {token: idx for idx, token in enumerate(self.idx_to_token)}
        for token, freq in self._token_freqs:
            if freq < min_freq:
                break
            if tokens not in self.idx_to_token:
                self.idx_to_token.append(token)
                self.token_to_idx[token] = len(self.idx_to_token) - 1

    def __len__(self):
        return len(self.idx_to_token)

    # vocab[tokens] -> idx
    def __getitem__(self, tokens):
        if not isinstance(tokens, (list, tuple)):
            return self.token_to_idx.get(tokens, self.unk)
        return [self.__getitem__(token) for token in tokens]

    def to_tokens(self, indices):
        if not isinstance(indices, (list, tuple)):
            return self.idx_to_token[indices]
        return [self.idx_to_token[index] for index in indices]

    # 方法包装成属性，让方法可以以属性的形式被访问和调用
    @property
    def unk(self):
        return 0

    @property
    def token_freqs(self):
        return self._token_freqs


def count_corpus(tokens):
    # 统计词元出现的频率
    # tokens 可能是一维列表也可能是二维列表
    if len(tokens) == 0 or isinstance(tokens[0], list):
        # 展成一维列表
        tokens = [token for line in tokens for token in line]
    # 字典计数器，每个token映射到count上
    return collections.Counter(tokens)


def tokenize(lines, token='word'):
    # lines->line->tokens
    if token == 'word':
        return [line.split() for line in lines]
    elif token == 'char':
        return [list(line) for line in lines]
    else:
        print('error: unk token', token)


def get_tokens_and_segments(token_a, token_b=None):
    tokens = ['<cls>'] + token_a + ['<sep>']
    segments = [0] * (len(token_a) + 2)
    if token_b is not None:
        tokens += token_b + ['<sep>']
        segments += [1] * (len(token_b) + 1)
    return tokens, segments


# wikitext-2 中，每一行代表一个段落
# 返回lines->line->sentence
def _read_wiki(data_dir):
    file_name = os.path.join(data_dir, 'wiki.train.tokens')
    with open(file_name, 'r', encoding='gb18030', errors='ignore') as f:
        lines = f.readlines()
    paragraphs = [line.strip().lower().split('.') for line in lines if len(line.split('.')) >= 2]
    random.shuffle(paragraphs)
    return paragraphs


# 给上下句，以50%的几率返回正确第二句，50%几率随机替换第二句
def _get_next_sentence(sentence, next_sentence, paragraphs):
    if random.random() < 0.5:
        is_next = True
    else:
        is_next = False
        next_sentence = random.choice(random.choice(paragraphs))
    return sentence, next_sentence, is_next


# 给段落和全文，返回该段落的上下句训练对
def _get_nsp_data_from_paragraph(paragraph, paragraphs, max_len):
    nsp_data = []
    for i in range(len(paragraph) - 1):
        sentence1, sentence2, is_next = _get_next_sentence(paragraph[i], paragraph[i+1], paragraphs)
        if len(sentence1) + len(sentence2) + 3 > max_len:
            continue
        tokens, segments = get_tokens_and_segments(sentence1, sentence2)
        nsp_data.append((tokens, segments, is_next))
    return nsp_data


def _replace_mlm_tokens(tokens, candidate_pred_position, num_mlm_preds, vocab):
    # tokens 是bert输入序列的词元的列表 '<cls> deep learning is a wonderful subject <sep>'
    # candidate_pred_position 是不包括特殊词元的bert输入序列的词元索引列表 [1, 2, 3,4, 5, 6]
    mlm_input_tokens = [token for token in tokens]
    pred_positions_and_labes = []
    random.shuffle(candidate_pred_position)
    for mlm_pred_position in candidate_pred_position:
        if len(pred_positions_and_labes) >= num_mlm_preds:
            break
        masked_token =None
        if random.random() < 0.8:
            masked_token = '<mask>'
        else:
            if random.random() < 0.5:
                masked_token = tokens[mlm_pred_position]
            else:
                masked_token = random.choice(vocab.idx_to_token)
        mlm_input_tokens[mlm_pred_position] = masked_token
        pred_positions_and_labes.append((mlm_pred_position, tokens[mlm_pred_position]))
    return mlm_input_tokens, pred_positions_and_labes


# 给句子，返回遮掩句子，以及遮掩位置和相应的原始token
def _get_mlm_data_from_tokens(tokens, vocab):
    candidate_pred_positions = []
    for i, token in enumerate(tokens):
        if token in ['<cls>', '<sep>']:
            continue
        candidate_pred_positions.append(i)
    # 遮掩任务中预测15%的词
    num_mlm_preds = max(1, round(len(tokens) * 0.15))
    mlm_input_tokens, pred_positions_and_labes = _replace_mlm_tokens(tokens, candidate_pred_positions, num_mlm_preds, vocab)
    pred_positions_and_labes = sorted(pred_positions_and_labes, key=lambda x: x[0])
    pred_positions = [v[0] for v in pred_positions_and_labes]
    mlm_pred_labes = [v[1] for v in pred_positions_and_labes]
    return vocab[mlm_input_tokens], pred_positions, vocab[mlm_pred_labes]


# 所有数据转tensor+pad
def _pad_bert_inputs(examples, max_len, vocab):
    max_num_mlm_preds = round(max_len * 0.15)
    all_tokens_ids, all_segments, valid_lens = [], [], []
    all_pred_positions, all_mlm_weights, all_mlm_labels = [], [], []
    nsp_labels = []
    for (token_ids, pred_positions, mlm_pred_label_ids, segments, is_next) in examples:
        all_tokens_ids.append(_totensor_and_pad(token_ids, vocab['<pad>'], max_len, torch.long))
        all_segments.append(_totensor_and_pad(segments, 0, max_len, torch.long))
        valid_lens.append(torch.tensor(len(token_ids), dtype=torch.float32))

        all_pred_positions.append(_totensor_and_pad(pred_positions, 0, max_num_mlm_preds, torch.long))
        all_mlm_labels.append(_totensor_and_pad(mlm_pred_label_ids, 0, max_num_mlm_preds, torch.long))
        all_mlm_weights.append(torch.tensor([1.0] * len(mlm_pred_label_ids) + [0.0] * (max_num_mlm_preds - len(mlm_pred_label_ids)), dtype=torch.float32))

        nsp_labels.append(torch.tensor(is_next, dtype=torch.long))
    return (all_tokens_ids, all_segments, valid_lens, all_pred_positions, all_mlm_weights, all_mlm_labels, nsp_labels)


def _totensor_and_pad(data, pad, max_len, torch_type):
    return torch.tensor(data + [pad] * (max_len - len(data)), dtype=torch_type)


class WikiTextDataset(torch.utils.data.Dataset):
    def __init__(self, paragraphs, max_len):
        # paragraphs input = [paragraph1[sentence1, sentence2 ...], paragraph2 ...]
        # output = [paragraph1[sentence1[token1, token2 ...] sentence2...] paragraph2...]
        # sentences output = [sentence1[token1, token2 ...], sentence2 ...]
        paragraphs = [tokenize(paragraph, token='word') for paragraph in paragraphs]
        sentences = [sentence for paragraph in paragraphs for sentence in paragraph]
        self.vocab = Vocab(sentences, min_freq=5, reserved_tokens=['<pad>', '<mask>', '<cls>', '<sep>'])

        examples = []
        for paragraph in paragraphs:
            tem = _get_nsp_data_from_paragraph(paragraph, paragraphs, max_len)
            examples.extend(tem)
        examples = [(_get_mlm_data_from_tokens(tokens, self.vocab) + (segments, is_next)) for tokens, segments, is_next in examples]

        # each example consists of (token_ids, pred_positions, mlm_pred_label_ids, segments, is_next)
        (self.all_tokens_ids, self.all_segments, self.valid_lens,
         self.all_pred_positions, self.all_mlm_weights, self.all_mlm_labels,
         self.nsp_labels) = _pad_bert_inputs(examples, max_len, self.vocab)

    def __getitem__(self, idx):
        return (self.all_tokens_ids[idx], self.all_segments[idx], self.valid_lens[idx],
         self.all_pred_positions[idx], self.all_mlm_weights[idx], self.all_mlm_labels[idx],
         self.nsp_labels[idx])

    def __len__(self):
        return len(self.all_tokens_ids)


def load_data_wiki(batch_size, max_len):
    data_dir = d2l.download_extract('wikitext-2', 'wikitext-2')
    paragraphs = _read_wiki(data_dir)
    train_set = WikiTextDataset(paragraphs, max_len)
    train_iter = torch.utils.data.DataLoader(train_set, batch_size, shuffle=True)
    return train_iter, train_set.vocab


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
    vis.line([[0., 0.]], [0.], win='train', opts=dict(showlegend=True, legend=['mlm_l', 'nsp_l']))
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
            vis.line([[metric[0] / metric[3], metric[1] / metric[3]]], [step], win='train', update='append')
            timer.stop()
            step += 1
            if step == num_steps:
                num_step_reached = True
                break

    print('MLM loss {0}'.format(metric[0] / metric[3]))
    print('NSP loss {0}'.format(metric[1] / metric[3]))
    print("{} sentence pairs/sec".format(metric[2] / metric[3]))


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

    torch.save(net.state_dict(), 'bert_pro.pt')