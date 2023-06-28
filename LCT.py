import torch
import os
import math
import collections
import random
from torch import nn
from visdom import Visdom
from d2l import torch as d2l

with open("WiKi_data.txt", 'r') as f:
    lines = f.readlines()
# .strip()移除字符串两端空白字符 .split()
paragraphs = [line.strip().lower().split('.') for line in lines if len(line.split('.')) >= 2]
sentences = [sentence for paragraph in paragraphs for sentence in paragraph if len(sentence.split()) >= 2]



class Vocab:
    def __init__(self, data, min_freq=0, special_tokens=None):
        vocab_counter = count(data)
        if special_tokens is None:
            special_tokens = []
        self.idx_to_token = ['<unk>'] + special_tokens
        tokens = [(vocab, freq) for vocab, freq in vocab_counter.items() if freq > min_freq]
        tokens = sorted(tokens, key=lambda x: x[1], reverse=True)
        for token, _ in tokens:
            if token not in self.idx_to_token:
                self.idx_to_token.append(token)
        self.token_to_idx = {token: idx for idx, token in enumerate(self.idx_to_token)}

    def __len__(self):
        return len(self.idx_to_token)

    def __getitem__(self, items):
        if isinstance(items, (list, tuple)):
            return [self.token_to_idx[item] for item in items]
        return self.token_to_idx[items]

    def idx2token(self, ids):
        if isinstance(ids, (list, tuple)):
            return [self.idx_to_token[idx] for idx in ids]
        elif type(ids) == torch.Tensor:
            return [self.idx_to_token[idx] for idx in ids]
        return self.idx_to_token[ids]


def count(data):
    sentences = [sentence for line in data for sentence in line]
    words = []
    for sentence in sentences:
        tokens = sentence.split()
        words.extend(tokens)
    vocab_counter = collections.Counter(words)
    return vocab_counter


def data_processing(sentences, max_len):
    inputs = []
    for i in range(len(sentences) - 1):
        input = nsp(sentences[i], sentences[i + 1], sentences)
        if len(input[0]) < max_len and len(input[1]) < max_len:
            inputs.append(input)
    return inputs


def nsp(sentence1, sentence2, all_sentences):
    is_next = 1
    if random.random() < 0.5:
        sentence2 = random.choice(all_sentences)
        is_next = 0
    return sentence1.split(), sentence2.split(), is_next


def get_pad(sentence, max_len, vocab):
    t = (max_len - len(sentence))
    if t % 2 == 0:
        sentence = ['<pad>'] * (t // 2) + sentence + ['<pad>'] * (t // 2)
    else:
        sentence = ['<pad>'] * (t // 2 + 1) + sentence + ['<pad>'] * (t // 2)
    return torch.tensor(vocab[sentence], dtype=torch.long)


class MyData(torch.utils.data.Dataset):
    def __init__(self, sentences, max_len, vocab):
        inputs = data_processing(sentences, max_len)
        self.sentences1 = []
        self.sentences2 = []
        self.valid_lens1 = []
        self.valid_lens2 = []
        self.is_nexts = []
        for (sentence1, sentence2, is_next) in inputs:
            self.valid_lens1.append(len(sentence1))
            self.valid_lens2.append(len(sentence2))
            self.sentences1.append(get_pad(sentence1, max_len, vocab))
            self.sentences2.append(get_pad(sentence2, max_len, vocab))
            self.is_nexts.append(torch.tensor(is_next, dtype=torch.long))

    def __len__(self):
        return len(self.sentences1)

    def __getitem__(self, idx):
        return (self.sentences1[idx], self.sentences2[idx],
                self.valid_lens1[idx], self.valid_lens2[idx], self.is_nexts[idx])


class LanguageConvolutionLikeTransformer(nn.Module):
    def __init__(self, num_hiddens, num_heads, ffn_hidden,num_layers, max_len, vocab_size, dropout):
        super(LanguageConvolutionLikeTransformer, self).__init__()
        self.token_embedding = nn.Embedding(vocab_size, num_hiddens)
        self.position_embedding = nn.Parameter(torch.rand(1, max_len, num_hiddens))  # 返回正太分布
        self.encoder_blks = nn.Sequential()
        for i in range(num_layers):
            self.encoder_blks.add_module(f'1_{i}', nn.TransformerEncoderLayer(d_model=num_hiddens, nhead=num_heads,
                                        dim_feedforward=ffn_hidden, dropout=dropout, batch_first=True))
        self.decoder_blks = nn.Sequential()
        for i in range((max_len - 1) // 2):
            self.decoder_blks.add_module(f'2_{i}', nn.TransformerDecoderLayer(d_model=num_hiddens, nhead=num_heads,
                                        dim_feedforward=ffn_hidden, dropout=dropout, batch_first=True))
        self.linear1 = nn.Linear(num_hiddens * 2, num_hiddens * 4)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(num_hiddens * 4, 2)

    def forward(self, sentences, valid_lens):
        t = None
        for sentence, valid_len in zip(sentences, valid_lens):
            max_len = sentence.shape[1]
            sentence = self.token_embedding(sentence)
            sentence += self.position_embedding.data

            for blk in self.encoder_blks:
                sentence = blk(sentence, src_key_padding_mask=get_mask(valid_len, max_len))

            for blk in self.decoder_blks:
                tgt = [s[1:max_len - 1, :] if v == 1
                       else (s[1:max_len - 1, :] + s[:max_len - 2, :] + s[2:max_len, :]) / 3
                       for v, s in zip(valid_len, sentence)]
                tgt = torch.stack(tgt)
                memory_mask = get_mask(valid_len, max_len)
                valid_len = new_valid_len(valid_len)
                max_len -= 2
                tgt_mask = get_mask(valid_len, max_len)
                sentence = blk(tgt=tgt, memory=sentence, memory_key_padding_mask=memory_mask,
                               tgt_key_padding_mask=tgt_mask)
            if t is None:
                t = sentence.reshape(sentence.shape[0], -1)
            else:
                t = torch.cat([t, sentence.reshape(sentence.shape[0], -1)], dim=1)

        t = self.relu(self.linear1(t))
        return self.linear2(t)


def get_mask(valid_lens, max_len):
    mask = [[1] * math.ceil((max_len - valid_len) / 2) + [0] * valid_len +
            [1] * math.floor((max_len - valid_len) / 2) for valid_len in valid_lens]
    return torch.tensor(mask, dtype=torch.bool)


def new_valid_len(valid_len):
    valid_len -= 2
    return torch.clip(valid_len, 1)



if __name__ == "__main__":
    batch_size = 32
    max_len = 19
    vocab = Vocab(paragraphs, special_tokens=['<pad>'])
    mydata = MyData(sentences, max_len=max_len, vocab=vocab)
    data_loader = torch.utils.data.DataLoader(mydata, batch_size=batch_size, shuffle=False)

    net = LanguageConvolutionLikeTransformer(num_hiddens=256, num_heads=4, ffn_hidden=512, num_layers=6,
                                             max_len=max_len, vocab_size=len(vocab), dropout=0.6)
    num_epochs = 4
    trainer = torch.optim.Adam(net.parameters(), lr=0.1)
    loss = nn.CrossEntropyLoss()
    step = 0
    loss_avg = 0
    vis = Visdom(env='train')
    
    for epoch in range(num_epochs):
        for (sentence1, sentence2, valid_len1, valid_len2, is_next) in data_loader:
            trainer.zero_grad()
            judge = net((sentence1, sentence2), (valid_len1, valid_len2))
            l = loss(judge, is_next)
            l.backward()
            trainer.step()
            loss_avg += l.detach()
            step += 1
            if step % 1 == 0:
                if step == 1:
                    vis.line([[loss_avg / step]], [0.], win='train_mydata', opts=dict(showlegend=True, legend=['loss']))
                else:
                    vis.line([[loss_avg / step]], [step], win='train_mydata', update='append')
                print(step, 'loss=', l.detach())
    
    torch.save(net.state_dict(), "LCT.pt")












