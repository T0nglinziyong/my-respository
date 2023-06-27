import torch
import os
import collections
import random
from torch import nn
from visdom import Visdom
from d2l import torch as d2l

with open("WiKi_data.txt", 'r') as f:
    lines = f.readlines()
# .strip()移除字符串两端空白字符 .split()
paragraphs = [line.strip().lower().split('.') for line in lines if len(line.split('.')) >= 2]
sentences = [sentence for paragraph in paragraphs for sentence in paragraph]


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
    for i in range(len(sentences) - 2):
        candidate = [sentences[i], sentences[i + 1], sentences[i + 2]]
        input = get_mask(candidate, sentences)
        if len(input[0]) < max_len and len(input[1]) < max_len:
            inputs.append(input)
    return inputs


def get_mask(sentences, all_sentences):
    masked_sentence = random.choice(sentences)
    unmasked = []
    for i, sentence in enumerate(sentences):
        if sentence == masked_sentence:
            unmasked.append('<mask>')
            sequence = i
        else:
            unmasked.extend(sentence.split())
    unmasked = ['<cls>'] + unmasked
    for idx, token in enumerate(unmasked):
        if token == '<mask>':
            position = idx

    if random.random() < 0.5:
        masked_sentence = random.choice(all_sentences)
        issame = False
    else:
        issame = True
    masked_sentence = ['<cls>'] + masked_sentence.split()
    return unmasked, masked_sentence, position, sequence, issame


def get_pad(sentence, max_len, vocab):
    sentence = sentence + ['<pad>'] * (max_len - len(sentence))
    return torch.tensor(vocab[sentence], dtype=torch.long)


class MyData(torch.utils.data.Dataset):
    def __init__(self, sentences, max_len, vocab):
        inputs = data_processing(sentences, max_len)
        self.unmasked_lst = []
        self.masked_lst = []
        self.positions = []
        self.sequences = []
        self.valid_lens1 = []
        self.valid_lens2 = []
        self.is_sames = []
        for (unmasked, masked, position, sequence, is_same) in inputs:
            self.valid_lens1.append([0] * len(unmasked) + [1] * (max_len - len(unmasked)))
            self.valid_lens2.append([0] * len(masked) + [1] * (max_len - len(masked)))
            self.unmasked_lst.append(get_pad(unmasked, max_len, vocab))
            self.masked_lst.append(get_pad(masked, max_len, vocab))
            self.positions.append(torch.tensor(position, dtype=torch.long))
            self.sequences.append(torch.tensor(sequence, dtype=torch.long))
            self.is_sames.append(torch.tensor(is_same, dtype=torch.long))

        self.valid_lens1 = torch.tensor(self.valid_lens1, dtype=torch.bool)
        self.valid_lens2 = torch.tensor(self.valid_lens2, dtype=torch.bool)

    def __len__(self):
        return len(self.unmasked_lst)

    def __getitem__(self, idx):
        return (self.unmasked_lst[idx], self.masked_lst[idx],
                self.valid_lens1[idx], self.valid_lens2[idx],
                self.positions[idx], self.sequences[idx], self.is_sames[idx])


class T_JEPAModule(nn.Module):
    def __init__(self, num_hiddens, num_layers, num_heads, ffn_num_hiddens, dropout, vocab_size, max_len, **kwargs):
        super(T_JEPAModule, self).__init__(**kwargs)
        self.token_embedding = nn.Embedding(vocab_size, num_hiddens)
        self.position_embedding = nn.Parameter(torch.rand(1, max_len, num_hiddens))  # 返回正太分布
        encoder_layer_1 = nn.TransformerEncoderLayer(d_model=num_hiddens, nhead=num_heads, batch_first=True, dropout=dropout)
        self.blks1 = nn.TransformerEncoder(encoder_layer_1, num_layers=num_layers)

        encoder_layer_2 = nn.TransformerEncoderLayer(d_model=num_hiddens, nhead=num_heads, batch_first=True, dropout=dropout)
        self.blks2 = nn.TransformerEncoder(encoder_layer_2, num_layers=num_layers)

        self.linear1 = nn.Linear(num_hiddens, ffn_num_hiddens)
        self.linear2 = nn.Linear(ffn_num_hiddens, num_hiddens)

    def forward(self, sentence1, sentence2, valid_len1, valid_len2, position, sequence):
        sentence1 = self.token_embedding(sentence1)
        sentence1 += self.position_embedding[:, :sentence1.shape[1], :]
        sentence1 = self.blks1(sentence1, src_key_padding_mask=valid_len1)

        sentence2 = self.token_embedding(sentence2)
        sentence2 += self.position_embedding[:, :sentence2.shape[1], :]
        sentence2 = self.blks2(sentence2, src_key_padding_mask=valid_len2)

        token1 = sentence1[[i for i in range(batch_size)], position, :]
        token2 = sentence2[:, 0, :]

        token1 = self.linear2(self.linear1(token1))

        return token1, token2


def get_loss(token1, token2, is_same, batch_size):
    token1 = token1.reshape(batch_size, -1)
    token2 = token2.reshape(batch_size, -1)
    loss = 0
    for tok1, tok2, iss in zip(token1, token2, is_same):
        cos = torch.dot(tok1, tok2) / (torch.norm(tok1) * torch.norm(tok2))
        if iss:
            loss += 1 - cos
        else:
            loss += cos
    return loss / batch_size * 100


def train_tjepa(batch_size, num_epochs, data_loader, net, loss_function):
    trainer = torch.optim.Adam(net.parameters(), lr=0.1)
    step = 0
    loss_avg = 0
    vis = Visdom(env='train')
    vis.line([[0.]], [0.], win='train_mydata', opts=dict(showlegend=True, legend=['loss']))
    for epoch in range(num_epochs):
        for (unmasked, masked, valid_len1, valid_len2, position, sequence, is_same) in data_loader:
            trainer.zero_grad()
            token1, token2 = net(unmasked, masked, valid_len1, valid_len2, position, sequence)
            loss = loss_function(token1, token2, is_same, batch_size)
            loss.backward()
            trainer.step()
            loss_avg += loss.detach()
            step += 1
            if step % 10 == 0:
                vis.line([loss_avg / 10], [step // 10], win='train_mydata', update='append')
            loss_avg = 0

            print(epoch, 'loss=', loss)


if __name__ == "__main__":
    batch_size = 32
    max_len = 32
    vocab = Vocab(paragraphs, special_tokens=['<mask>', '<cls>', '<pad>'])
    mydata = MyData(sentences, max_len=max_len, vocab=vocab)
    data_loader = torch.utils.data.DataLoader(mydata, batch_size=batch_size, shuffle=False)

    num_epochs = 1

    model = T_JEPAModule(num_hiddens=256, num_layers=4, num_heads=4,
                         ffn_num_hiddens=512, dropout=0.9, vocab_size=len(vocab), max_len=max_len)
    train_tjepa(batch_size=batch_size, num_epochs=num_epochs, data_loader=data_loader, net=model, loss_function=get_loss)

    torch.save(model.state_dict(), 'T-JEPA.pt')










