import sys
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils import data
from torch import optim
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
from os.path import join
from codecs import open
import os
from copy import deepcopy
import pickle
from collections import Counter
from torch.nn import Parameter

from itertools import zip_longest
from sklearn.model_selection import train_test_split

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class Encoder(nn.Module):
    def __init__(self, embedding_dim,
                 hidden_dim):

        super(Encoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(embedding_dim,
                            self.hidden_dim)  # 自动将LSTM输入的第一维度转化为sequence_length, 输出的shape第一维度为batch_size

    def forward(self, embedded_inputs):
        # [sequence_max_length, batch_size, embed_dim]
        embedded_inputs = embedded_inputs.permute(1, 0, 2)

        # outputs:(sequence_length, batch_size, hidden_dim)
        outputs, hidden = self.lstm(embedded_inputs)  # hidden[0/1]:(n_layers, batch_size, embed_size)
        
        return outputs, hidden
    
class Attention(nn.Module):
    # 实现函数 a(k,q)=V*tanh(Wk*X+Wq*Q)
    def __init__(self, enc_hidden_dim, dec_hidden_dim):
        super(Attention, self).__init__()

        self.input_dim = enc_hidden_dim
        self.hidden_dim = dec_hidden_dim

        self.W_k = nn.Linear(self.input_dim, self.hidden_dim, bias=False)  # 全连接层，内含学习参数Wk
        self.W_q = nn.Linear(self.input_dim, self.hidden_dim, bias=False)  # 全连接层，内含学习参数Wq
        
        self.V = Parameter(torch.FloatTensor(self.hidden_dim), requires_grad=True)  # 学习参数V
        # 初始化 向量 V
        nn.init.uniform(self.V, -1, 1)  # 均匀分布初始化
        
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax()

    def forward(self, enc_hidden, dec_hidden, valid_length=None):  
        # enc_hidden:(batch_size, seq_len, hidden_dim)
        # dec_hidden:(batch_size, hidden_dim)
        ctx = self.W_k(enc_hidden).permute(0, 2, 1)  # (batch_size, hidden_dim, seq_len)
        Wq = self.W_q(dec_hidden)  # (batch_size, hidden_dim)
        Wq = Wq.unsqueeze(2).expand(-1, -1, enc_hidden.size(1))  # (batch_size, hidden_dim, seq_len)
        
        # V:(batch_size, 1, hidden_dim)
        V = self.V.unsqueeze(0).expand(dec_hidden.size(0), -1).unsqueeze(1)
        # score:(batch_size, hidden_dim, seq_len)
        att_score = self.tanh(ctx + Wq)
        score = torch.bmm(V, att_score).squeeze(1)  # (batch_size, seq_len)注意力得分（标量）
        alpha = self.softmax(score)  # shape: (batch_size, seq_len)分布概率
        hidden_state = torch.bmm(ctx, alpha.unsqueeze(2)).squeeze(2)  # (batch_size, hidden_dim)
#         score[mask]
        return hidden_state, score
    
class Decoder(nn.Module):
    def __init__(self, embedding_dim,
                 hidden_dim):
        super(Decoder, self).__init__()
        self.dec_hidden_dim = hidden_dim
        self.lstm = nn.LSTMCell(embedding_dim,
                            self.dec_hidden_dim)
        
        self.att = Attention(enc_hidden_dim = hidden_dim, dec_hidden_dim = hidden_dim)
        self.hidden_out = nn.Linear(hidden_dim * 2, hidden_dim)

    def forward(self, decoder_input,  # 此处decoder输入是单个单词，(batch, seq_len, embed_dim)
                dec_hidden,  # decoder的初始隐层状态,注意是2维而非3维
                enc_output):  # encoder的输出隐层状态，shape为

        # 注意此处的LSTMCell是单个Cell，而不是完整的LSTM，需要手动迭代计算。
        # 另外，Cell的输出仅有两个h_t, c_t，没有LSTM中的output
        def step(x, hidden):  
            # Regular LSTM  计算单个数据的隐含状态
            dec_h, dec_c = self.lstm(x, hidden)  # dec_h.shape:(batch_size, dec_hidden_dim)
            # Attention section
            out = dec_h
            context = enc_output
            dec_hidden_t, output_alpha = self.att(context, out)  # dec_hidden_t:(batch_size, hidden_dim)
            dec_t = F.tanh(self.hidden_out(torch.cat((dec_hidden_t, dec_h), 1)))  # 将注意力层的dec_hidden和decoder每一步的hidden串联后非线性处理
            hidden = (dec_t, dec_c)
            return hidden, output_alpha
            
        decoder_input = decoder_input.permute(1, 0, 2)  # (seq_len, batch, embed_dim)
     
        outputs = []
        pointers = []

        # Recurrence loop
        for dec_input in decoder_input:  # dec_input shape:(batch, embed_dim)
#             if tag_id == 1:  # 起始边界的词才作为decoder输入进行训练
            (h_t, c_t), outs = step(dec_input, dec_hidden)
            dec_hidden = (h_t, c_t)
            outputs.append(outs.unsqueeze(0))
#             else:
#             outputs.append(torch.zeros((outs.size(0), outs.size(1))))
#             pointers.append(score)
        outputs = torch.cat(outputs).permute(1, 0, 2)  # seq_len个结果拼接，再转换->(batch_size, seq_len, embed_dim)

        return outputs

class PointerNet(nn.Module):
    def __init__(self, vocab_size,
                 embedding_dim,
                 hidden_dim):
        super(PointerNet, self).__init__()
        self.embedding_dim = embedding_dim
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.encoder = Encoder(embedding_dim, hidden_dim)
        self.decoder = Decoder(embedding_dim, hidden_dim)

    def forward(self, inputs):  # inputs shape:(batch, seq)
        embed_input = self.embedding(inputs)  # (batch, seq, embed)
        enc_output, enc_hidden = self.encoder(embed_input)  # enc_output shape:(seq, batch, hidden_dim)
        dec_h_t, dec_c_t = enc_hidden[0][-1], enc_hidden[1][-1]  # decoder的初始隐含状态
        dec_hidden0 = (dec_h_t, dec_c_t)
        enc_output = enc_output.permute(1, 0, 2)  # 需要转化shape，以符合attention的输入
        pointer_output = self.decoder(embed_input, dec_hidden0, enc_output)
        
        return pointer_output


def build_corpus(split, make_vocab=True, data_dir="./ResumeNER"):
    def build_map(lists, maps):
#         maps = {}
        for list_ in lists:
            for e in list_:
                if e not in maps:
                    maps[e] = len(maps)
        return maps
    """读取数据"""
    assert split in ['train', 'dev', 'test']

    word_lists = []
    tag_lists = []
    with open(data_dir, 'r', encoding='utf-8') as f:
        word_list = ["<SOS>"]
        tag_list = ["<SOS>"]
        for line in f:
            if line != "\r\n":
                try:
                    word, tag = line.strip('\n').split()
                    word_list.append(word)
                    tag_list.append(tag)
                except:
                    print(line)
#                 print(tag)
            else:
                word_lists.append(word_list)
                tag_lists.append(tag_list)
                word_list = ["<SOS>"]
                tag_list = ["<SOS>"]
        word_lists.append(word_list)
        tag_lists.append(tag_list)
    # 如果make_vocab为True，还需要返回word2id和tag2id
    if make_vocab:
        word2id = {}
        tag2id = {}
        word2id['<unk>'] = len(word2id)
        word2id['<pad>'] = len(word2id)
        tag2id['<unk>'] = len(tag2id)
        tag2id['<pad>'] = len(tag2id)
        word2id = build_map(word_lists, word2id)
        tag2id = build_map(tag_lists, tag2id)
        return word_lists, tag_lists, word2id, tag2id
    else:
        return word_lists, tag_lists

def build_vocab(word_lists, tag_lists):
    def build_map(lists):
        maps = {}
        for list_ in lists:
            for e in list_:
                if e not in maps:
                    maps[e] = len(maps)
        return maps
    word2id = build_map(word_lists)
    tag2id = build_map(tag_lists)
    word2id['<unk>'] = len(word2id)
    word2id['<pad>'] = len(word2id)
    tag2id['<unk>'] = len(tag2id)
    tag2id['<pad>'] = len(tag2id)
    return word2id, tag2id

def sort_by_lengths(word_lists, tag_lists):
    pairs = list(zip(word_lists, tag_lists))
    indices = sorted(range(len(pairs)),
                     key=lambda k: len(pairs[k][0]),
                     reverse=True)
    pairs = [pairs[i] for i in indices]
    # pairs.sort(key=lambda pair: len(pair[0]), reverse=True)

    word_lists, tag_lists = list(zip(*pairs))

    return word_lists, tag_lists, indices

def tensorized(batch, maps):
    PAD = maps.get('<pad>')
    UNK = maps.get('<unk>')

    max_len = len(batch[0])
    batch_size = len(batch)

    batch_tensor = torch.ones(batch_size, max_len).long() * PAD
    for i, l in enumerate(batch):
        for j, e in enumerate(l):
            batch_tensor[i][j] = maps.get(e, UNK)
    # batch各个元素的长度
    lengths = [len(l)-1 for l in batch]

    return batch_tensor, lengths

print("Reading dataset...")
data_path = "../dataset/ZH_BIOE_V2.txt"
word_lists, tag_lists = build_corpus("train", data_dir=data_path, make_vocab=False)

# 60%用于训练集，20%用于测试集，20%用于验证集
print("Spliting dataset...")
train_word_lists, X_test, train_tag_lists, y_test = train_test_split(word_lists,tag_lists,test_size=0.4,random_state=0)
test_word_lists, dev_word_lists, test_tag_lists, dev_tag_lists = train_test_split(X_test,y_test,test_size=0.5,random_state=0)

print("Making vocab...")
word2id, tag2id = build_vocab(train_word_lists, train_tag_lists)

print("Sorting by lengths")
train_word_sort_lists, train_tag_sort_lists, _  = sort_by_lengths(train_word_lists, train_tag_lists)
train_word_idx_lists, train_tag_length = tensorized(train_word_sort_lists, word2id)
train_word_idx_lists = train_word_idx_lists.to(device)  # 数据->GPU化
train_tag_idx_lists, train_tag_length = tensorized(train_tag_sort_lists, tag2id)
# train_tag_idx_lists = train_tag_idx_lists.to(device)

target_point = []
target_point_batch = []
start_idx = tag2id.get("B")
end_idx = tag2id.get("E")

print("Word->PointerWord")
for words, tags in zip(train_word_idx_lists.tolist(), train_tag_idx_lists.tolist()):
    target_point = []
    start_p = []
    for ind, (word, tag) in enumerate(zip(words, tags)):
        if tag != start_idx and tag != end_idx:  # 非起始标签，结果指向null
            target_point.append(0)
        elif tag == start_idx:
            if len(start_p) > 0:  # 单个标签成独立概念
                start_p.clear()
            start_p.append(ind)  # 记录start端下标,只允许记录一个
            target_point.append(ind)
        elif tag == end_idx:
            target_point.append(0)  # end端下标位置补0
            end_word_ind = start_p.pop(0)  # 将start下标取出 
            target_point[end_word_ind] = ind  # end标签内容替代start位置内置
    target_point_batch.append(target_point)
# target_point_batch

print("Building model and Training")
model = PointerNet(vocab_size=len(word2id), embedding_dim=100, hidden_dim=128).to(device)  # 定义模型及参数
loss = torch.nn.CrossEntropyLoss()
optim = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.01)
y_gold = torch.tensor(target_point_batch).long()[:, 1:].to(device)
# print("y_gold:", y_gold.shape)
train_word_input = train_word_idx_lists[:, 1:]
train_set  = data.TensorDataset(train_word_input, y_gold)
batch_word_input = data.DataLoader(train_set, batch_size=10)
for epoch in range(10):
    for batch_word in batch_word_input:
        pred_word = model(batch_word)
    #     print(pred_word.shape)
        l = F.cross_entropy(pred_word.reshape(-1,y_gold.size(-1)), y_gold.reshape(-1)).to(device)
        optim.zero_grad()
        l.backward()
        optim.step()
        print(l)
#     print(F.softmax(pred_word, dim=-1).argmax(-1))
#     print(pred_word.shape)
#     break