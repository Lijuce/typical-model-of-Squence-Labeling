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
from itertools import zip_longest
from sklearn.model_selection import train_test_split

from utils import *
from evaluating import Metrics  # 实体单个标签指标计算
from evaluating_Metric import *  # 完整实体指标计算
from parse_args import *  # python命令行解析
from test_logging import logger_class


def build_corpus(split, make_vocab=True, data_dir="./ResumeNER"):
    def build_map(lists):
        maps = {}
        for list_ in lists:
            for e in list_:
                if e not in maps:
                    maps[e] = len(maps)
        return maps
    """读取数据"""
    assert split in ['train', 'dev', 'test']
    word_lists = []
    tag_lists = []
    #     with open(join(data_dir, split+".char.bmes"), 'r', encoding='utf-8') as f:
    with open(data_dir, 'r', encoding='utf-8') as f:
        word_list = []
        tag_list = []
        for line in f:
            if line != "\r\n":
                try:
                    word, tag = line.strip('\n').split()
                except Exception:
                    pass
                else:
                    word_list.append(word)
                    tag_list.append(tag)
            else:
                word_lists.append(word_list)
                tag_lists.append(tag_list)
                word_list = []
                tag_list = []
    # 如果make_vocab为True，还需要返回word2id和tag2id
    if make_vocab:
        word2id = build_map(word_lists)
        tag2id = build_map(tag_lists)
        word2id['<unk>'] = len(word2id)
        word2id['<pad>'] = len(word2id)
        tag2id['<unk>'] = len(tag2id)
        tag2id['<pad>'] = len(tag2id)
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
    # pairs_ = []
    # for i in indices:
    #     if len(pairs[i][0]) > 5:  # 剔除长度小于5的句子
    #         pairs_.append(pairs[i])
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
    lengths = [len(l) for l in batch]

    return batch_tensor, lengths

def prepocess_data_for_lstmcrf(word_lists, tag_lists, test=False):
    assert len(word_lists) == len(tag_lists)
    for i in range(len(word_lists)):
        word_lists[i].append("<end>")
        if not test:  # 如果是测试数据，就不需要加end token了
            tag_lists[i].append("<end>")

    return word_lists, tag_lists

class BiLSTM(nn.Module):
    def __init__(self, vocab_size, emb_size, hidden_size, out_size):
        """初始化参数：
            vocab_size:字典的大小
            emb_size:词向量的维数
            hidden_size：隐向量的维数
            out_size:标注的种类
        """
        super(BiLSTM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, emb_size)
        self.bilstm = nn.LSTM(emb_size, hidden_size,
                              batch_first=True,
                              bidirectional=True)

        self.lin = nn.Linear(2*hidden_size, out_size)

    def forward(self, sents_tensor, lengths):
        emb = self.embedding(sents_tensor)  # [B, L, emb_size]

        packed = pack_padded_sequence(emb, lengths, batch_first=True)
        rnn_out, _ = self.bilstm(packed)
        # rnn_out:[B, L, hidden_size*2]
        rnn_out, _ = pad_packed_sequence(rnn_out, batch_first=True)

        scores = self.lin(rnn_out)  # [B, L, out_size]

        return scores

    def test(self, sents_tensor, lengths, _):
        """第三个参数不会用到，加它是为了与BiLSTM_CRF保持同样的接口"""
        logits = self.forward(sents_tensor, lengths)  # [B, L, out_size]
        _, batch_tagids = torch.max(logits, dim=2)

        return batch_tagids
    
class BiLSTM_CRF(nn.Module):
    def __init__(self, vocab_size, emb_size, hidden_size, out_size):
        """初始化参数：
            vocab_size:字典的大小
            emb_size:词向量的维数
            hidden_size：隐向量的维数
            out_size:标注的种类
        """
        super(BiLSTM_CRF, self).__init__()
        self.bilstm = BiLSTM(vocab_size, emb_size, hidden_size, out_size)

        # CRF实际上就是多学习一个转移矩阵 [out_size, out_size] 初始化为均匀分布
        self.transition = nn.Parameter(
            torch.ones(out_size, out_size) * 1/out_size)
        # self.transition.data.zero_()

    def forward(self, sents_tensor, lengths):
        # [B, L, out_size]
        emission = self.bilstm(sents_tensor, lengths)

        # 计算CRF scores, 这个scores大小为[B, L, out_size, out_size]
        # 也就是每个字对应对应一个 [out_size, out_size]的矩阵
        # 这个矩阵第i行第j列的元素的含义是：上一时刻tag为i，这一时刻tag为j的分数
        batch_size, max_len, out_size = emission.size()
        crf_scores = emission.unsqueeze(
            2).expand(-1, -1, out_size, -1) + self.transition.unsqueeze(0)

        return crf_scores

    def test(self, test_sents_tensor, lengths, tag2id):
        """使用维特比算法进行解码"""
        start_id = tag2id['<start>']
        end_id = tag2id['<end>']
        pad = tag2id['<pad>']
        tagset_size = len(tag2id)

        crf_scores = self.forward(test_sents_tensor, lengths)
        device = crf_scores.device
        # B:batch_size, L:max_len, T:target set size
        B, L, T, _ = crf_scores.size()
        # viterbi[i, j, k]表示第i个句子，第j个字对应第k个标记的最大分数
        viterbi = torch.zeros(B, L, T).to(device)
        # backpointer[i, j, k]表示第i个句子，第j个字对应第k个标记时前一个标记的id，用于回溯
        backpointer = (torch.zeros(B, L, T).long() * end_id).to(device)
        lengths = torch.LongTensor(lengths).to(device)
        # 向前递推
        for step in range(L):
            batch_size_t = (lengths > step).sum().item()
            if step == 0:
                # 第一个字它的前一个标记只能是start_id
                viterbi[:batch_size_t, step,
                        :] = crf_scores[: batch_size_t, step, start_id, :]
                backpointer[: batch_size_t, step, :] = start_id
            else:
                max_scores, prev_tags = torch.max(
                    viterbi[:batch_size_t, step-1, :].unsqueeze(2) +
                    crf_scores[:batch_size_t, step, :, :],     # [B, T, T]
                    dim=1
                )
                viterbi[:batch_size_t, step, :] = max_scores
                backpointer[:batch_size_t, step, :] = prev_tags

        # 在回溯的时候我们只需要用到backpointer矩阵
        backpointer = backpointer.view(B, -1)  # [B, L * T]
        tagids = []  # 存放结果
        tags_t = None
        for step in range(L-1, 0, -1):
            batch_size_t = (lengths > step).sum().item()
            if step == L-1:
                index = torch.ones(batch_size_t).long() * (step * tagset_size)
                index = index.to(device)
                index += end_id
            else:
                prev_batch_size_t = len(tags_t)

                new_in_batch = torch.LongTensor(
                    [end_id] * (batch_size_t - prev_batch_size_t)).to(device)
                offset = torch.cat(
                    [tags_t, new_in_batch],
                    dim=0
                )  # 这个offset实际上就是前一时刻的
                index = torch.ones(batch_size_t).long() * (step * tagset_size)
                index = index.to(device)
                index += offset.long()

            try:
                tags_t = backpointer[:batch_size_t].gather(
                    dim=1, index=index.unsqueeze(1).long())
            except RuntimeError:
                import pdb
                pdb.set_trace()
            tags_t = tags_t.squeeze(1)
            tagids.append(tags_t.tolist())

        # tagids:[L-1]（L-1是因为扣去了end_token),大小的liebiao
        # 其中列表内的元素是该batch在该时刻的标记
        # 下面修正其顺序，并将维度转换为 [B, L]
        tagids = list(zip_longest(*reversed(tagids), fillvalue=pad))
        tagids = torch.Tensor(tagids).long()

        # 返回解码的结果
        return tagids

# def cal_loss(logits, targets, tag2id):
    # """计算损失
    # 参数:
        # logits: [B, L, out_size]
        # targets: [B, L]
        # lengths: [B]
    # """
    # PAD = tag2id.get('<pad>')
    # assert PAD is not None

    # mask = (targets != PAD)  # [B, L]
    # targets = targets[mask]
    # out_size = logits.size(2)
    # logits = logits.masked_select(
        # mask.unsqueeze(2).expand(-1, -1, out_size)
    # ).contiguous().view(-1, out_size)

    # assert logits.size(0) == targets.size(0)
    # loss = F.cross_entropy(logits, targets)

    # return loss
    
# def cal_lstm_crf_loss(crf_scores, targets, tag2id):
    # """计算双向LSTM-CRF模型的损失
    # 该损失函数的计算可以参考:https://arxiv.org/pdf/1603.01360.pdf
    # """
    # pad_id = tag2id.get('<pad>')
    # start_id = tag2id.get('<start>')
    # end_id = tag2id.get('<end>')

    # device = crf_scores.device

    # # targets:[B, L] crf_scores:[B, L, T, T]
    # batch_size, max_len = targets.size()
    # target_size = len(tag2id)

    # # mask = 1 - ((targets == pad_id) + (targets == end_id))  # [B, L]
    # mask = (targets != pad_id)
    # lengths = mask.sum(dim=1)
    # targets = indexed(targets, target_size, start_id)

    # # # 计算Golden scores方法１
    # # import pdb
    # # pdb.set_trace()
    # targets = targets.masked_select(mask)  # [real_L]

    # flatten_scores = crf_scores.masked_select(
        # mask.view(batch_size, max_len, 1, 1).expand_as(crf_scores)
    # ).view(-1, target_size*target_size).contiguous()

    # golden_scores = flatten_scores.gather(
        # dim=1, index=targets.unsqueeze(1)).sum()

    # # 计算golden_scores方法２：利用pack_padded_sequence函数
    # # targets[targets == end_id] = pad_id
    # # scores_at_targets = torch.gather(
    # #     crf_scores.view(batch_size, max_len, -1), 2, targets.unsqueeze(2)).squeeze(2)
    # # scores_at_targets, _ = pack_padded_sequence(
    # #     scores_at_targets, lengths-1, batch_first=True
    # # )
    # # golden_scores = scores_at_targets.sum()

    # # 计算all path scores
    # # scores_upto_t[i, j]表示第i个句子的第t个词被标注为j标记的所有t时刻事前的所有子路径的分数之和
    # scores_upto_t = torch.zeros(batch_size, target_size).to(device)
    # for t in range(max_len):
        # # 当前时刻 有效的batch_size（因为有些序列比较短)
        # batch_size_t = (lengths > t).sum().item()
        # if t == 0:
            # scores_upto_t[:batch_size_t] = crf_scores[:batch_size_t,
                                                      # t, start_id, :]
        # else:
            # # We add scores at current timestep to scores accumulated up to previous
            # # timestep, and log-sum-exp Remember, the cur_tag of the previous
            # # timestep is the prev_tag of this timestep
            # # So, broadcast prev. timestep's cur_tag scores
            # # along cur. timestep's cur_tag dimension
            # scores_upto_t[:batch_size_t] = torch.logsumexp(
                # crf_scores[:batch_size_t, t, :, :] +
                # scores_upto_t[:batch_size_t].unsqueeze(2),
                # dim=1
            # )
    # all_path_scores = scores_upto_t[:, end_id].sum()

    # # 训练大约两个epoch loss变成负数，从数学的角度上来说，loss = -logP
    # loss = (all_path_scores - golden_scores) / batch_size
    # return loss

def indexed(targets, tagset_size, start_id):
    """将targets中的数转化为在[T*T]大小序列中的索引,T是标注的种类"""
    batch_size, max_len = targets.size()
    for col in range(max_len-1, 0, -1):
        targets[:, col] += (targets[:, col-1] * tagset_size)
    targets[:, 0] += (start_id * tagset_size)
    return targets
    
    
class BILSTM_Model(object):
    def __init__(self, vocab_size, out_size, crf=True):
        """功能：对LSTM的模型进行训练与测试
           参数:
            vocab_size:词典大小
            out_size:标注种类
            crf选择是否添加CRF层"""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # 加载模型参数
        self.emb_size = 100
        self.hidden_size = 128

        self.crf = crf

        # 根据是否添加crf初始化不同的模型 选择不一样的损失计算函数
        if not crf:
            self.model = BiLSTM(vocab_size, self.emb_size,
                                self.hidden_size, out_size).to(self.device)
            self.cal_loss_func = cal_loss
        else:
            self.model = BiLSTM_CRF(vocab_size, self.emb_size,
                                    self.hidden_size, out_size).to(self.device)
            self.cal_loss_func = cal_lstm_crf_loss

        # 加载训练参数：
        self.epoches = arguments['epochs']
        self.print_step = 5
        self.lr = arguments['lr']
        self.batch_size = 30  # 批次大小

        # 初始化优化器
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

        # 初始化其他指标
        self.step = 0
        self._best_val_loss = 1e18
        self.best_model = None

    def train(self, word_lists, tag_lists,
              dev_word_lists, dev_tag_lists,
              word2id, tag2id):
        # 对数据集按照长度进行排序
        word_lists, tag_lists, _ = sort_by_lengths(word_lists, tag_lists)
        dev_word_lists, dev_tag_lists, _ = sort_by_lengths(
            dev_word_lists, dev_tag_lists)

        B = self.batch_size
        for e in range(1, self.epoches+1):
            self.step = 0
            losses = 0.
            for ind in range(0, len(word_lists), B):
                batch_sents = word_lists[ind:ind+B]
                batch_tags = tag_lists[ind:ind+B]
                losses += self.train_step(batch_sents,
                                          batch_tags, word2id, tag2id)

                if self.step % self.print_step == 0:
                    total_step = (len(word_lists) // B + 1)
                    logger.log_info("Epoch {}, step/total_step: {}/{} {:.2f}% Loss:{:.4f}".format(
                        e, self.step, total_step,
                        100. * self.step / total_step,
                        losses / self.print_step
                    ))
                    losses = 0.

            # 每轮结束测试在验证集上的性能，保存最好的一个
            val_loss = self.validate(dev_word_lists, dev_tag_lists, word2id, tag2id)
            logger.log_info("Epoch {}, Val Loss:{:.4f}".format(e, val_loss))

    def train_step(self, batch_sents, batch_tags, word2id, tag2id):
        self.model.train()
        self.step += 1
        # 准备数据
        tensorized_sents, lengths = tensorized(batch_sents, word2id)
        tensorized_sents = tensorized_sents.to(self.device)
        targets, _ = tensorized(batch_tags, tag2id)
        targets = targets.to(self.device)

        # forward
        # print(tensorized_sents)
        scores = self.model(tensorized_sents, lengths)

        # 计算损失 更新参数
        self.optimizer.zero_grad()
        loss = self.cal_loss_func(scores, targets, tag2id).to(self.device)
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def validate(self, dev_word_lists, dev_tag_lists, word2id, tag2id):
        self.model.eval()
        with torch.no_grad():
            val_losses = 0.
            val_step = 0
            for ind in range(0, len(dev_word_lists), self.batch_size):
                val_step += 1
                # 准备batch数据
                batch_sents = dev_word_lists[ind:ind+self.batch_size]
                batch_tags = dev_tag_lists[ind:ind+self.batch_size]
                tensorized_sents, lengths = tensorized(
                    batch_sents, word2id)
                tensorized_sents = tensorized_sents.to(self.device)
                targets, lengths = tensorized(batch_tags, tag2id)
                targets = targets.to(self.device)
                # print(lengths)

                # forward
                scores = self.model(tensorized_sents, lengths)

                # 计算损失
                loss = self.cal_loss_func(
                    scores, targets, tag2id).to(self.device)
                val_losses += loss.item()
            val_loss = val_losses / val_step

            if val_loss < self._best_val_loss:
                logger.log_info("保存模型...")
                self.best_model = deepcopy(self.model)
                self._best_val_loss = val_loss
            return val_loss

    def test(self, word_lists, tag_lists, word2id, tag2id):
        """返回最佳模型在测试集上的预测结果"""
        # 准备数据
        word_lists, tag_lists, indices = sort_by_lengths(word_lists, tag_lists)
        tensorized_sents, lengths = tensorized(word_lists, word2id)
        tensorized_sents = tensorized_sents.to(self.device)

        self.best_model.eval()
        with torch.no_grad():
            batch_tagids = self.best_model.test(
                tensorized_sents, lengths, tag2id)

        # 将id转化为标注
        pred_tag_lists = []
        id2tag = dict((id_, tag) for tag, id_ in tag2id.items())
        for i, ids in enumerate(batch_tagids):
            tag_list = []
            if self.crf:
                for j in range(lengths[i] - 1):  # crf解码过程中，end被舍弃
                    tag_list.append(id2tag[ids[j].item()])
            else:
                for j in range(lengths[i]):
                    tag_list.append(id2tag[ids[j].item()])
            pred_tag_lists.append(tag_list)

        # indices存有根据长度排序后的索引映射的信息
        # 比如若indices = [1, 2, 0] 则说明原先索引为1的元素映射到的新的索引是0，
        # 索引为2的元素映射到新的索引是1...
        # 下面根据indices将pred_tag_lists和tag_lists转化为原来的顺序
        ind_maps = sorted(list(enumerate(indices)), key=lambda e: e[1])
        indices, _ = list(zip(*ind_maps))

        pred_tag_lists = [pred_tag_lists[i] for i in indices]
        tag_lists = [tag_lists[i] for i in indices]

        return pred_tag_lists, tag_lists
        
        
def bilstm_train_and_eval(train_data, dev_data, test_data, word2id, tag2id, crf=True, remove_O=False):
    train_word_lists, train_tag_lists = train_data
    dev_word_lists, dev_tag_lists = dev_data
    test_word_lists, test_tag_lists = test_data

    start = time.time()
    vocab_size = len(word2id)
    out_size = len(tag2id)
    
    bilstm_model = BILSTM_Model(vocab_size, out_size, crf=crf)
    bilstm_model.train(train_word_lists, train_tag_lists, dev_word_lists, dev_tag_lists, word2id, tag2id)

    with open(arguments['model_file'], "wb") as f:
        pickle.dump(bilstm_model, f)
    logger.log_info("模型已保存至:" + arguments['model_file'])
    logger.log_info("训练完毕,共用时{}秒.".format(int(time.time()-start)))
    # 导入训练好的模型, 进行评估
    with open(arguments['model_file'], "rb") as f:
        bilstm_model = pickle.load(f)
    logger.log_info("评估{}模型中...".format(arguments['model_file']))
    pred_tag_lists, test_tag_lists = bilstm_model.test(
        test_word_lists, test_tag_lists, word2id, tag2id)
     
    logger.log_info("开始生成指标...")
    result_filepath = arguments['output_file']
    results = flatten_words(test_word_lists, test_tag_lists, pred_tag_lists)
    if os.path.exists(result_filepath):  # 避免与之前的实验结果混淆
        os.remove(result_filepath)
    with open(result_filepath, "a", encoding='utf8') as f:
        for r in results:
            f.writelines(r)
            f.write('\n')
        print("done")
        
    with open(result_filepath, encoding='utf8') as f:
        counts = evaluate(f)
    report(counts)
        
if __name__ == "__main__":
        # 导入训练好的模型, 进行评估
    if len(sys.argv) > 1:
        arguments = parse_arguments(sys.argv[1:])
    else:
        arguments = parse_arguments()
    logger = logger_class(arguments['log_file'])  # 日志记录
    logger.log_info(arguments)
    logger.log_info("Reading dataset...")
    data_path = arguments['input_file']
    word_lists, tag_lists = build_corpus("train", data_dir=data_path, make_vocab=False)

    logger.log_info("Spliting dataset...")
    train_word_lists, X_test, train_tag_lists, y_test = train_test_split(word_lists,tag_lists,test_size=0.4,random_state=0)
    test_word_lists, dev_word_lists, test_tag_lists, dev_tag_lists = train_test_split(X_test,y_test,test_size=0.5,random_state=0)
    logger.log_info("The length of train_word_lists, test_word_lists and dev_word_lists is :{}, {}, {}".format(len(train_word_lists), len(test_word_lists), len(dev_word_lists)))
    logger.log_info("Make vocab...")
    word2id, tag2id = build_vocab(train_word_lists, train_tag_lists)

    # 如果是加了CRF的lstm还要加入<start>和<end> (解码的时候需要用到)
    if arguments['crf']:
        logger.log_info("Make prepocess dataset for CRF_layer...")
        word2id['<start>'] = len(word2id)
        word2id['<end>'] = len(word2id)
        tag2id['<start>'] = len(tag2id)
        tag2id['<end>'] = len(tag2id)
#        crf_word2id, crf_tag2id = word2id, tag2id
        train_word_lists, train_tag_lists = prepocess_data_for_lstmcrf(train_word_lists, train_tag_lists)
        dev_word_lists, dev_tag_lists = prepocess_data_for_lstmcrf(dev_word_lists, dev_tag_lists)
        test_word_lists, test_tag_lists = prepocess_data_for_lstmcrf(test_word_lists, test_tag_lists, test=True)

    if arguments['train']:
        logger.log_info("Training...")
        bilstm_train_and_eval(
            (train_word_lists, train_tag_lists),
            (dev_word_lists, dev_tag_lists),
            (test_word_lists, test_tag_lists),
            word2id, tag2id,
            crf=arguments['crf']
        )
    else:
        # 导入训练好的模型
        with open(arguments['model_file'], "rb") as f:
            bilstm_model = pickle.load(f)  
        print("评估模型中...")
        pred_tag_lists, test_tag_lists = bilstm_model.test(
            test_word_lists, test_tag_lists, word2id, tag2id)

        logger.log_info("开始生成指标...")
        result_filepath = arguments['output_file']
        results = flatten_words(test_word_lists, test_tag_lists, pred_tag_lists)
        # results = []
        # for t_w, t_t, p_t in zip(test_word_lists, test_tag_lists, pred_tag_lists):
        #     print(t_w, t_t, p_t)
        #     result = []
        #     result.append(" ".join([t_w, t_t, p_t]))
        #     results.append(result)
        if os.path.exists(result_filepath):  # 避免与之前的实验结果混淆
            os.remove(result_filepath)

        with open(result_filepath, "a", encoding='utf8') as f:
            for r in results:
                f.writelines(r)
                f.write('\n')
            print("done")
            
        with open(result_filepath, encoding='utf8') as f:
            counts = evaluate(f)
        report(counts)
        
# Case:----------------------------------
# 中文数据训练
# python main.py --train=True --epochs=1 --model_file=./ckpt.pkl --input_file=./dataset/test.txt --output_file=./result.txt --log_file=./log
# python main.py --train=False --epochs=1 --model_file=./model_result/ckpt.pkl --input_file=./dataset/caption.txt --output_file=./result.txt --log_file=./log/log
# 英文数据训练
# python main.py --train=False --epochs=1 --model_file=./model_result/en_bilstm.pkl --input_file=./dataset/en_BIO.txt --output_file=./output_result/en_bilstm.txt --log_file=./log/log
# python main.py --train=False --crf=True --epochs=30 --model_file=./model_result/en_bilstm_crf.pkl --input_file=./dataset/en_BIO.txt --output_file=./output_result/en_bilstm_crf.txt --log_file=./log/log

