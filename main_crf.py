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
from sklearn_crfsuite import CRF

from utils import *
from evaluating import Metrics  # 实体单个标签指标计算
from evaluating_Metric import *  # 完整实体指标计算
from parse_args import *  # python命令行解析
from test_logging import logger_class

class CRFModel(object):
    def __init__(self,
                 algorithm='lbfgs',
                 c1=0.1,
                 c2=0.1,
                 max_iterations=100,
                 all_possible_transitions=False
                 ):

        self.model = CRF(algorithm=algorithm,
                         c1=c1,
                         c2=c2,
                         max_iterations=max_iterations,
                         all_possible_transitions=all_possible_transitions)

    def train(self, sentences, tag_lists):
        features = [sent2features(s) for s in sentences]
        self.model.fit(features, tag_lists)

    def test(self, sentences):
        features = [sent2features(s) for s in sentences]
        pred_tag_lists = self.model.predict(features)
        return pred_tag_lists


def crf_train_eval(train_data, test_data, remove_O=False):

    # 训练CRF模型
    train_word_lists, train_tag_lists = train_data
    test_word_lists, test_tag_lists = test_data

    crf_model = CRFModel()
    # crf_model.train(train_word_lists, train_tag_lists)
    # with open(arguments['model_file'], "wb") as f:
    #     pickle.dump(crf_model, f)

    model_file = arguments['model_file']
    print("Loading model: ", model_file)
    with open(model_file, "rb") as f:
        crf_model = pickle.load(f)
    print("Testing model")

    pred_tag_lists = crf_model.test(test_word_lists)

    return pred_tag_lists

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
        
if __name__ == "__main__":
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

    
    print("正在训练评估CRF模型...")
    crf_pred = crf_train_eval(
        (train_word_lists, train_tag_lists),
        (test_word_lists, test_tag_lists)
    )
    
    # print(crf_pred)
    
    logger.log_info("开始生成指标...")
    result_filepath = arguments['output_file']
    results = flatten_words(test_word_lists, test_tag_lists, crf_pred)
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

# python main_crf.py --train=True --input_file=./dataset/caption.txt --output_file=./output_result/crf_re.txt --model_file=./model_result/crf.pkl --log_file=./log/log_crf


