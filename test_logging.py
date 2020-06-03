# coding=utf-8
__author__ = 'liu.chunming'
import logging
from parse_args import *  # python命令行解析

class logger_class(object):
    def __init__(self, log_file):

        # 第一步，创建一个logger
        self.logger = logging.getLogger()
        self.logger.setLevel(logging.INFO)    # Log等级总开关
         
        # 第二步，创建一个handler，用于写入日志文件
        #logfile = './logger.txt'
        fh = logging.FileHandler(log_file, mode='w')
        fh.setLevel(logging.INFO)   # 输出到file的log等级的开关
         
        # 第三步，再创建一个handler，用于输出到控制台
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)   # 输出到console的log等级的开关
     
        # 第四步，定义handler的输出格式
        formatter = logging.Formatter("%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s")
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)
     
        # 第五步，将logger添加到handler里面
        self.logger.addHandler(fh)
        self.logger.addHandler(ch)
        
    def log_info(self, information):
        self.logger.info(information)
        
def main():
    # Case
    log = logger_class()
    log.log_info("hello")
    log.log_info("hello")
    log.log_info("hello")

