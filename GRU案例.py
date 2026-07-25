import torch
import torch.nn as nn

def gru_demo():
    # 1.创建GRU
    # 输入参数：输入维度，输出维度，层数
    gru = nn.GRU(5,6,1)

    # 2.创建输入
    # 输入参数：句子长度，批量大小，输入维度
    input = torch.randn(2,3,5)

    #3.创建初始隐藏状态
    #参数：层数，批量大小，隐藏维度
    hidden = torch.randn(1,3,6)

    #4.运行GRU模型
    output, hn = gru(input, hidden)

    print(f'output: {output}')
    print(f'hidden: {hn}')
    print(f'gru模型: {gru}')



if __name__ == '__main__':
    gru_demo()