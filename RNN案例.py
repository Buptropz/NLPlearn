import torch
import torch.nn as nn

# RNN模型
def run_for_base():
    #1.创建RNN模型
    #参数：输入维度，隐藏维度，隐藏层数
    rnn = nn.RNN(input_size=5, hidden_size=6, num_layers=1)

    #2.准备输入数据
    #参数：句子长度，批量大小，输入维度
    input = torch.randn(1, 3, 5)

    #3.初始化隐藏层
    #参数：隐藏层数，批量大小，隐藏维度
    h0 = torch.randn(1,3,6)

    #4.运行RNN模型
    output, hn = rnn(input, h0)

    print(f'output: {output}')
    print(f'hidden: {hn}')
    print(f'rnn模型: {rnn}')

#修改模型（句子）的长度
def run_for_sequence_length():
    #1.创建RNN模型
    #参数：输入维度，隐藏维度，隐藏层数
    rnn = nn.RNN(input_size=5, hidden_size=6, num_layers=1)

    #2.准备输入数据
    #参数：句子长度，批量大小，输入维度
    input = torch.randn(20, 3, 5)

    #3.初始化隐藏层
    #参数：隐藏层数，批量大小，隐藏维度
    h0 = torch.randn(1,3,6)

    #4.运行RNN模型
    output, hn = rnn(input, h0)

    print(f'output: {output}')
    print(f'hidden: {hn}')
    print(f'rnn模型: {rnn}')

#修改模型隐藏层的个数
def run_for_hidden():
    #1.创建RNN模型
    #参数：输入维度，隐藏维度，隐藏层数
    rnn = nn.RNN(input_size=5, hidden_size=6, num_layers=2)

    #2.准备输入数据
    #参数：句子长度，批量大小，输入维度
    input = torch.randn(1,3,5)

    #3.初始化隐藏层
    #参数：隐藏层数，批量大小，隐藏维度
    h0 = torch.randn(2,3,6)

    #4.运行RNN模型
    output, hn = rnn(input, h0)

    print(f'output: {output}')
    print(f'hidden: {hn}')
    print(f'rnn模型: {rnn}')

if __name__ == '__main__':
    # run_for_base()
    run_for_sequence_length()