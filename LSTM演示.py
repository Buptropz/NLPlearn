import torch
import torch.nn as nn

def lstm_demo():
    #1.创建LSTM模型
    #参数：输入维度，隐藏维度，隐藏层数, 是否双向
    lstm = nn.LSTM(input_size=5, hidden_size=6, num_layers=1, bidirectional=False)

    #2.构建输入张量
    #参数：句子长度，批量大小，输入维度
    input = torch.randn(4, 3, 5)

    #3.构建初始隐藏层参数 和 细胞状态
    #参数：隐藏层数，批量大小，隐藏维度
    h0 = torch.randn(1, 3, 6)
    c0 = torch.randn(1, 3, 6)

    #4.模型计算
    #写法一：手动传入h0,c0
    #参数：输入张量，初始隐藏层参数，初始细胞状态
    #返回：输出张量，最终隐藏层参数，最终细胞状态
    output, (hn, cn) = lstm(input, (h0, c0))

    #写法二：不传入h0,c0——>自动创建一个全0的隐藏层参数 和 细胞状态
    #output, (hn, cn) = lstm(input)

    #5.打印结果
    print(f'output: {output}')
    print(f'final hidden state: {hn}')
    print(f'final cell state: {cn}')

if __name__ == '__main__':
    lstm_demo()
