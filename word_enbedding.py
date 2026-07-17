import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import torch
from tensorflow.keras.preprocessing.text import Tokenizer   #词汇映射器
from torch.utils.tensorboard import SummaryWriter    #可视化 词向量
import jieba        #分词器
import torch.nn as nn

#定义函数，演示：word embedding
def embedding_show():
    #1.定义遍历，记录待处理的文本
    sentence1='我喜欢你很久了，不知道你喜不喜欢我'
    sentence2='我其实不喜欢你，我只是想和你搞暧昧罢了'

    #2.把上述的两句话，封装成：列表
    sentences=[sentence1,sentence2]

    #3.使用jiaba分词器，对句子进行分词
    word_list=[]
    for sentence in sentences:
        word_list.append(jieba.lcut(sentence))
    print(word_list)

    #4.构建词汇表，进行文本数值化（词向量）
    my_tokenizer=Tokenizer()
    my_tokenizer.fit_on_texts(word_list)
    print(my_tokenizer.word_index)
    my_token_list=my_tokenizer.word_index.values()
    seq2id=my_tokenizer.texts_to_sequences(word_list)  #文本转序列

    #5.创建词嵌入层，把文本转成 词向量
    embed = nn.Embedding(num_embeddings=len(my_token_list), embedding_dim=8)   #参一：词汇表大小，参二：词向量的维度
    print(f'embed:{ embed.weight.data}')
    print(f'embed.shape:{ embed.weight.data.shape}')

    #6.词向量可视化
    #6.1创建TensorBoard写入器,将词向量写入runs目录
    my_summary = SummaryWriter(log_dir='./runs')
    #6.2将词向量和索引写入TensorBoard
    my_summary.add_embedding(embed.weight.data,my_token_list)
    #6.3关闭TensorBoard写入器
    my_summary.close()

    #7.查看每个单词对应的词向量
    for idx in range(len(my_tokenizer.word_index)):
        #获取单词对应的词向量
        temp_vector = embed(torch.tensor(idx))
        print(f'单词：{my_tokenizer.index_word[idx+1]}, 词向量：{temp_vector}')
        #获得索引对应的单词
        # word = my_tokenizer.index_word[idx+1]


if __name__ == '__main__':
    embedding_show()