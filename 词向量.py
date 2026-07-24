import fasttext

#训练并保存模型
def train_save():
    #1.开始训练，以 无监督的方式运行 训练
    my_model = fasttext.train_unsupervised('data/fil9')
    #2.保存模型
    my_model.save_model('./model/file9_bin')
    print('训练完成')

#加载模型并预测
def get_word_vector():
    #1.加载预训练的模型
    my_model = fasttext.load_model('./model/file9_bin')
    #2.获取单个词的词向量表示
    result=my_model.get_word_vector('the')
    #3.打印结果
    print(f'type:{type(result)}')
    print(f'shape:{result.shape}')
    print(f'result:{result}')

#查看单词的相似度
def get_similarity():
    #1.加载预训练的模型
    my_model = fasttext.load_model('./model/file9_bin')
    #2.查看单词的近义词
    results=my_model.get_nearest_neighbors('dog')
    print(f'results:{results}')

#模型超参数设定
def set_hyper_parameter():
    my_model = fasttext.train_unsupervised(
        input='data/fil9',
        model='cbow',
        dim=50,           #词向量的维度
        epoch=1,        #训练轮数
        lr=0.1,    #学习率
        thread=10   #线程数
    )
    print('训练完成')


if __name__ == '__main__':
    #train_save()
    #get_similarity()
    #get_word_vector()
    set_hyper_parameter()