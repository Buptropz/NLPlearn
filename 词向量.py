import fasttext

def train_save():
    #1.开始训练，以 无监督的方式运行 训练
    my_model = fasttext.train_unsupervised('data/fil9')
    #2.保存模型
    my_model.save_model('./model/file9_bin')
    print('训练完成')


if __name__ == '__main__':
    train_save()