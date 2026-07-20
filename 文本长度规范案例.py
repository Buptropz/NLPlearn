from tensorflow.keras.preprocessing import sequence

#定义遍历,截断和补齐长度参数
cutlen=10

#对输入文本进行补齐
def padding(x_train):
    #参数:输入文本,填充长度,截断方式(pre:前端截取；post：后端截取),填充方式
    return sequence.pad_sequences(x_train,maxlen=cutlen,truncating='post',padding='post')

def padding_custom(x_train):
    #参数:输入文本,填充长度,填充方式
    return sequence.pad_sequences(x_train,maxlen=cutlen,padding='post')

if __name__ == '__main__':
    #定义输入文本
    x_train=[[1,2,3,4,5,6,7,8,9,10],[1,2,3,4,5,6,7,8,9],[1,2,3,4,5,6,7,8,9,10,11]]
    #调用padding函数进行填充
    x_train=padding(x_train)
    #打印填充后的结果
    print(x_train)