import torch
import torch.nn as nn
import torch.nn.functional as F         #常用的函数
import torch.optim as optim             #优化器模型
from torch.utils.data import DataLoader,Dataset
import string
import time
import matplotlib.pyplot as plt
from tqdm import tqdm        #进度条

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

#todo 1:获取常见字符的数量
#获取所有的常用字符
all_char = string.ascii_letters+" .,;'"       #获取所有的字符
#获取常用的字符的数量
n_letters = len(all_char)

#todo 2:获取常用国家名 种类数 和 个数
#1.国家名 种类数
categories = ['Arabic', 'Chinese', 'Czech', 'Dutch', 'English', 'French', 'German', 'Greek', 'Irish', 'Italian', 'Japanese', 'Korean', 'Polish', 'Portuguese', 'Russian', 'Scottish', 'Spanish', 'Vietnamese']
#2.国家名 个数
n_categories = len(categories)

#todo 3:定义数据读取函数
def read_data(file_path):
    """
    读取数据
    :param file_path:
    :return:my_list:
    """

    #1.创建两个列表，分别存储：人名，国家名
    my_list_x,my_list_y = [],[]

    #2.打开文件,读取其中内容
    with open(file_path,'r',encoding='utf-8') as f:
        #3.遍历文件中的每一行
        for line in f.readlines():
            #4.过滤数据
            if len(line)<=5:
                 continue
            #5.获取数据
            x,y=line.strip().split('\t')
            #6.将数据添加到列表中
            my_list_x.append(x)
            my_list_y.append(y)

    return my_list_x,my_list_y

#todo 4:创建数据集对象，即：原始数据——数据集生成TensorDataset->数据加载器DataLoader
class NameclassDataset(Dataset):
    def __init__(self,my_list_x,my_list_y):
        self.my_list_x = my_list_x
        self.my_list_y = my_list_y
        self.sample_len = len(my_list_x)

    def __len__(self):
        return self.sample_len

    def __getitem__(self, index):     #支持像列表一样按索引取值（例如 dataset[0]），也是后续 DataLoader 按批次（Batch）抓取数据时调用的接口
        #1.索引边界校检，确保索引在范围内
        index = min(max(index,0),self.sample_len-1)
        #2.按照索引获取原始样本 和 标签
        x = self.my_list_x[index]
        y = self.my_list_y[index]
        #3.人名数据转化为one - hot编码
        #3.1 生成全0张量
        tensor_x = torch.zeros(len(x),n_letters)
        #3.2 遍历整个列表
        for li,letter in enumerate(x):
            #3.2.1 获取字母在全局字母表中的索引数量
            letter_index = all_char.find(letter)
            #3.2.2 将索引位置设置为1
            tensor_x[li][letter_index] = 1

        #4.国家数据转换为 张量
        tensor_y = torch.tensor(categories.index(y),dtype=torch.long)

        return tensor_x,tensor_y

#todo 5:创建数据加载器
def get_dataloader():
    #读取数据文件
    my_list_x,my_list_y = read_data('data/name_classfication.txt')

    #创建数据集对象
    my_dataset = NameclassDataset(my_list_x,my_list_y)

    #创建数据加载器
    #参数：1.数据集对象 2.batch_size 3.是否打乱数据
    my_dataloader = DataLoader(my_dataset,batch_size=1,shuffle=True)

    #测试数据加载器
    for x,y in my_dataloader:
        print(f'x.shape:{x.shape},y.shape:{y.shape}')
        break

    #返回加载数据
    return my_dataloader




if __name__ == '__main__':
    #调用read_data函数，获取数据
    #my_list_x,my_list_y = read_data('data/name_classfication.txt')

    #测试数据集对象
    get_dataloader()

