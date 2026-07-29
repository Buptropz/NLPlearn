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

#todo 6:创建模型
#1.RNN模型
class RNN(nn.Module):
    def __init__(self,input_size,hidden_size,output_size,n_layers=1):
        #初始化父类成员
        super().__init__()

        #创建成员变量
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers

        #定义模型
        self.rnn = nn.RNN(self.input_size,self.hidden_size,self.n_layers)
        #定义全连接层
        self.fc = nn.Linear(self.hidden_size,self.output_size)
        #定义激活函数，将输出类别转换为概率分布
        #NLLLOSS损失函数+LogSoftmax(dim=-1）
        self.log_softmax = nn.LogSoftmax(dim=-1)

    #定义前向传播
    #参数：输入张量，隐藏状态
    def forward(self,input,hidden):
        #调整张量的形状
        input = input.unsqueeze(1)
        #计算输出张量
        output,hidden = self.rnn(input,hidden)
        #提取最后一个时间步的隐藏状态
        output = output[-1]
        #计算输出张量
        output = self.fc(output)
        #计算概率分布
        return self.log_softmax(output),hidden

    #定义初始化隐藏状态
    def init_hidden(self):
        #创建张量
        #参数：层数，批量大小，隐藏维度
        return torch.zeros(self.n_layers,1,self.hidden_size)

#2.lstm模型
class LSTM(nn.Module):
    def __init__(self,input_size,hidden_size,output_size,n_layers=1):
        #初始化父类成员
        super().__init__()

        #创建成员变量
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers

        #定义模型
        self.rnn = nn.LSTM(self.input_size,self.hidden_size,self.n_layers)
        #定义全连接层
        self.fc = nn.Linear(self.hidden_size,self.output_size)
        #定义激活函数，将输出类别转换为概率分布
        #NLLLOSS损失函数+LogSoftmax(dim=-1）
        self.log_softmax = nn.LogSoftmax(dim=-1)

    #定义前向传播
    #参数：输入张量，隐藏状态
    def forward(self,input,hidden,c):
        #调整张量的形状
        input = input.unsqueeze(1)
        #计算输出张量
        output,(hidden,c)= self.rnn(input,(hidden,c))
        #提取最后一个时间步的隐藏状态
        output = output[-1]
        #计算输出张量
        output = self.fc(output)
        #计算概率分布
        #返回值：概率分布张量，隐藏状态张量
        return self.log_softmax(output),hidden,c

    #定义初始化隐藏状态
    def init_hidden(self):
        #创建张量
        #参数：层数，批量大小，隐藏维度
        hidden = c = torch.zeros(self.n_layers,1,self.hidden_size)
        return hidden,c

#3.GRU模型
class GRU(nn.Module):
    def __init__(self,input_size,hidden_size,output_size,n_layers=1):
        #初始化父类成员
        super().__init__()

        #创建成员变量
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers

        #定义模型
        self.rnn = nn.GRU(self.input_size,self.hidden_size,self.n_layers)
        #定义全连接层
        self.fc = nn.Linear(self.hidden_size,self.output_size)
        #定义激活函数，将输出类别转换为概率分布
        #NLLLOSS损失函数+LogSoftmax(dim=-1）
        self.log_softmax = nn.LogSoftmax(dim=-1)

    #定义前向传播
    #参数：输入张量，隐藏状态
    def forward(self,input,hidden):
        #调整张量的形状
        input = input.unsqueeze(1)
        #计算输出张量
        output,hidden = self.rnn(input,hidden)
        #提取最后一个时间步的隐藏状态
        output = output[-1]
        #计算输出张量
        output = self.fc(output)
        #计算概率分布
        return self.log_softmax(output),hidden

    #定义初始化隐藏状态
    def init_hidden(self):
        #创建张量
        #参数：层数，批量大小，隐藏维度
        return torch.zeros(self.n_layers,1,self.hidden_size)

#todo 7:测试神经网络
#1.RNN模型测试
def test_model_rnn():
    #创建模型
    model = RNN(57,128,18)
    #准备测试数据
    input = torch.randn(6,57)
    #创建隐藏状态
    hidden = model.init_hidden()
    #4.测试模型
    output,hidden = model(input,hidden)
    print(f'output.shape:{output.shape}')
    print(f'hidden.shape:{hidden.shape}')

#2.RNN，lstm和GRU模型测试案例
def test_model_lstm_gru():
    #定义遍历，获取数据
    input_size,n_hidden,output_size = n_letters,128,n_categories
    #加载数据
    my_list_x,my_list_y = read_data('data/name_classfication.txt')
    #创建数据集对象
    my_dataset = NameclassDataset(my_list_x,my_list_y)
    #创建数据加载器
    my_dataloader = DataLoader(my_dataset,batch_size=1,shuffle=True)
    #模型初始化
    my_rnn= RNN(input_size,n_hidden,output_size)
    my_lstm = LSTM(input_size,n_hidden,output_size)
    my_gru = GRU(input_size,n_hidden,output_size)
    #模型可视化
    print(f'rnn:{my_rnn}')
    print(f'lstm:{my_lstm}')
    print(f'gru:{my_gru}')

    #测试RNN模型
    for i,(x,y) in enumerate(my_dataloader):
        print(f'i:{i}')      #编号，第i条数据
        print(f'x:{x}')       #输入数据
        print(f'y:{y}')       #输出数据

        #创建隐藏状态
        hidden = my_rnn.init_hidden()
        #前向传播
        output,hidden = my_rnn(x[0],hidden)
        print(f'RNN输出形状：{output.shape}')

        if i==0:
            break

    #测试LSTM模型
    for i,(x,y) in enumerate(my_dataloader):
        print(f'i:{i}')
        print(f'x:{x}')
        print(f'y:{y}')
        #创建隐藏状态
        hidden,c = my_lstm.init_hidden()
        #前向传播
        output,(hidden,c) = my_lstm(x[0],(hidden,c))
        print(f'LSTM输出形状：{output.shape}')

        if i==0:
            break

    #测试GRU模型
    for i,(x,y) in enumerate(my_dataloader):
        #创建隐藏状态
        hidden = my_gru.init_hidden()
        #前向传播
        output,hidden = my_gru(x[0],hidden)
        print(f'GRU输出形状：{output.shape}')

        if i==0:
            break

#todo 8.模型训练
#定义变量，记录：学习率，训练轮数
my_lr,my_epochs = 0.005,1

#RNN模型训练
def train_model_rnn():
    #数据准备
    my_list_x,my_list_y = read_data('data/name_classfication.txt')
    my_dataset = NameclassDataset(my_list_x,my_list_y)

    #模型与优化器初始化
    #参数：输入维度，隐藏维度，隐藏层数
    input_size,n_hidden,output_size = n_letters,128,n_categories
    my_rnn = RNN(input_size,n_hidden,output_size)
    #定义损失函数和优化器
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(my_rnn.parameters(),lr=my_lr)

    #参数初始化
    start_time= time.time()         #记录开始时间
    total_num = 0                   #已训练的样本数
    total_loss = 0.0                #已训练的损失和
    total_loss_list = []        #已训练的loss列表
    total_acc_num = 0           #已训练的准确数
    total_acc_list = []         #已训练的准确列表

    #4.具体的训练过程，循环轮数
    for epoch in range(my_epochs):
        print(f'epoch:{epoch+1}/{my_epochs}')
        #4.1创建数据加载器
        my_dataloader = DataLoader(my_dataset,batch_size=1,shuffle=True)
        #4.2训练过程
        for i,(x,y) in enumerate(tqdm(my_dataloader)):
            #前向传播、
            output,hidden = my_rnn(x[0],my_rnn.init_hidden())
            #计算损失
            loss = criterion(output,y)
            #梯度清零，反向传播，优化器更新参数
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            #训练统计
            total_num += 1
            total_loss += loss.item()
            #计算准确率
            pred_tag = torch.argmax(output).item()
            total_acc_num +=(1 if pred_tag == y else 0)
            if total_num%100 == 0:
                avg_loss = total_loss/total_num
                total_loss_list.append(avg_loss)
                total_acc_list.append(total_acc_num / total_num)

            #打印训练日志
            if total_num % 2000 == 0:
                #计算损失
                avg_loss = total_loss/total_num
                #计算耗时
                used_time = int(time.time()-start_time)
                print(f'轮次：{epoch+1}/{my_epochs}，样本：{total_num}，损失：{avg_loss:.4f}，准确率：{total_acc_num/total_num:.4f}，耗时：{used_time}秒')

        #保存模型
        torch.save(my_rnn.state_dict(),f'./model/my_rnn_{epoch+1}.bin')

    #5.走到这里，训练结束，返回结果
    total_time = int(time.time()-start_time)
    print(f'训练完成，总耗时：{total_time}秒，总样本：{total_num}，总损失：{total_loss:.4f}，总准确率：{total_acc_num/total_num:.4f}')

    return total_loss_list,total_time,total_acc_list

#LSTM模型训练
def train_model_lstm():
    #数据准备
    my_list_x,my_list_y = read_data('data/name_classfication.txt')
    my_dataset = NameclassDataset(my_list_x,my_list_y)

    #模型与优化器初始化
    #参数：输入维度，隐藏维度，隐藏层数
    input_size,n_hidden,output_size = n_letters,128,n_categories
    my_rnn = LSTM(input_size,n_hidden,output_size)
    #定义损失函数和优化器
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(my_rnn.parameters(),lr=my_lr)

    #参数初始化
    start_time= time.time()         #记录开始时间
    total_num = 0                   #已训练的样本数
    total_loss = 0.0                #已训练的损失和
    total_loss_list = []        #已训练的loss列表
    total_acc_num = 0           #已训练的准确数
    total_acc_list = []         #已训练的准确列表

    #4.具体的训练过程，循环轮数
    for epoch in range(my_epochs):
        print(f'epoch:{epoch+1}/{my_epochs}')
        #4.1创建数据加载器
        my_dataloader = DataLoader(my_dataset,batch_size=1,shuffle=True)
        #4.2训练过程
        for i,(x,y) in enumerate(tqdm(my_dataloader)):
            #前向传播、
            hidden,c= my_rnn.init_hidden()
            output,hidden,c = my_rnn(x[0],hidden,c)
            #计算损失
            loss = criterion(output,y)
            #梯度清零，反向传播，优化器更新参数
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            #训练统计
            total_num += 1
            total_loss += loss.item()
            #计算准确率
            pred_tag = torch.argmax(output).item()
            total_acc_num +=(1 if pred_tag == y else 0)
            if total_num%100 == 0:
                avg_loss = total_loss/total_num
                total_loss_list.append(avg_loss)
                total_acc_list.append(total_acc_num / total_num)

            #打印训练日志
            if total_num % 2000 == 0:
                #计算损失
                avg_loss = total_loss/total_num
                #计算耗时
                used_time = int(time.time()-start_time)
                print(f'轮次：{epoch+1}/{my_epochs}，样本：{total_num}，损失：{avg_loss:.4f}，准确率：{total_acc_num/total_num:.4f}，耗时：{used_time}秒')

        #保存模型
        torch.save(my_rnn.state_dict(),f'./model/my_lstm_{epoch+1}.bin')

    #5.走到这里，训练结束，返回结果
    total_time = int(time.time()-start_time)
    print(f'训练完成，总耗时：{total_time}秒，总样本：{total_num}，总损失：{total_loss:.4f}，总准确率：{total_acc_num/total_num:.4f}')

    return total_loss_list,total_time,total_acc_list

#GPU模型训练
def train_model_gpu():
    #数据准备
    my_list_x,my_list_y = read_data('data/name_classfication.txt')
    my_dataset = NameclassDataset(my_list_x,my_list_y)

    #模型与优化器初始化
    #参数：输入维度，隐藏维度，隐藏层数
    input_size,n_hidden,output_size = n_letters,128,n_categories
    my_rnn = GRU(input_size,n_hidden,output_size)
    #定义损失函数和优化器
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(my_rnn.parameters(),lr=my_lr)

    #参数初始化
    start_time= time.time()         #记录开始时间
    total_num = 0                   #已训练的样本数
    total_loss = 0.0                #已训练的损失和
    total_loss_list = []        #已训练的loss列表
    total_acc_num = 0           #已训练的准确数
    total_acc_list = []         #已训练的准确列表

    #4.具体的训练过程，循环轮数
    for epoch in range(my_epochs):
        print(f'epoch:{epoch+1}/{my_epochs}')
        #4.1创建数据加载器
        my_dataloader = DataLoader(my_dataset,batch_size=1,shuffle=True)
        #4.2训练过程
        for i,(x,y) in enumerate(tqdm(my_dataloader)):
            #前向传播、
            output,hidden = my_rnn(x[0],my_rnn.init_hidden())
            #计算损失
            loss = criterion(output,y)
            #梯度清零，反向传播，优化器更新参数
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            #训练统计
            total_num += 1
            total_loss += loss.item()
            #计算准确率
            pred_tag = torch.argmax(output).item()
            total_acc_num +=(1 if pred_tag == y else 0)
            if total_num%100 == 0:
                avg_loss = total_loss/total_num
                total_loss_list.append(avg_loss)
                total_acc_list.append(total_acc_num / total_num)

            #打印训练日志
            if total_num % 2000 == 0:
                #计算损失
                avg_loss = total_loss/total_num
                #计算耗时
                used_time = int(time.time()-start_time)
                print(f'轮次：{epoch+1}/{my_epochs}，样本：{total_num}，损失：{avg_loss:.4f}，准确率：{total_acc_num/total_num:.4f}，耗时：{used_time}秒')

        #保存模型
        torch.save(my_rnn.state_dict(),f'./model/my_gru_{epoch+1}.bin')

    #5.走到这里，训练结束，返回结果
    total_time = int(time.time()-start_time)
    print(f'训练完成，总耗时：{total_time}秒，总样本：{total_num}，总损失：{total_loss:.4f}，总准确率：{total_acc_num/total_num:.4f}')

    return total_loss_list,total_time,total_acc_list


def plot_training_history(loss_list, acc_list, model_name, total_time, save_dir='./runs'):
    """
    绘制训练过程的 Loss 曲线、Accuracy 曲线、双轴合并图及汇总信息面板。
    """
    import os
    os.makedirs(save_dir, exist_ok=True)

    steps = [(i + 1) * 100 for i in range(len(loss_list))]

    plt.style.use('fivethirtyeight')
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'{model_name} 训练面板', fontsize=16, fontweight='bold')

    # 左上：Loss 曲线
    ax1 = axes[0, 0]
    ax1.plot(steps, loss_list, color='#E74C3C', linewidth=1.5)
    ax1.set_title('Loss 曲线')
    ax1.set_xlabel('样本数')
    ax1.set_ylabel('Loss')
    ax1.grid(True, alpha=0.3)

    # 右上：Accuracy 曲线
    ax2 = axes[0, 1]
    ax2.plot(steps, acc_list, color='#2ECC71', linewidth=1.5)
    ax2.set_title('Accuracy 曲线')
    ax2.set_xlabel('样本数')
    ax2.set_ylabel('Accuracy')
    ax2.grid(True, alpha=0.3)

    # 左下：Loss + Accuracy 双轴合并图
    ax3 = axes[1, 0]
    ax3.plot(steps, loss_list, color='#E74C3C', linewidth=1.5, label='Loss')
    ax3.set_xlabel('样本数')
    ax3.set_ylabel('Loss', color='#E74C3C')
    ax3.tick_params(axis='y', labelcolor='#E74C3C')

    ax3_twin = ax3.twinx()
    ax3_twin.plot(steps, acc_list, color='#2ECC71', linewidth=1.5, label='Accuracy')
    ax3_twin.set_ylabel('Accuracy', color='#2ECC71')
    ax3_twin.tick_params(axis='y', labelcolor='#2ECC71')

    ax3.set_title('Loss & Accuracy 双轴合并')
    lines1, labels1 = ax3.get_legend_handles_labels()
    lines2, labels2 = ax3_twin.get_legend_handles_labels()
    ax3.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
    ax3.grid(True, alpha=0.3)

    # 右下：汇总信息
    ax4 = axes[1, 1]
    ax4.axis('off')
    final_loss = loss_list[-1] if loss_list else 0
    final_acc = acc_list[-1] if acc_list else 0
    hours = total_time // 3600
    minutes = (total_time % 3600) // 60
    seconds = total_time % 60

    summary_lines = [
        f'模型名称：{model_name}',
        f'总耗时：{hours}h {minutes}m {seconds}s',
        f'总样本步数：{steps[-1] if steps else 0}',
        f'',
        f'最终 Loss：{final_loss:.4f}',
        f'最终 Accuracy：{final_acc:.4f}',
    ]
    for i, line in enumerate(summary_lines):
        ax4.text(0.1, 0.85 - i * 0.12, line, transform=ax4.transAxes,
                 fontsize=13, family='monospace')

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    save_path = os.path.join(save_dir, f'{model_name}_training.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f'训练面板图已保存至：{save_path}')


if __name__ == '__main__':
    loss_list, total_time, acc_list = train_model_rnn()
    plot_training_history(loss_list, acc_list, 'RNN', total_time)

    
