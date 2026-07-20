#记录：n的值，一般n-gram的n为2-5
ngram_range =2

# 创建n-gram
def create_ngram(input_list, ngram_range):
    #1.通过滑动窗口，获取n-gram的值
    sliced_list = [input_list[i:] for i in range(ngram_range)]
    #2.zip函数将多个列表进行组合
    ngram_tuples= zip(*sliced_list)
    #3.将元组转换为集合
    return set(ngram_tuples)


if __name__ == '__main__':
    input_list = ['I', 'am', 'a', 'student']
    print(create_ngram(input_list, ngram_range))