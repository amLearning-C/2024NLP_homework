import os
import jieba
import math
import re
import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置字体为中文字体，这里以宋体为例


class TextEntropyCalculator:
    def __init__(self, name):
        self.name = name

    def get_unigram_tf(self, text):
        # 计算一元模型的词频
        unigram_tf = {}
        for char in text:
            if char != '\n':
                unigram_tf[char] = unigram_tf.get(char, 0) + 1
        return unigram_tf

    def get_bigram_tf(self, word):
        # 得到二元词的词频表
        bigram_tf = {}
        for i in range(len(word) - 1):
            bigram_tf[(word[i], word[i + 1])] = bigram_tf.get(
                (word[i], word[i + 1]), 0) + 1
        return bigram_tf

    def get_trigram_tf(self, word):
        # 得到三元词的词频表
        trigram_tf = {}
        for i in range(len(word) - 2):
            trigram_tf[(word[i], word[i + 1], word[i + 2])] = trigram_tf.get(
                (word[i], word[i + 1], word[i + 2]), 0) + 1
        return trigram_tf

    def calc_entropy_unigram(self, text, is_ci=0):
        # 计算一元模型的信息熵
        word_tf = self.get_unigram_tf(text)
        total_word_count = sum(word_tf.values())
        entropy = -sum([(count / total_word_count) * math.log2(count / total_word_count) for count in word_tf.values()])
        if is_ci:
            print("<{}>基于词的一元模型的中文信息熵为：{}比特/词".format(self.name, entropy))
        else:
            print("<{}>基于字的一元模型的中文信息熵为：{}比特/字".format(self.name, entropy))
        return entropy

    def calc_entropy_bigram(self, text, is_ci=0):
        # 计算二元模型的信息熵
        word_tf = self.get_bigram_tf(text)
        last_word_tf = self.get_unigram_tf(text)
        bigram_len = sum(word_tf.values())
        entropy = []
        for bigram in word_tf.items():
            p_xy = bigram[1] / bigram_len  # 联合概率p(xy)
            p_x_y = bigram[1] / last_word_tf[bigram[0][0]]  # 条件概率p(x|y)
            entropy.append(-p_xy * math.log(p_x_y, 2))
        entropy = sum(entropy)
        if is_ci:
            print("<{}>基于词的二元模型的中文信息熵为：{}比特/词".format(self.name, entropy))
        else:
            print("<{}>基于字的二元模型的中文信息熵为：{}比特/字".format(self.name, entropy))
        return entropy

    def calc_entropy_trigram(self, text, is_ci=0):
        # 计算三元模型的信息熵
        word_tf = self.get_trigram_tf(text)
        last_word_tf = self.get_bigram_tf(text)
        trigram_len = sum(word_tf.values())
        entropy = []
        for trigram in word_tf.items():
            p_xy = trigram[1] / trigram_len  # 联合概率p(xy)
            p_x_y = trigram[1] / last_word_tf[(trigram[0][0], trigram[0][1])]  # 条件概率p(x|y)
            entropy.append(-p_xy * math.log(p_x_y, 2))
        entropy = sum(entropy)
        if is_ci:
            print("<{}>基于词的三元模型的中文信息熵为：{}比特/词".format(self.name, entropy))
        else:
            print("<{}>基于字的三元模型的中文信息熵为：{}比特/字".format(self.name, entropy))
        return entropy


def preprocess_text(text):
    # 删除所有的隐藏符号
    cleaned_text = ''.join(char for char in text if char.isprintable())
    # 删除所有的非中文字符
    cleaned_text = re.sub(r'[^\u4e00-\u9fa5]', '', cleaned_text)
    # 删除所有停用词
    with open('cn_stopwords.txt', "r", encoding='utf-8') as f:
        stopwords = set([line.strip() for line in f])
        cleaned_text = ''.join(char for char in cleaned_text if char not in stopwords)
    # 删除所有的标点符号
    punctuation_pattern = re.compile(r'[\s!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~]+')
    cleaned_text = punctuation_pattern.sub('', cleaned_text)
    return cleaned_text


# 文件夹路径
folder_path = r"D:/Users/Vincent W/Desktop/研究生/学习/研一下NIP/jyxstxtqj_downcc.com/"

# 初始化数据列表
data = []

# 遍历文件夹中的所有文件
for file_name in os.listdir(folder_path):
    if file_name.endswith(".txt"):  # 只处理以 .txt 结尾的文件
        file_path = os.path.join(folder_path, file_name)
        print("处理文件:", file_path)

        # 读取文本内容并进行分词
        with open(file_path, "r", encoding='ansi') as file:
            text = file.read()
            preprocessed_text = preprocess_text(text)
            words = jieba.lcut(preprocessed_text)

        # 创建文本信息熵计算器实例
        text_entropy_calculator = TextEntropyCalculator(file_name)

        # 计算一元模型的信息熵
        word_entropy_unigram = text_entropy_calculator.calc_entropy_unigram(words, is_ci=1)
        char_entropy_unigram = text_entropy_calculator.calc_entropy_unigram(preprocessed_text)

        # 计算二元模型的信息熵
        word_entropy_bigram = text_entropy_calculator.calc_entropy_bigram(words, is_ci=1)
        char_entropy_bigram = text_entropy_calculator.calc_entropy_bigram(preprocessed_text)

        # 计算三元模型的信息熵
        word_entropy_trigram = text_entropy_calculator.calc_entropy_trigram(words, is_ci=1)
        char_entropy_trigram = text_entropy_calculator.calc_entropy_trigram(preprocessed_text)

        # 添加到数据列表中
        data.append({
            '文件名': file_name,
            '一元词信息熵': word_entropy_unigram,
            '一元字信息熵': char_entropy_unigram,
            '二元词信息熵': word_entropy_bigram,
            '二元字信息熵': char_entropy_bigram,
            '三元词信息熵': word_entropy_trigram,
            '三元字信息熵': char_entropy_trigram
        })

# 将数据列表转换为 DataFrame
df = pd.DataFrame(data)

# 写入 Excel 文件
excel_path = "entropy_data.xlsx"
df.to_excel(excel_path, index=False)
print("信息熵数据已写入 Excel 文件:", excel_path)

# 绘制折线图
plt.figure(figsize=(10, 6))

# 一元模型信息熵
plt.plot(df['文件名'], df['一元词信息熵'], label='一元词信息熵')
plt.plot(df['文件名'], df['一元字信息熵'], label='一元字信息熵')

# 二元模型信息熵
plt.plot(df['文件名'], df['二元词信息熵'], label='二元词信息熵')
plt.plot(df['文件名'], df['二元字信息熵'], label='二元字信息熵')

# 三元模型信息熵
plt.plot(df['文件名'], df['三元词信息熵'], label='三元词信息熵')
plt.plot(df['文件名'], df['三元字信息熵'], label='三元字信息熵')

plt.title('文本模型信息熵')
plt.xlabel('文件名')
plt.ylabel('信息熵')
plt.xticks(rotation=45)

ax = plt.gca()
box = ax.get_position()

# 将坐标轴的位置上移10%
ax.set_position([box.x0, box.y0 + box.height * 0.1,
                 box.width, box.height * 0.9])

plt.legend()
plt.tight_layout()

# 保存图像
plt.savefig('entropy_plot.png')
plt.show()
