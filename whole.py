import pandas as pd
import spacy
from collections import Counter
import re
from concurrent.futures import ThreadPoolExecutor

# 设置 SpaCy 模型
MODEL = "de_core_news_sm"

def read_txt_file(txt_path):
    """读取文本文件"""
    with open(txt_path, 'r', encoding='utf-8') as f:
        return f.read()

def parse_dictionary(dict_path):
    """解析词典文件"""
    data = []
    with open(dict_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        entry = {}
        for line in lines:
            line = line.strip()
            if line.startswith("词条："):
                if entry:
                    data.append(entry)
                entry = {"Lemma": line.replace("词条：", "").strip()}
            elif line.startswith("中文翻译："):
                entry["short_def"] = line.replace("中文翻译：", "").strip()
            elif line.startswith("--------------------------------------------------"):
                continue
        if entry:
            data.append(entry)
    return pd.DataFrame(data)

def extract_word_frequencies(text, nlp, pos_list=None):
    """提取词频"""
    doc = nlp(text)
    words = [token.lemma_ for token in doc if not token.is_stop and token.is_alpha]
    if pos_list:
        words = [token.lemma_ for token in doc if token.pos_ in pos_list]
    return Counter(words)

def match_frequencies_with_dict(word_freq, dictionary):
    """将词频与词典匹配"""
    matches = []
    for word, freq in word_freq.items():
        if word in dictionary:
            matches.append({"Lemma": word, "Frequency": freq, "Translation": dictionary[word]})
    return pd.DataFrame(matches)

def annotate_text(text, matches, max_translations=3):
    """为文本添加注释"""
    annotations = {}
    translation_counts = {row['Lemma']: 0 for _, row in matches.iterrows()}

    for _, row in matches.iterrows():
        if translation_counts[row['Lemma']] < max_translations:
            annotations[row['Lemma']] = row['Translation']
            translation_counts[row['Lemma']] += 1

    pattern = re.compile(r'\b(' + '|'.join(re.escape(word) for word in annotations.keys()) + r')\b')
    annotated_text = pattern.sub(lambda x: f"{x.group(0)}({annotations[x.group(0)]})", text)
    return annotated_text

def process_text(txt_path, dict_path, output_txt_path, pos_list=None, top_n=50, max_translations=3):
    """主流程"""
    # 加载 SpaCy 模型
    nlp = spacy.load(MODEL, disable=["parser", "ner"])
    
    # 读取文件内容
    text = read_txt_file(txt_path)
    
    # 提取词频
    word_freq = extract_word_frequencies(text, nlp, pos_list)
    print(f"Extracted {len(word_freq)} unique words from the text.")
    
    # 解析词典
    dictionary_df = parse_dictionary(dict_path)
    dictionary = {row['Lemma']: row['short_def'] for _, row in dictionary_df.iterrows()}
    
    # 匹配高频词与词典
    matches = match_frequencies_with_dict(word_freq, dictionary)
    matches = matches.nlargest(top_n, 'Frequency')
    print(f"Matched {len(matches)} words with the dictionary.")
    
    # 为文本添加注释
    annotated_text = annotate_text(text, matches, max_translations)

    # 保存结果
    with open(output_txt_path, 'w', encoding='utf-8') as f:
        f.write(annotated_text)
    print(f"Annotated text saved to {output_txt_path}")

# 文件路径配置
txt_path = r"你的txt文件地址"  # 输入文本路径
dict_path = r"字典在你电脑上的地址"  # 词典路径
output_txt_path = r"你想把生成的文件存放在哪里"  # 输出路径

# 参数配置
pos_list = ['NOUN', 'VERB']  # 提取词性的限制，默认为名词和动词
top_n = 50  # 选取的高频词数量
max_translations = 3  # 每个单词最多注释次数

# 运行主程序
process_text(txt_path, dict_path, output_txt_path, pos_list, top_n, max_translations)
