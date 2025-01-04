import spacy
from collections import Counter
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np  # 确保导入 numpy
from wordcloud import WordCloud
from mpl_toolkits.mplot3d import Axes3D  # 用于绘制3D柱状图
import seaborn as sns
from sklearn.cluster import KMeans  # 用于聚类

# 设置 SpaCy 模型
MODEL = "de_core_news_sm"

def read_txt_file(txt_path):
    """读取文本文件"""
    with open(txt_path, 'r', encoding='utf-8') as f:
        return f.read()

def extract_word_frequencies(text, nlp, pos_list=None):
    """
    提取词频
    - text: 文本内容
    - nlp: SpaCy 模型
    - pos_list: 限制的词性列表
    """
    doc = nlp(text)
    if pos_list:
        words = [token.lemma_ for token in doc if token.pos_ in pos_list and not token.is_stop and token.is_alpha]
    else:
        words = [token.lemma_ for token in doc if not token.is_stop and token.is_alpha]
    return Counter(words)

def save_word_frequencies_to_excel(word_freq, output_excel_path):
    """
    保存词频到 Excel 文件
    - word_freq: 词频 Counter 对象
    - output_excel_path: 输出路径
    """
    df = pd.DataFrame(word_freq.items(), columns=["Word", "Frequency"])
    df = df.sort_values(by="Frequency", ascending=False)
    df.to_excel(output_excel_path, index=False, sheet_name="Word Frequencies")
    print(f"Word frequencies saved to {output_excel_path}")

def plot_word_frequencies_3d(word_freq, top_n, output_png_path):
    """
    绘制词频的 3D 面状图
    - word_freq: 词频 Counter 对象
    - top_n: 展示的高频词数量
    - output_png_path: 输出路径
    """
    top_words = word_freq.most_common(top_n)
    words, frequencies = zip(*top_words)

    x = np.arange(len(words))
    y = np.array([0, 1])
    x, y = np.meshgrid(x, y)
    z = np.zeros_like(x, dtype=float)
    z[1, :] = frequencies

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    ax.plot_surface(x, y, z, cmap='viridis', edgecolor='grey', alpha=0.8)
    ax.plot(np.arange(len(words)), [1] * len(words), frequencies, color='black', linewidth=2, label="Frequency Line")

    ax.set_xticks(np.arange(len(words)))
    ax.set_xticklabels(words, rotation=45, ha='right', fontsize=10)
    ax.set_xlabel("Words")
    ax.set_ylabel("")
    ax.set_zlabel("Frequencies")
    ax.set_title(f"Top {top_n} Word Frequencies (3D Area Chart)")

    plt.tight_layout()
    plt.savefig(output_png_path, dpi=300)
    plt.show()
    print(f"3D Word frequency area chart saved to {output_png_path}")

def generate_wordcloud(word_freq, output_png_path):
    """
    生成词云
    - word_freq: 词频 Counter 对象
    - output_png_path: 输出路径
    """
    wordcloud = WordCloud(width=800, height=400, background_color="white").generate_from_frequencies(word_freq)
    
    plt.figure(figsize=(10, 6))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.title("Word Cloud")
    plt.tight_layout()
    plt.savefig(output_png_path, dpi=300)
    plt.show()
    print(f"Word cloud saved to {output_png_path}")

def plot_cluster_density(word_freq, top_n, output_png_path):
    """
    绘制聚类密度可视化图
    - word_freq: 词频 Counter 对象
    - top_n: 展示的高频词数量
    - output_png_path: 输出路径
    """
    top_words = word_freq.most_common(top_n)
    words, frequencies = zip(*top_words)
    x = np.random.rand(len(words)) * 100
    y = np.random.rand(len(words)) * 100
    data = {"word": words, "frequency": frequencies, "x": x, "y": y}
    df = pd.DataFrame(data)

    kmeans = KMeans(n_clusters=3, random_state=0).fit(df[["x", "y"]])
    df["cluster"] = kmeans.labels_

    plt.figure(figsize=(12, 8))
    sns.set(style="darkgrid")

    for cluster in df["cluster"].unique():
        subset = df[df["cluster"] == cluster]
        sns.kdeplot(x=subset["x"], y=subset["y"], cmap="coolwarm", fill=True, alpha=0.6, levels=30)

    plt.scatter(df["x"], df["y"], s=df["frequency"] * 10, c=df["cluster"], cmap="viridis", edgecolor="k", alpha=0.8)
    for i in range(len(df)):
        plt.text(df["x"][i], df["y"][i], df["word"][i], fontsize=10, ha="center", va="center", color="white")

    plt.title("Cluster Density Visualization", fontsize=16)
    plt.xlabel("X-axis")
    plt.ylabel("Y-axis")
    plt.tight_layout()
    plt.savefig(output_png_path, dpi=300)
    plt.show()
    print(f"Cluster density visualization saved to {output_png_path}")

def main(txt_path, output_excel_path, output_bar3d_png, output_wordcloud_png, output_cluster_density_png, pos_list=None, top_n=20):
    """主流程"""
    nlp = spacy.load(MODEL, disable=["parser", "ner"])
    text = read_txt_file(txt_path)
    word_freq = extract_word_frequencies(text, nlp, pos_list)
    print(f"Extracted {len(word_freq)} unique words from the text.")
    save_word_frequencies_to_excel(word_freq, output_excel_path)
    plot_word_frequencies_3d(word_freq, top_n, output_bar3d_png)
    generate_wordcloud(word_freq, output_wordcloud_png)
    plot_cluster_density(word_freq, top_n, output_cluster_density_png)

txt_path = r"你的文件地址.txt" #待处理文本文件
output_excel_path = r"你想存放的文件地址.xlsx" #生成词频表格
output_bar3d_png = r"你想存放的文件地址" #3d词频图
output_wordcloud_png = r"你想存放的文件地址" #词云图
output_cluster_density_png = r"你想存放的文件地址.png" #聚类密度可视化图（python版）

pos_list = ['NOUN', 'VERB', 'ADJ']
top_n = 20

if __name__ == "__main__":
    main(txt_path, output_excel_path, output_bar3d_png, output_wordcloud_png, output_cluster_density_png, pos_list, top_n)
