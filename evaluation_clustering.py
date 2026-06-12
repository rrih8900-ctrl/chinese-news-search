# clustering_from_segfile.py
# 直接利用第一步分词结果 fenci_optimized_result.txt 进行聚类分析
# 每行格式：类别 空格 词1 词2 ...，提取后面的词序列作为文档内容

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
import re

# ==================== 1. 加载停用词 ====================
def load_stopwords(filepath="stopwords.txt"):
    """加载停用词文件，每行一个词"""
    stop_words = set()
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                word = line.strip()
                if word:
                    stop_words.add(word)
        print(f"✅ 从 {filepath} 加载停用词 {len(stop_words)} 个")
    except FileNotFoundError:
        print(f"⚠️ 未找到 {filepath}，将不使用外部停用词")
    # 补充一些常见但可能遗漏的停用词（可根据需要调整）
    extra_stops = {
        "我们", "他们", "你们", "这些", "那些", "这个", "那个", "这里", "那里",
        "什么", "怎么", "为什么", "可以", "进行", "已经", "没有", "还有", "还是",
        "或者", "但是", "所以", "因为", "如果", "然而", "而且", "并且", "一种",
        "一个", "一些", "那么", "这样", "那样", "虽然", "由于", "因此", "于是",
        "并", "就", "都", "也", "还", "又", "只", "每", "各", "某", "第一",
        "第二", "第三", "最后", "目前", "近日", "今日", "昨天", "今天", "明天",
        "将", "会", "能", "可能", "需要", "希望", "要求", "得到", "成为", "作为",
        "具有", "实现", "提供", "使用", "通过", "根据", "按照", "对于", "关于"
    }
    stop_words.update(extra_stops)
    print(f"✅ 合并内置停用词后，总停用词数: {len(stop_words)}")
    return list(stop_words)

# ==================== 2. 加载分词结果文件 ====================
def load_segmented_file(filepath="fenci_optimized_result.txt"):
    """
    读取分词结果，每行格式：类别 空格 词1 词2 ...
    返回：文档内容列表（只保留词序列，去掉类别）、原始行（用于显示）
    """
    contents = []
    raw_lines = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            raw_lines.append(line)
            # 去掉第一个词（类别），后面的用空格连接
            parts = line.split()
            if len(parts) > 1:
                # 将词序列重新用空格连接成字符串
                content = " ".join(parts[1:])
            else:
                content = ""
            contents.append(content)
    return contents, raw_lines

# ==================== 3. 主流程 ====================
def main():
    # 加载停用词
    stop_words = load_stopwords("stopwords.txt")
    
    # 加载分词结果文件
    print("加载 fenci_optimized_result.txt ...")
    documents, raw_lines = load_segmented_file("fenci_optimized_result.txt")
    print(f"文档总数: {len(documents)}")
    
    # TF-IDF 向量化（直接使用分词后的词序列字符串）
    print("计算 TF-IDF 矩阵，设置 max_df=0.7, min_df=3, 停用词过滤...")
    vectorizer = TfidfVectorizer(
        max_df=0.7,          # 过滤超过 70% 文档中出现的词（高频通用词）
        min_df=3,            # 过滤出现在少于 3 篇文档中的词
        max_features=500,    # 保留最多 500 个特征，控制计算量
        stop_words=stop_words  # 使用加载的停用词表
    )
    X = vectorizer.fit_transform(documents).toarray()
    feature_names = vectorizer.get_feature_names_out()
    print(f"TF-IDF 矩阵形状: {X.shape}")
    print(f"有效特征词数量: {len(feature_names)}")
    
    # ========== 肘部法则确定最佳 K 值 ==========
    sse = []
    K_range = range(2, 13)
    print("正在通过肘部法则寻找最佳 K 值...")
    for k in K_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(X)
        sse.append(kmeans.inertia_)
    
    # 绘制肘部法则图
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
    plt.rcParams['axes.unicode_minus'] = False
    plt.figure(figsize=(8, 5))
    plt.plot(K_range, sse, 'bo-')
    plt.xlabel("簇的数量 K")
    plt.ylabel("SSE（簇内误差平方和）")
    plt.title("肘部法则确定最佳聚类数")
    plt.grid(True)
    plt.savefig("elbow_method.png", dpi=150)
    print("肘部法则曲线已保存为 elbow_method.png")
    plt.show()
    
    # 自动选择 K（二阶差分最大对应的 K，仅供参考）
    if len(sse) >= 3:
        second_diffs = [sse[i-2] - 2*sse[i-1] + sse[i] for i in range(2, len(sse))]
        best_k = K_range[2 + second_diffs.index(max(second_diffs))]
    else:
        best_k = 5
    print(f"自动建议最佳 K 值: {best_k}（可根据肘部图手动修改下面的 n_clusters）")
    
    # 手动设置聚类数（可覆盖自动结果）
    n_clusters = best_k   # 改为你观察到的拐点值
    print(f"使用 K={n_clusters} 进行 KMeans 聚类...")
    
    # 最终聚类
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X)
    
    # 轮廓系数
    sil_score = silhouette_score(X, labels)
    print(f"轮廓系数: {sil_score:.4f} (越接近1表示聚类质量越好)")
    
    # ========== t-SNE 降维可视化 ==========
    print("正在进行 t-SNE 降维（可能较慢，请耐心等待）...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, init='random', learning_rate='auto')
    X_tsne = tsne.fit_transform(X)
    
    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=labels, cmap='tab10', alpha=0.7)
    plt.colorbar(scatter, label="簇编号")
    plt.title(f"文档聚类可视化 (t-SNE, K={n_clusters})")
    plt.xlabel("t-SNE 第一维度")
    plt.ylabel("t-SNE 第二维度")
    plt.tight_layout()
    plt.savefig("clustering_result.png", dpi=150)
    print("聚类散点图已保存为 clustering_result.png")
    plt.show()
    
    # ========== 输出文档簇标签 ==========
    doc_ids = [str(i+1) for i in range(len(documents))]
    cluster_df = pd.DataFrame({
        "文档ID": doc_ids,
        "簇编号": labels,
        "原文片段（分词行）": [line[:80] + "..." for line in raw_lines]
    })
    cluster_df.to_csv("doc_clusters.csv", index=False, encoding="utf-8")
    print("文档簇标签已保存到 doc_clusters.csv")
    
    # ========== 每个簇的代表性关键词 ==========
    print("\n各簇代表性关键词（TF-IDF 均值最高的词）:")
    centers = kmeans.cluster_centers_
    for i in range(n_clusters):
        top_indices = centers[i].argsort()[-8:][::-1]   # 取前8个
        top_words = [feature_names[idx] for idx in top_indices]
        print(f"簇 {i}: {', '.join(top_words)}")
    
    # 各簇文档数量
    print("\n各簇文档数量分布:")
    for i in range(n_clusters):
        count = sum(labels == i)
        print(f"簇 {i}: {count} 篇文档")

if __name__ == "__main__":
    main()