# text_classification.py
# 基于分词结果进行新闻分类，注意防止数据泄露、特征对齐、不平衡评估

import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from collections import Counter

# ==================== 1. 加载数据 ====================
print("加载分词结果文件 fenci_optimized_result.txt ...")
with open("fenci_optimized_result.txt", "r", encoding="utf-8") as f:
    lines = [line.strip() for line in f if line.strip()]

# 提取标签（每行第一个词）和文档内容（去掉类别后的词序列）
labels = []
documents = []
for line in lines:
    parts = line.split()
    if len(parts) < 2:
        continue  # 跳过空行或只有类别的行
    labels.append(parts[0])          # 类别
    documents.append(" ".join(parts[1:]))  # 词序列，空格连接

print(f"总文档数: {len(documents)}")
print(f"类别分布: {Counter(labels)}")

# ==================== 2. 划分训练集和测试集（分层抽样）====================
X_train, X_test, y_train, y_test = train_test_split(
    documents, labels,
    test_size=0.2,          # 20% 测试集
    random_state=42,
    stratify=labels         # 保持类别比例
)

print(f"\n训练集大小: {len(X_train)}")
print(f"测试集大小: {len(X_test)}")
print(f"训练集类别分布: {Counter(y_train)}")
print(f"测试集类别分布: {Counter(y_test)}")

# ==================== 3. TF-IDF 特征提取（注意对齐）====================
# 使用与检索系统一致的参数，但可调整 max_df 和 min_df 以适应分类
vectorizer = TfidfVectorizer(
    max_df=0.7,          # 忽略超过70%文档中出现的词
    min_df=2,            # 忽略低于2篇文档的词
    max_features=5000,   # 适当增加特征数，但控制计算量
    sublinear_tf=True,   # 使用 1+log(tf) 平滑
    ngram_range=(1, 2)   # 可选：加入二元组
)

# 训练集上拟合（学习词表）
X_train_tfidf = vectorizer.fit_transform(X_train)
print(f"\n训练集 TF-IDF 矩阵形状: {X_train_tfidf.shape}")

# 测试集上仅转换（使用训练集的词表）
X_test_tfidf = vectorizer.transform(X_test)
print(f"测试集 TF-IDF 矩阵形状: {X_test_tfidf.shape}")

# 保存 vectorizer 供后续预测新文本使用
joblib.dump(vectorizer, "tfidf_vectorizer.pkl")
print("\nTF-IDF 向量化器已保存为 tfidf_vectorizer.pkl")

# ==================== 4. 训练分类器 ====================
# 4.1 朴素贝叶斯（基线模型）
print("\n" + "="*60)
print("训练多项式朴素贝叶斯分类器...")
nb_model = MultinomialNB(alpha=1.0)
nb_model.fit(X_train_tfidf, y_train)
joblib.dump(nb_model, "nb_classifier.pkl")
print("朴素贝叶斯模型已保存为 nb_classifier.pkl")

# 测试集评估
y_pred_nb = nb_model.predict(X_test_tfidf)
acc_nb = accuracy_score(y_test, y_pred_nb)
print(f"朴素贝叶斯准确率 (Accuracy): {acc_nb:.4f}")
print("\n分类报告（朴素贝叶斯）:")
print(classification_report(y_test, y_pred_nb, digits=4))
print("混淆矩阵:")
print(confusion_matrix(y_test, y_pred_nb))

# 4.2 线性 SVM（进阶模型）
print("\n" + "="*60)
print("训练线性 SVM 分类器...")
svm_model = LinearSVC(C=1.0, random_state=42, max_iter=2000)
svm_model.fit(X_train_tfidf, y_train)
joblib.dump(svm_model, "svm_classifier.pkl")
print("SVM 模型已保存为 svm_classifier.pkl")

y_pred_svm = svm_model.predict(X_test_tfidf)
acc_svm = accuracy_score(y_test, y_pred_svm)
print(f"SVM 准确率 (Accuracy): {acc_svm:.4f}")
print("\n分类报告（线性 SVM）:")
print(classification_report(y_test, y_pred_svm, digits=4))
print("混淆矩阵:")
print(confusion_matrix(y_test, y_pred_svm))

# ==================== 5. 分析不平衡影响 ====================
print("\n" + "="*60)
print("类别不平衡分析:")
print(f"训练集各类别占比:")
total_train = len(y_train)
for cat, count in Counter(y_train).items():
    print(f"  {cat}: {count} ({count/total_train*100:.1f}%)")
print(f"整体准确率（朴素贝叶斯）: {acc_nb:.4f}")
print(f"宏平均 F1（朴素贝叶斯）: {classification_report(y_test, y_pred_nb, output_dict=True)['macro avg']['f1-score']:.4f}")
print(f"宏平均 F1（SVM）: {classification_report(y_test, y_pred_svm, output_dict=True)['macro avg']['f1-score']:.4f}")
print("建议：当类别不平衡时，宏平均 F1 比准确率更能反映模型性能。")

# ==================== 6. 新文本预测示例 ====================
print("\n" + "="*60)
print("预测示例（使用保存的 SVM 模型）:")
# 假设有一篇新新闻
new_texts = [
    "湖人 詹姆斯 浓眉 大胜 勇士 季后赛",
    "基金 发行 募集 规模 超百亿 市场 火爆"
]
# 注意：需要先加载 vectorizer 和模型（这里直接使用已经存在的）
# 对新文本进行同样的预处理（保留词序列，不需要类别）
new_docs = [" ".join(t.split()) for t in new_texts]  # 简单分词，实际情况应与训练时一致
new_tfidf = vectorizer.transform(new_docs)
predictions = svm_model.predict(new_tfidf)
for text, pred in zip(new_texts, predictions):
    print(f"文本: {text[:50]}... 预测类别: {pred}")