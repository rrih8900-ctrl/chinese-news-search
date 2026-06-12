# evaluation.py
import json
import math
import re
from collections import defaultdict
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']  # 用于正常显示中文
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
import jieba.posseg as pseg

# ==================== 1. 加载数据 ====================
print("加载倒排索引...")
with open("inverted_index.json", "r", encoding="utf-8") as f:
    inverted_index = json.load(f)

print("加载文档关键词...")
with open("doc_keywords.json", "r", encoding="utf-8") as f:
    doc_keywords = json.load(f)

print("加载原始文本 data.txt...")
with open("data.txt", "r", encoding="utf-8") as f:
    raw_docs = [line.strip() for line in f if line.strip()]   # 每行一篇文档
N = len(raw_docs)   # 总文档数
print(f"文档总数: {N}")

# 预计算文档向量的模长（用于余弦相似度）
doc_norms = {}
for doc_id in doc_keywords.keys():   # 注意 doc_id 是字符串
    norm_sq = 0.0
    for term, posting in inverted_index.items():
        if doc_id in posting:
            item = posting[doc_id]
            score = item.get("score") if isinstance(item, dict) else item
            norm_sq += score * score
    doc_norms[doc_id] = math.sqrt(norm_sq) if norm_sq > 0 else 1.0
print(f"预计算文档模长完成，有效文档数: {len(doc_norms)}")

# ==================== 2. 检索核心函数（与 app.py 保持一致） ====================
def process_query(query: str):
    """分词、停用词过滤、词性筛选"""
    stopwords = {"的", "了", "在", "是", "我", "有", "和", "就", "不", "人", "都", "一",
                 "也", "还", "这", "那", "去", "能", "可以", "将", "被", "让", "会", "到"}
    query = re.sub(r'(\d+)[ \t]+([a-zA-Z]+)', r'\1\2', query.strip())
    words = pseg.cut(query)
    allowed_pos = {'n', 'nr', 'ns', 'nt', 'nz', 'v', 'vn', 'a', 'eng', 'm', 'i', 'l'}
    terms = []
    for word, flag in words:
        word = word.strip()
        if not word or word in stopwords:
            continue
        if len(word) < 2 and not re.match(r'^[a-zA-Z]+$', word):
            continue
        if re.match(r'^\d+$', word):
            continue
        if flag in allowed_pos or re.match(r'^[a-zA-Z0-9]+$', word):
            terms.append(word)
    return terms

def execute_phrase_search(phrase_words, inverted_index):
    if not phrase_words:
        return {}
    common_docs = set(inverted_index.get(phrase_words[0], {}).keys())
    for word in phrase_words[1:]:
        common_docs &= set(inverted_index.get(word, {}).keys())
    results = {}
    for doc_id in common_docs:
        current_positions = inverted_index[phrase_words[0]][doc_id]['pos']
        for word in phrase_words[1:]:
            next_positions = inverted_index[word][doc_id]['pos']
            valid_next = []
            for p in current_positions:
                if (p + 1) in next_positions:
                    valid_next.append(p + 1)
            current_positions = valid_next
            if not current_positions:
                break
        if current_positions:
            score = sum(inverted_index[w][doc_id]['score'] for w in phrase_words)
            results[doc_id] = score
    return results

def advanced_search(query_string, inverted_index, process_query_func, doc_norms, default_op="OR"):
    """返回 [(doc_id, cosine_similarity), ...] 按相似度降序"""
    query_string = query_string.strip()
    if not query_string:
        return []

    # 短语检索
    phrase_match = re.search(r'"(.*?)"', query_string)
    if phrase_match:
        phrase_text = phrase_match.group(1)
        phrase_words = process_query_func(phrase_text)
        if not phrase_words:
            return []
        results_dict = execute_phrase_search(phrase_words, inverted_index)
        return sorted(results_dict.items(), key=lambda x: x[1], reverse=True)

    # 解析布尔操作符
    tokens = query_string.split()
    operators = [t.upper() for t in tokens if t.upper() in ["AND", "OR", "NOT"]]
    terms = [t for t in tokens if t.upper() not in ["AND", "OR", "NOT"]]
    processed_terms = []
    for term in terms:
        processed_terms.extend(process_query_func(term))
    if not processed_terms:
        return []

    if operators:
        op = operators[0]
    else:
        op = default_op

    term_postings = {term: inverted_index.get(term, {}) for term in processed_terms}

    # 确定候选文档集
    if op == "OR":
        candidate_docs = set()
        for posting in term_postings.values():
            candidate_docs.update(posting.keys())
    elif op == "AND":
        if not processed_terms:
            return []
        candidate_docs = set(term_postings[processed_terms[0]].keys())
        for term in processed_terms[1:]:
            candidate_docs &= set(term_postings[term].keys())
    elif op == "NOT":
        if len(processed_terms) < 2:
            return []
        positive = set(term_postings[processed_terms[0]].keys())
        negative = set(term_postings[processed_terms[1]].keys())
        candidate_docs = positive - negative
    else:
        return []

    if not candidate_docs:
        return []

    # 构建查询向量 (TF-IDF)
    q_tf = defaultdict(int)
    for term in processed_terms:
        q_tf[term] += 1
    query_vec = {}
    q_norm_sq = 0.0
    for term, tf in q_tf.items():
        if term in inverted_index:
            df = len(inverted_index[term])
            idf = math.log((N - df + 0.5) / (df + 0.5) + 1.0)
            weight = tf * idf
            query_vec[term] = weight
            q_norm_sq += weight * weight
    q_norm = math.sqrt(q_norm_sq) if q_norm_sq > 0 else 1.0

    # 计算余弦相似度
    doc_scores = {}
    for doc_id in candidate_docs:
        dot = 0.0
        for term, q_weight in query_vec.items():
            if term in inverted_index and doc_id in inverted_index[term]:
                item = inverted_index[term][doc_id]
                doc_weight = item.get("score") if isinstance(item, dict) else item
                dot += q_weight * doc_weight
        norm = doc_norms.get(doc_id, 1.0)
        sim = dot / (q_norm * norm) if q_norm > 0 and norm > 0 else 0.0
        if sim > 0:
            doc_scores[doc_id] = sim
    return sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)

# ==================== 3. 定义测试查询集和 ground truth ====================
# 你可以根据实际需求修改查询和相关性判断
# 方法：基于文档关键词自动判断（包含查询词中任一关键词即相关）
def get_relevant_by_keywords(query_terms, doc_keywords):
    """如果文档的关键词列表与查询词有交集，则相关"""
    rel_docs = set()
    for doc_id, keywords in doc_keywords.items():
        if any(term in keywords for term in query_terms):
            rel_docs.add(doc_id)
    return rel_docs

# 测试查询集 (可修改)
test_queries = [
    "湖人",
    "OL",
    "比赛",
    "游戏",
    "A股",
    "手机",
    "教练",
    "电影",
    "基金",
    "MP3",
    # 布尔查询示例
    "电影 AND 手机",
    "体育 OR 足球",
    "票房 NOT 基金",
    # 短语查询
    "\"篮球比赛\""
]

# 预先计算每个查询的相关文档集合
ground_truth = {}
for q in test_queries:
    # 去除引号提取真实词
    clean_q = re.sub(r'"', '', q)
    q_terms = process_query(clean_q)
    ground_truth[q] = get_relevant_by_keywords(q_terms, doc_keywords)

print("\n查询集及相关文档数:")
for q, rel_set in ground_truth.items():
    print(f"  {q}: {len(rel_set)} 篇相关")

# ==================== 4. 评测指标函数 ====================
def precision_at_k(retrieved_ids, relevant_ids, k):
    if k == 0:
        return 0.0
    retrieved_k = retrieved_ids[:k]
    if not retrieved_k:
        return 0.0
    return len([doc for doc in retrieved_k if doc in relevant_ids]) / k

def recall_at_k(retrieved_ids, relevant_ids, k):
    if len(relevant_ids) == 0:
        return 0.0
    retrieved_k = retrieved_ids[:k]
    return len([doc for doc in retrieved_k if doc in relevant_ids]) / len(relevant_ids)

def f1_at_k(retrieved_ids, relevant_ids, k):
    p = precision_at_k(retrieved_ids, relevant_ids, k)
    r = recall_at_k(retrieved_ids, relevant_ids, k)
    return 2 * p * r / (p + r) if (p + r) > 0 else 0.0

def average_precision(retrieved_ids, relevant_ids):
    if not relevant_ids:
        return 0.0
    ap = 0.0
    hits = 0
    for i, doc in enumerate(retrieved_ids, 1):
        if doc in relevant_ids:
            hits += 1
            ap += hits / i
    return ap / len(relevant_ids)

def ndcg_at_k(retrieved_ids, relevant_ids, k, rel_gain=1):
    dcg = 0.0
    for i, doc in enumerate(retrieved_ids[:k], 1):
        gain = rel_gain if doc in relevant_ids else 0
        dcg += gain / np.log2(i + 1)
    ideal_rel_count = min(k, len(relevant_ids))
    idcg = sum(rel_gain / np.log2(i + 1) for i in range(1, ideal_rel_count + 1))
    return dcg / idcg if idcg > 0 else 0.0

# ==================== 5. 执行评测 ====================
K_values = [5, 10, 20]
results = {k: {"precision": [], "recall": [], "f1": [], "ndcg": []} for k in K_values}
map_scores = []

for query in test_queries:
    # 调用检索函数，获取前 20 个结果
    retrieved = advanced_search(query, inverted_index, process_query, doc_norms, default_op="OR")
    retrieved_ids = [doc_id for doc_id, _ in retrieved[:max(K_values)]]
    relevant_ids = ground_truth[query]
    # 计算 AP
    ap = average_precision(retrieved_ids, relevant_ids)
    map_scores.append(ap)
    # 计算各 K 指标
    for k in K_values:
        p = precision_at_k(retrieved_ids, relevant_ids, k)
        r = recall_at_k(retrieved_ids, relevant_ids, k)
        f = f1_at_k(retrieved_ids, relevant_ids, k)
        ndcg = ndcg_at_k(retrieved_ids, relevant_ids, k)
        results[k]["precision"].append(p)
        results[k]["recall"].append(r)
        results[k]["f1"].append(f)
        results[k]["ndcg"].append(ndcg)

# 计算平均值
avg_metrics = {}
for k in K_values:
    avg_metrics[k] = {
        "MAP": np.mean(map_scores),
        "Precision": np.mean(results[k]["precision"]),
        "Recall": np.mean(results[k]["recall"]),
        "F1": np.mean(results[k]["f1"]),
        "NDCG": np.mean(results[k]["ndcg"])
    }

# ==================== 6. 输出结果 ====================
print("\n" + "="*60)
print("检索系统质量评测报告 (基于余弦相似度)")
print("="*60)
print(f"测试查询数量: {len(test_queries)}")
print(f"文档总数: {N}")
print(f"倒排索引词汇量: {len(inverted_index)}")
print("\n平均指标:")
for k in K_values:
    print(f"\n--- K = {k} ---")
    print(f"  MAP: {avg_metrics[k]['MAP']:.4f}")
    print(f"  Precision@{k}: {avg_metrics[k]['Precision']:.4f}")
    print(f"  Recall@{k}: {avg_metrics[k]['Recall']:.4f}")
    print(f"  F1@{k}: {avg_metrics[k]['F1']:.4f}")
    print(f"  NDCG@{k}: {avg_metrics[k]['NDCG']:.4f}")

# ==================== 7. 可视化 ====================
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']  # 中文显示
plt.rcParams['axes.unicode_minus'] = False

metrics_to_plot = ["Precision", "Recall", "F1", "NDCG"]
x = np.arange(len(K_values))
width = 0.2
fig, ax = plt.subplots(figsize=(10, 6))
for i, metric in enumerate(metrics_to_plot):
    values = [avg_metrics[k][metric] for k in K_values]
    ax.bar(x + i*width, values, width, label=metric)
ax.set_xlabel("K 值")
ax.set_ylabel("得分")
ax.set_title("检索质量指标随 K 变化")
ax.set_xticks(x + width*1.5)
ax.set_xticklabels([f"K={k}" for k in K_values])
ax.legend()
ax.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig("evaluation_metrics.png")
print("\n指标曲线图已保存为 evaluation_metrics.png")

# 输出每个查询的 AP
print("\n各查询平均精度 (AP):")
for q, ap in zip(test_queries, map_scores):
    print(f"  {q}: {ap:.4f}")