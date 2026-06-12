import math
import json
import os
from collections import defaultdict, Counter
import re

def build_index_and_extract_keywords(input_file):
    documents = {}
    inverted_index = defaultdict(dict)
    
    # 读取文件
    with open(input_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # 按行处理：每行是一篇文档，格式：类别 词1 词2 ...
    doc_id = 1
    for line in lines:
        line = line.strip()
        if not line:
            # 保留空行作为空文档，但通常不应有空行，这里跳过并记录警告
            print(f"⚠️ 警告：第 {doc_id} 行是空行，已跳过")
            continue
        # 分割：第一个词是类别，后续是分词结果
        parts = line.split()
        if len(parts) >= 2:
            words = parts[1:]   # 只取正文词，丢弃类别
        else:
            words = []          # 只有类别没有正文词
        documents[str(doc_id)] = words
        doc_id += 1
    
    total_docs = len(documents)
    print(f"✅ 成功加载 {total_docs} 篇完整文档，开始计算特征...")
    
    if total_docs == 0:
        return None, None

    # ==========================================
    # 3. 计算 IDF (逆文档频率)
    # ==========================================
    document_frequency = defaultdict(int)
    for doc_id, words in documents.items():
        unique_words = set(words)
        for word in unique_words:
            document_frequency[word] += 1

    idf = {}
    for word, df in document_frequency.items():
        idf[word] = math.log(total_docs / (df + 1))

    # ==========================================
    # 4. 计算 TF-IDF 并构建【带有词位信息的倒排索引】
    # ==========================================
    doc_keywords = {} 
    
    for doc_id, words in documents.items():
        total_words_in_doc = len(words)
        if total_words_in_doc == 0:
            doc_keywords[doc_id] = []
            continue
            
        term_positions = defaultdict(list)
        for pos, word in enumerate(words):
            term_positions[word].append(pos)
            
        doc_word_scores = {}
        
        for word, positions in term_positions.items():
            tf = len(positions) / total_words_in_doc
            tfidf = tf * idf[word]
            doc_word_scores[word] = tfidf
            
            inverted_index[word][doc_id] = {
                "score": round(tfidf, 4),
                "pos": positions
            }
            
        sorted_keywords = sorted(doc_word_scores.items(), key=lambda x: x[1], reverse=True)
        doc_keywords[doc_id] = [kw[0] for kw in sorted_keywords[:5]]

    return inverted_index, doc_keywords

def main():
    # 🚨 注意这里！如果依然报错空文件，请把下面这行改为：
    # input_file = "fenci_result.txt"
    input_file = "fenci_optimized_result.txt" 
    
    index_file = "inverted_index.json"
    keywords_file = "doc_keywords.json"
    
    results = build_index_and_extract_keywords(input_file)
    if results is None or results[0] is None:
        return  
        
    inverted_index, doc_keywords = results
    
    with open(index_file, 'w', encoding='utf-8') as f:
        json.dump(inverted_index, f, ensure_ascii=False, indent=2)
        
    with open(keywords_file, 'w', encoding='utf-8') as f:
        json.dump(doc_keywords, f, ensure_ascii=False, indent=2)
        
    print(f"✅ 倒排索引已构建完成！包含 {len(inverted_index)} 个唯一词汇。保存至 {index_file}")
    print(f"✅ 关键词抽取完成！保存至 {keywords_file}")
    
    if len(doc_keywords) > 0:
        print("\n--- 抽样展示 ---")
        sample_doc = next((doc for doc, kws in doc_keywords.items() if kws), None)
        if sample_doc:
            print(f"文档ID [{sample_doc}] 的前5个关键词: {doc_keywords[sample_doc]}")
            sample_word = doc_keywords[sample_doc][0]
            print(f"关键词 '{sample_word}' 的倒排列表 (部分):")
            print(list(inverted_index[sample_word].items())[:3])

if __name__ == "__main__":
    main()