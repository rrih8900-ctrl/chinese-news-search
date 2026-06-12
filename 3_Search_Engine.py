import json
import jieba
import jieba.posseg as pseg
import re
import os

# ==========================================
# 1. 预加载基础数据 (索引与停用词)
# ==========================================
def load_search_data():
    index_file = "inverted_index.json"
    keywords_file = "doc_keywords.json"      # 修正：应该加载文档关键词JSON文件
    
    if not os.path.exists(index_file):
        print(f"❌ 找不到索引文件 {index_file}，请先运行 2_Index_keywords.py 生成索引。")
        return None, None
    if not os.path.exists(keywords_file):
        print(f"⚠️ 未找到关键词文件 {keywords_file}，将无法显示文档标签。")
        doc_keywords = {}
    else:
        print("⏳ 正在加载倒排索引引擎入内存...")
        with open(index_file, 'r', encoding='utf-8') as f:
            inverted_index = json.load(f)
        with open(keywords_file, 'r', encoding='utf-8') as f:
            doc_keywords = json.load(f)
        print(f"✅ 引擎启动成功！当前索引词汇量: {len(inverted_index)}")
        return inverted_index, doc_keywords
    
    # 若关键词文件不存在，仍然返回索引，但 doc_keywords 为空
    with open(index_file, 'r', encoding='utf-8') as f:
        inverted_index = json.load(f)
    print(f"✅ 引擎启动成功！当前索引词汇量: {len(inverted_index)}")
    return inverted_index, {}

def get_stopwords():
    """加载停用词表：优先从 stopwords.txt 读取，再合并内置基础停用词"""
    stopwords = set()
    # 尝试加载外部停用词表
    if os.path.exists("stopwords.txt"):
        with open("stopwords.txt", 'r', encoding='utf-8') as f:
            for line in f:
                word = line.strip()
                if word:
                    stopwords.add(word)
        print(f"✅ 已加载外部停用词表，共 {len(stopwords)} 个词。")
    else:
        print("⚠️ 未找到 stopwords.txt，将仅使用内置停用词。")
    
    # 内置基础停用词（与 1_fenci.py 保持一致）
    builtin_stops = {
        "下面", "就让", "我们", "第一眼", "要么", "虽然", "之外", "近日",
        "可谓", "同时", "加上", "使得", "经过", "仍能", "一款", "一身",
        "此款", "多名", "各种", "一直", "不由", "轻轻", "绝不", "他们",
        "各自", "的", "了", "在", "是", "我", "有", "和", "就", "不", "人", "都", "一",
        # 额外补充一些常见停用词（可根据需要调整）
        "一个", "时间", "让", "会", "到", "也", "还", "这", "那", "去", "能", "可以", "将", "被"
    }
    stopwords.update(builtin_stops)
    return stopwords

def process_query(query, stopwords):
    """对用户的搜索输入进行与建库时一致的分词处理"""
    query = re.sub(r'(\d+)[ \t]+([a-zA-Z]+)', r'\1\2', query.strip())
    words = pseg.cut(query)
    
    allowed_pos = {'n', 'nr', 'ns', 'nt', 'nz', 'v', 'vn', 'a', 'eng', 'm', 'i', 'l'}
    query_terms = []
    
    for word, flag in words:
        word = word.strip()
        if not word or word in stopwords:
            continue
        # 保留长度>=2的中文词，或英文/数字组合
        if len(word) < 2 and not re.match(r'^[a-zA-Z]+$', word):
            continue
        if re.match(r'^\d+$', word):
            continue
        if flag in allowed_pos or re.match(r'^[a-zA-Z0-9]+$', word):
            query_terms.append(word)
    return query_terms

# ==========================================
# 2. 核心检索打分模型 (基于向量空间模型的简化版)
# ==========================================
def search(query_terms, inverted_index, top_k=5):
    if not query_terms:
        return []
        
    doc_scores = {}
    
    for term in query_terms:
        if term in inverted_index:
            posting_list = inverted_index[term]
            for doc_id, item in posting_list.items():
                # 如果 item 是字典则取 'score'，否则直接当作数值（兼容旧格式）
                if isinstance(item, dict):
                    score = item.get("score", 0.0)
                else:
                    score = item if isinstance(item, (int, float)) else 0.0
                doc_scores[doc_id] = doc_scores.get(doc_id, 0.0) + score
                
    ranked_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
    return ranked_docs[:top_k]

# ==========================================
# 3. 终端交互与信息展示
# ==========================================
def main():
    inverted_index, doc_keywords = load_search_data()
    if inverted_index is None:
        return
        
    stopwords = get_stopwords()
    
    print("\n" + "="*50)
    print(" 🚀 欢迎使用中文新闻智能检索系统 🚀 ")
    print("   (输入 'exit' 或 'quit' 退出系统)")
    print("="*50 + "\n")
    
    while True:
        query = input("🔍 请输入搜索关键词: ").strip()
        if query.lower() in ['exit', 'quit']:
            print("👋 感谢使用，再见！")
            break
        if not query:
            continue
            
        query_terms = process_query(query, stopwords)
        print(f"\n🧠 [系统解析意图] -> 提取检索词: {query_terms}")
        
        if not query_terms:
            print("⚠️ 您的查询词都是停用词或标点，请尝试输入更有实质意义的词汇（如名词、动作）。\n")
            continue
            
        results = search(query_terms, inverted_index, top_k=5)
        
        if not results:
            print("📭 抱歉，未能找到与您的查询相关的新闻文档。\n")
        else:
            print(f"🎯 检索完成，为您找到最相关的 {len(results)} 篇文档:\n")
            for rank, (doc_id, score) in enumerate(results, 1):
                tags = doc_keywords.get(doc_id, [])
                tag_str = " | ".join(tags) if tags else "无标签"
                print(f"  {rank}. 【文档ID: {doc_id}】  ⚡ 相关度得分: {score:.4f}")
                print(f"      🏷️  核心标签: [{tag_str}]")
                print("-" * 45)
            print("\n")

if __name__ == "__main__":
    main()