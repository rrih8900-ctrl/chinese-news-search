import streamlit as st
import json
import jieba.posseg as pseg
import re
import pandas as pd
import math
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt

# 尝试导入 sklearn，若失败则设置标志
try:
    from sklearn.manifold import TSNE
    from sklearn.cluster import KMeans
    from sklearn.feature_extraction.text import TfidfVectorizer
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

# ==========================================
# 0. 页面全局配置
# ==========================================
st.set_page_config(
    page_title="中文新闻智能检索与分析系统",
    page_icon="📰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================
# 1. 数据加载与缓存（适配每行一篇的格式）
# ==========================================
@st.cache_resource
def load_data():
    try:
        with open("inverted_index.json", 'r', encoding='utf-8') as f:
            inverted_index = json.load(f)
        with open("doc_keywords.json", 'r', encoding='utf-8') as f:
            doc_keywords = json.load(f)
    except Exception as e:
        st.error(f"❌ 加载索引文件失败：{e}")
        return None, None, None

    # 加载原始新闻全文（data.txt 每行：类别 空格 正文）
    original_texts = {}
    try:
        with open("data.txt", 'r', encoding='utf-8') as f:
            for idx, line in enumerate(f, start=1):
                line = line.strip()
                if line:
                    original_texts[str(idx)] = line
    except Exception as e:
        st.warning(f"⚠️ 加载原始新闻文本失败，将使用分词结果代替：{e}")

    if len(original_texts) == 0:
        try:
            with open("fenci_optimized_result.txt", 'r', encoding='utf-8') as f:
                for idx, line in enumerate(f, start=1):
                    line = line.strip()
                    if line:
                        original_texts[str(idx)] = line
        except Exception as e:
            st.error(f"❌ 加载分词结果文件也失败：{e}")

    doc_keywords = {str(k): v for k, v in doc_keywords.items()}
    return inverted_index, doc_keywords, original_texts

inverted_index, doc_keywords, original_texts = load_data()

# 提取文档类别（用于主题词分析，但不再单独统计分布）
def extract_category(doc_text):
    if not doc_text:
        return "未知"
    parts = doc_text.split(maxsplit=1)
    return parts[0] if parts else "未知"

# ==========================================
# 2. 分词与查询处理
# ==========================================
def process_query(query: str):
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

# ==========================================
# 3. 高级检索（布尔/短语/TF-IDF）
# ==========================================
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

def advanced_search(query_string, inverted_index, process_query_func, default_op="OR"):
    query_string = query_string.strip()
    if not query_string:
        return []
    phrase_match = re.search(r'"(.*?)"', query_string)
    if phrase_match:
        phrase_text = phrase_match.group(1)
        phrase_words = process_query_func(phrase_text)
        if not phrase_words:
            return []
        results_dict = execute_phrase_search(phrase_words, inverted_index)
        return sorted(results_dict.items(), key=lambda x: x[1], reverse=True)

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
    doc_scores = {}
    if op == "OR":
        for term, posting in term_postings.items():
            for doc_id, info in posting.items():
                doc_scores[doc_id] = doc_scores.get(doc_id, 0.0) + info['score']
    elif op == "AND":
        if not processed_terms:
            return []
        common_docs = set(term_postings[processed_terms[0]].keys())
        for term in processed_terms[1:]:
            common_docs &= set(term_postings[term].keys())
        for doc_id in common_docs:
            doc_scores[doc_id] = sum(term_postings[t][doc_id]['score'] for t in processed_terms)
    elif op == "NOT":
        if len(processed_terms) < 2:
            return []
        positive = set(term_postings[processed_terms[0]].keys())
        negative = set(term_postings[processed_terms[1]].keys())
        valid = positive - negative
        for doc_id in valid:
            doc_scores[doc_id] = term_postings[processed_terms[0]][doc_id]['score']
    return sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)

# ==========================================
# 4. 辅助函数
# ==========================================
def highlight_text(text, keywords):
    if not text:
        return ""
    highlighted = text
    for kw in keywords:
        pattern = re.compile(re.escape(kw), re.IGNORECASE)
        highlighted = pattern.sub(
            f"<mark style='background-color: #ffe066; font-weight: bold; padding: 0 4px; border-radius: 3px;'>{kw}</mark>",
            highlighted
        )
    return highlighted

# ==========================================
# 5. 侧边栏
# ==========================================
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3226/3226016.png", width=100)
    st.title("系统控制台")
    st.markdown("---")
    user_role = st.selectbox("👤 当前登录角色", ["普通用户", "研究人员", "数据分析师"],
                             help="普通用户默认OR，研究人员默认AND，数据分析师可访问深度分析工具")
    st.markdown("---")
    st.subheader("⚙️ 检索设置")
    page_size = st.slider("📄 每页显示结果数量", min_value=5, max_value=50, value=10)
    with st.expander("📖 检索功能说明", expanded=False):
        st.markdown(f"""
        **当前角色：{user_role}**
        - 普通用户：未指定操作符时默认 **OR**（覆盖面广）
        - 研究人员：未指定操作符时默认 **AND**（精确匹配）
        - 数据分析师：检索行为与普通用户相同，但可访问「数据分析」标签页中的深度分析工具
        **支持的检索语法：**
        - 普通词检索：自动应用默认逻辑
        - 布尔检索：`AND` / `OR` / `NOT`，例如 `科技 AND 手机`
        - 短语检索：英文双引号，例如 `"索爱 MP3"`
        """)

# ==========================================
# 6. 主界面
# ==========================================
if inverted_index is None:
    st.stop()

tab1, tab2 = st.tabs(["🔍 文档检索 (Search)", "📊 数据分析 (Analytics)"])

# ------------------ 检索标签页 ------------------
with tab1:
    st.markdown("### 📰 智能检索中心")
    col1, col2 = st.columns([5, 1])
    with col1:
        query = st.text_input(
            "请输入检索式（支持布尔/短语）",
            key="search_input",
            placeholder="示例: 科技 AND 手机   |   \"索爱 MP3\"   |  湖人",
            label_visibility="collapsed"
        )
    with col2:
        search_btn = st.button("🚀 检索", use_container_width=True)

    # 检索日志记录
    if 'search_log' not in st.session_state:
        st.session_state.search_log = []

    if 'full_results' not in st.session_state:
        st.session_state.full_results = []
        st.session_state.current_page = 1
        st.session_state.last_query = ""
        st.session_state.last_role = user_role

    if user_role != st.session_state.last_role:
        st.session_state.full_results = []
        st.session_state.current_page = 1
        st.session_state.last_query = ""
        st.session_state.last_role = user_role

    if search_btn and query:
        with st.spinner("正在执行高级检索..."):
            default_op = "AND" if user_role == "研究人员" else "OR"
            full_results = advanced_search(query, inverted_index, process_query, default_op)
            st.session_state.full_results = full_results
            st.session_state.current_page = 1
            st.session_state.last_query = query
            from datetime import datetime
            st.session_state.search_log.insert(0, (query, len(full_results), datetime.now().strftime("%H:%M:%S")))

    full_results = st.session_state.full_results
    total_count = len(full_results)
    total_pages = math.ceil(total_count / page_size) if total_count > 0 else 1

    if total_count > 0:
        col_prev, col_page_info, col_next = st.columns([1, 3, 1])
        with col_prev:
            if st.button("◀ 上一页", disabled=(st.session_state.current_page <= 1)):
                st.session_state.current_page -= 1
                st.rerun()
        with col_page_info:
            st.markdown(f"<div style='text-align: center;'>第 {st.session_state.current_page} / {total_pages} 页 （共 {total_count} 条结果）</div>", unsafe_allow_html=True)
        with col_next:
            if st.button("下一页 ▶", disabled=(st.session_state.current_page >= total_pages)):
                st.session_state.current_page += 1
                st.rerun()

    if total_count == 0:
        if search_btn and query:
            st.warning("📭 未找到相关文档，请尝试其他检索式。")
        else:
            st.info("💡 输入检索词后点击「检索」开始查询。")
    else:
        start_idx = (st.session_state.current_page - 1) * page_size
        end_idx = start_idx + page_size
        display_results = full_results[start_idx:end_idx]

        query_for_highlight = re.sub(r'"(.*?)"', r'\1', st.session_state.last_query)
        highlight_terms = process_query(query_for_highlight)

        st.success(f"🎯 检索完成！共找到 {total_count} 篇相关文档（当前第 {st.session_state.current_page} / {total_pages} 页）")
        st.markdown("---")

        for rank_in_page, (doc_id, score) in enumerate(display_results, start=1):
            global_rank = start_idx + rank_in_page
            tags = doc_keywords.get(doc_id, [])
            if user_role == "普通用户":
                display_tags = tags[:3] if len(tags) > 3 else tags
            else:
                display_tags = tags
            tag_str = " | ".join([f"`{t}`" for t in display_tags]) if display_tags else "无标签"
            with st.expander(f"Top {global_rank}. 【文档ID: {doc_id}】 ⚡ 相关度: {score:.4f}", expanded=(global_rank == 1)):
                st.markdown(f"**🏷️ 核心主题**: {tag_str}")
                raw_text = original_texts.get(doc_id, "暂无原始文本数据")
                highlighted_html = highlight_text(raw_text, highlight_terms)
                st.markdown("**📄 原文详情:**")
                st.markdown(
                    f"<div style='line-height:1.6; color:#333; background:#f9f9f9; padding:15px; border-radius:5px;'>{highlighted_html}</div>",
                    unsafe_allow_html=True
                )

# ------------------ 数据分析标签页（数据分析师专属工具） ------------------
with tab2:
    st.markdown("### 📈 语料库全局数据洞察")
    col1, col2 = st.columns(2)
    col1.metric("总文档数", len(original_texts))
    col2.metric("独立词汇总数", len(inverted_index))

    # 基础高频词（所有角色可见）
    st.markdown("#### 🏆 全局高频词 (Top 20)")
    word_freq = []
    for word, posting in inverted_index.items():
        word_freq.append({"词汇": word, "文档覆盖数": len(posting)})
    df_freq = pd.DataFrame(word_freq).sort_values(by="文档覆盖数", ascending=False).head(20)
    st.bar_chart(data=df_freq.set_index("词汇"))

    # 如果角色是数据分析师，显示高级分析工具（已移除类别分布统计）
    if user_role == "数据分析师":
        st.markdown("---")
        st.markdown("## 🔬 数据分析师专属工具")

        # 1. 各类别主题词（基于TF-IDF）
        with st.expander("🏷️ 各类别代表性关键词", expanded=True):
            doc_to_cat = {doc_id: extract_category(text) for doc_id, text in original_texts.items()}
            cat_term_score = defaultdict(lambda: defaultdict(float))
            for term, posting in inverted_index.items():
                for doc_id, info in posting.items():
                    cat = doc_to_cat.get(doc_id)
                    if cat:
                        cat_term_score[cat][term] += info['score']
            for cat in sorted(cat_term_score.keys()):
                top_terms = sorted(cat_term_score[cat].items(), key=lambda x: x[1], reverse=True)[:5]
                terms_str = " | ".join([f"{t}({s:.2f})" for t, s in top_terms])
                st.markdown(f"**{cat}** : {terms_str}")

        # 2. 文档聚类可视化（动态聚类 + t-SNE）
        with st.expander("🔘 文档聚类可视化 (t-SNE)", expanded=False):
            if not SKLEARN_AVAILABLE:
                st.warning("⚠️ 未安装 scikit-learn，无法进行聚类可视化。请在终端执行以下命令安装：\n```\npip install scikit-learn\n```")
            else:
                if st.button("运行聚类并生成散点图"):
                    with st.spinner("正在计算文档向量和降维，请稍候..."):
                        doc_ids = list(original_texts.keys())
                        docs_text = [original_texts[d] for d in doc_ids]
                        # 使用 jieba 分词（若未导入则动态导入）
                        import jieba
                        docs_words = [" ".join(jieba.lcut(t)) for t in docs_text]
                        vectorizer = TfidfVectorizer(max_features=500)
                        X = vectorizer.fit_transform(docs_words).toarray()
                        tsne = TSNE(n_components=2, random_state=42, perplexity=30)
                        X_tsne = tsne.fit_transform(X)
                        n_clusters = min(8, len(doc_ids)//10)
                        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
                        labels = kmeans.fit_predict(X)
                        fig, ax = plt.subplots(figsize=(10, 6))
                        scatter = ax.scatter(X_tsne[:, 0], X_tsne[:, 1], c=labels, cmap='tab10', alpha=0.7)
                        ax.set_title(f"文档聚类可视化 (t-SNE, {n_clusters} 个簇)")
                        ax.set_xlabel("t-SNE 1")
                        ax.set_ylabel("t-SNE 2")
                        plt.colorbar(scatter, label="簇编号")
                        st.pyplot(fig)
                        st.caption(f"聚类数: {n_clusters}，基于 TF‑IDF 特征 (top 500) 计算。")

        # 3. 检索日志分析
        with st.expander("📜 检索日志分析", expanded=False):
            if len(st.session_state.search_log) == 0:
                st.info("暂无检索记录。请先在检索标签页执行查询。")
            else:
                log_df = pd.DataFrame(st.session_state.search_log, columns=["查询词", "结果数量", "时间"])
                st.dataframe(log_df, use_container_width=True)
                if st.button("清空日志"):
                    st.session_state.search_log = []
                    st.rerun()
                from collections import Counter
                query_counts = Counter([log[0] for log in st.session_state.search_log])
                if query_counts:
                    st.markdown("**热门查询 Top 5**")
                    hot_df = pd.DataFrame(query_counts.most_common(5), columns=["查询词", "出现次数"])
                    st.bar_chart(hot_df.set_index("查询词"))

        # 4. 文档详情浏览器
        with st.expander("🔍 文档详情浏览器", expanded=False):
            doc_id_input = st.text_input("输入文档ID (数字)", value="1")
            if st.button("查看文档"):
                if doc_id_input in original_texts:
                    text = original_texts[doc_id_input]
                    st.markdown(f"**文档ID: {doc_id_input}**")
                    st.markdown(f"**原文**: {text}")
                    tags = doc_keywords.get(doc_id_input, [])
                    st.markdown(f"**关键词标签**: {', '.join(tags) if tags else '无'}")
                    terms_scores = []
                    for term, posting in inverted_index.items():
                        if doc_id_input in posting:
                            terms_scores.append((term, posting[doc_id_input]['score']))
                    terms_scores.sort(key=lambda x: x[1], reverse=True)
                    st.markdown("**文档内高TF-IDF词**")
                    st.write(terms_scores[:10])
                else:
                    st.error("文档ID不存在")

    else:
        st.info("🔒 当前角色为「{}」，如需查看高级分析（主题词、聚类可视化、检索日志等），请切换至「数据分析师」角色。".format(user_role))