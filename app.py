import streamlit as st
import json
import jieba.posseg as pseg
import re
import pandas as pd
import math
import numpy as np
from collections import defaultdict
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import font_manager
import altair as alt

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
# 中文字体全局配置（防止 matplotlib 中文乱码）
# ==========================================
def _setup_chinese_font():
    """从系统中查找支持中文的字体并应用到 matplotlib 全局。"""
    candidate_fonts = [
        "Microsoft YaHei", "SimHei", "SimSun", "PingFang SC",
        "Heiti SC", "WenQuanYi Micro Hei", "WenQuanYi Zen Hei",
        "Noto Sans CJK SC", "Noto Sans CJK", "Source Han Sans CN",
        "Arial Unicode MS",
    ]
    available = {f.name for f in font_manager.fontManager.ttflist}
    chosen = None
    for name in candidate_fonts:
        if name in available:
            chosen = name
            break
    if chosen is None:
        # 退化：尝试用任意包含 CJK / Chinese / YaHei / Hei 的字体
        for f in font_manager.fontManager.ttflist:
            lname = f.name.lower()
            if "cjk" in lname or "chinese" in lname or "yahei" in lname or "hei" in lname or "pingfang" in lname:
                chosen = f.name
                break
    if chosen is None:
        chosen = "DejaVu Sans"  # 最终回退
    plt.rcParams['font.sans-serif'] = [chosen, "DejaVu Sans"]
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['axes.unicode_minus'] = False
    return chosen

_CHINESE_FONT_NAME = _setup_chinese_font()

# 尝试导入 sklearn，若失败则设置标志
try:
    from sklearn.manifold import TSNE
    from sklearn.cluster import KMeans
    from sklearn.feature_extraction.text import TfidfVectorizer
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

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
        return None, None, None, None

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
    # 计算文档向量模长
    doc_norms = {}
    for doc_id in original_texts.keys():
        norm_sq = 0.0
        for term, posting in inverted_index.items():
            if doc_id in posting:
                score = posting[doc_id]['score'] if isinstance(posting[doc_id], dict) else posting[doc_id]
                norm_sq += score * score
        doc_norms[doc_id] = math.sqrt(norm_sq) if norm_sq > 0 else 1.0
    return inverted_index, doc_keywords, original_texts, doc_norms

inverted_index, doc_keywords, original_texts, doc_norms = load_data()

import joblib
import jieba

@st.cache_resource
def load_classification_models():
    try:
        vectorizer = joblib.load("tfidf_vectorizer.pkl")
        nb_model = joblib.load("nb_classifier.pkl")
        svm_model = joblib.load("svm_classifier.pkl")
        return vectorizer, nb_model, svm_model
    except:
        return None, None, None

tfidf_vec, nb_clf, svm_clf = load_classification_models()

# 加载聚类结果映射（文档ID -> 簇编号）
@st.cache_resource
def load_cluster_map():
    try:
        df = pd.read_csv("doc_clusters.csv", encoding="utf-8")
        return {str(row['文档ID']): int(row['簇编号']) for _, row in df.iterrows()}
    except Exception as e:
        st.warning(f"未找到聚类结果文件 doc_clusters.csv，相关推荐功能不可用。错误: {e}")
        return {}

doc_cluster_map = load_cluster_map()

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

def advanced_search(query_string, inverted_index, process_query_func, original_texts, doc_norms, default_op="OR"):
    """
    返回 [(doc_id, cosine_similarity), ...] 按相似度降序
    """
    query_string = query_string.strip()
    if not query_string:
        return []

    # 短语检索（保留原位置评分）
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
    N = len(original_texts)   # 总文档数
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
        - 普通用户：未指定布尔操作符时默认 **OR**（覆盖面广）
        - 研究人员：未指定布尔操作符时默认 **AND**（精确匹配）
        - 数据分析师：检索行为与普通用户相同，但可访问「数据分析」标签页中的深度分析工具

        **排序算法：余弦相似度（Cosine Similarity）**  
        系统基于 TF‑IDF 构建文档向量，计算查询向量与每篇文档向量的余弦夹角，值越接近1表示越相关。该排序方式能有效平衡词频与文档长度，相比简单分数累加更符合信息检索理论。

        **支持的检索语法:**
        - 普通词检索：自动应用默认逻辑（OR / AND），
        - 布尔检索：AND (*)/ OR(+) / NOT，例如 篮球 OR 基金（或 篮球 + 基金）
        - 短语检索：英文双引号，例如 "篮球比赛"（仍使用原始位置评分）。

        **结果展示：** 显示原始新闻全文，查询关键词高亮，相似度得分即为余弦值。检索结果下方将展示同一聚类簇的相关新闻推荐。
        """)

# ==========================================
# 6. 主界面
# ==========================================
if inverted_index is None:
    st.stop()

# 创建三个标签页：文档检索、数据分析、文本分类预测
tab1, tab2, tab3 = st.tabs(["🔍 文档检索 (Search)", "📊 数据分析 (Analytics)", "🔮 文本分类预测"])

# ------------------ 检索标签页 ------------------
with tab1:
    st.markdown("### 📰 智能检索中心")
    col1, col2 = st.columns([5, 1])
    with col1:
        query = st.text_input(
            "请输入检索式（支持布尔/短语）",
            key="search_input",
            placeholder="示例：篮球 AND 金融 | \"篮球比赛\" | 湖人",
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
            full_results = advanced_search(query, inverted_index, process_query, original_texts, doc_norms, default_op)
            st.session_state.full_results = full_results
            st.session_state.current_page = 1
            st.session_state.last_query = query
            from datetime import datetime
            st.session_state.search_log.insert(0, (query, len(full_results), datetime.now().strftime("%H:%M:%S")))

    # 从 session_state 获取结果（确保始终有定义）
    full_results = st.session_state.full_results
    total_count = len(full_results)
    total_pages = math.ceil(total_count / page_size) if total_count > 0 else 1

    # 分页控件（仅在 results 非空时显示）
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

    # 结果显示
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
        # 过滤掉布尔操作符 OR / AND / NOT（避免在原文中被错误高亮）
        highlight_terms = [t for t in process_query(query_for_highlight)
                           if t.upper() not in ("OR", "AND", "NOT")]

        st.success(f"🎯 检索完成！共找到 {total_count} 篇相关文档（当前第 {st.session_state.current_page} / {total_pages} 页）")
        st.markdown("---")
        st.caption("💡 排序依据：余弦相似度（基于 TF‑IDF 向量）")

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

                # ----- 新增：同簇相关新闻推荐 -----
                if doc_cluster_map:
                    cluster_id = doc_cluster_map.get(doc_id, None)
                    if cluster_id is not None:
                        # 找出同簇的其他文档ID（排除自身）
                        same_cluster_docs = [d for d, c in doc_cluster_map.items() if c == cluster_id and d != doc_id]
                        if same_cluster_docs:
                            with st.expander(f"📂 同簇相关新闻（簇 {cluster_id}，共 {len(same_cluster_docs)} 篇）", expanded=False):
                                # 限制最多显示5篇
                                for other_id in same_cluster_docs[:5]:
                                    other_text = original_texts.get(other_id, "无内容")
                                    # 提取简短标题（取前60字）
                                    short_title = other_text[:60] + "..." if len(other_text) > 60 else other_text
                                    # 使用 st.write 显示，添加点击按钮查看全文
                                    st.markdown(f"**相关文档 ID: {other_id}**")
                                    st.markdown(f"📄 {short_title}")
                                    if st.button(f"查看全文", key=f"cluster_{doc_id}_{other_id}"):
                                        st.markdown(f"**原文:** {other_text}")
                                    st.markdown("---")
                # ----------------------------------

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
    # 使用 Altair 绘制交互式条形图，保持动态效果，横轴标签横向显示，去掉边框
    chart = alt.Chart(df_freq).mark_bar(color="#1f77b4").encode(
        x=alt.X("词汇:N", axis=alt.Axis(labelAngle=0, title="词汇")),
        y=alt.Y("文档覆盖数:Q", axis=alt.Axis(title="文档覆盖数"))
    ).properties(
        width=800,
        height=400,
        title="全局高频词 (Top 20)"
    ).configure_axis(
        grid=False
    ).configure_view(
        strokeOpacity=0  # 去掉边框
    )
    st.altair_chart(chart, use_container_width=True)

    # 如果角色是数据分析师，显示高级分析工具
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
                st.warning("⚠️ 未安装 scikit-learn，无法进行聚类可视化。请在终端执行：\n```\npip install scikit-learn\n```")
            else:
                if st.button("运行聚类并生成散点图"):
                    with st.spinner("正在计算文档向量和降维，请稍候..."):
                        # 中文字体已在模块顶部完成全局配置（_CHINESE_FONT_NAME）
                        doc_ids = list(original_texts.keys())
                        docs_text = [original_texts[d] for d in doc_ids]
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
                        ax.set_xlabel("t-SNE 第一维度")
                        ax.set_ylabel("t-SNE 第二维度")
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

# ------------------ 文本分类预测标签页（所有角色可见） ------------------
with tab3:
    st.markdown("### 🔮 文本分类预测")
    st.markdown("基于已训练的朴素贝叶斯和线性 SVM 模型，输入新闻正文即可自动预测类别。")
    
    if tfidf_vec is None:
        st.warning("⚠️ 分类模型未就绪，请先运行 `text_classification.py` 训练模型并保存 `tfidf_vectorizer.pkl`, `nb_classifier.pkl`, `svm_classifier.pkl` 文件。")
    else:
        # 输入区域
        new_text = st.text_area(
            "输入新闻正文（不含类别，支持长文本）",
            height=200,
            placeholder="例如：湖人队詹姆斯砍下30分，带领球队取胜，季后赛形势大好..."
        )
        col1, col2, col3 = st.columns([1, 1, 3])
        with col1:
            predict_btn = st.button("🔍 开始预测", use_container_width=True, type="primary")
        
        if predict_btn and new_text.strip():
            with st.spinner("正在预测..."):
                # 分词（与训练时保持一致：使用 jieba 分词，空格连接）
                words = jieba.lcut(new_text.strip())
                doc_str = " ".join(words)
                # 转换为 TF-IDF 向量（使用训练时的 vectorizer）
                X_new = tfidf_vec.transform([doc_str])
                # 预测
                pred_nb = nb_clf.predict(X_new)[0]
                pred_svm = svm_clf.predict(X_new)[0]
                
                # 显示结果
                st.markdown("---")
                st.markdown("#### 预测结果")
                col_a, col_b = st.columns(2)
                with col_a:
                    st.success(f"**朴素贝叶斯 (MultinomialNB)**\n\n预测类别：`{pred_nb}`")
                with col_b:
                    st.success(f"**线性支持向量机 (LinearSVC)**\n\n预测类别：`{pred_svm}`")
                
                # 可选：显示预测置信度（朴素贝叶斯可以输出概率）
                if hasattr(nb_clf, "predict_proba"):
                    proba = nb_clf.predict_proba(X_new)[0]
                    categories = nb_clf.classes_
                    prob_df = pd.DataFrame({"类别": categories, "概率": proba}).sort_values("概率", ascending=False)
                    st.markdown("**朴素贝叶斯各类别概率分布**")
                    st.bar_chart(prob_df.set_index("类别"))
        elif predict_btn and not new_text.strip():
            st.warning("请输入新闻正文。")

        # 示例新闻（方便测试）
        with st.expander("📝 点击查看示例新闻"):
            st.markdown("""
            **体育类示例：**  
            湖人队詹姆斯砍下30分，带领球队取胜，季后赛形势大好。

            **财经类示例：**  
            基金发行募集规模超百亿，市场反应火爆，投资者认购踊跃。

            **科技类示例：**  
            索尼发布新款微单相机，像素高达5000万，支持8K视频录制。

            **游戏类示例：**  
            王者荣耀新赛季开启，新增英雄和皮肤，玩家热情高涨。

            **娱乐类示例：**  
            周杰伦新专辑《最伟大的作品》上线，首日销量破百万。
            """)