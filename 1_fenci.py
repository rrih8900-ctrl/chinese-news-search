import jieba
import jieba.posseg as pseg
import re
import os

# ==========================================
# 1. 配置自定义词典与领域词汇
# ==========================================
# 您可以创建一个 custom_dict.txt 文件，每行一个词。
# 这里为了方便演示，直接使用代码动态添加之前发现的易错词
custom_stops = {
        "下面", "就让", "我们", "第一眼", "要么", "虽然", "之外", "近日", 
        "可谓", "同时", "加上", "使得", "经过", "仍能", "一款", "一身", 
        "此款", "多名", "各种", "一直", "不由", "轻轻", "绝不", "他们", 
        "各自", "的", "了", "在", "是", "我", "有", "和", "就", "不", "人", "都", "一",
        "一个", "时间", "让", "会", "到", "也", "还", "这", "那", "去", "能", "可以", "将", "被"
    }
# 修正：原变量 custom_words 未定义，改为空列表（不添加任何词）
for word in []:
    jieba.add_word(word)

# ==========================================
# 2. 构建完善的停用词表
# ==========================================
def load_stopwords(filepath=None):
    stopwords = set()
# 1. 尝试从文件加载外部停用词
    # 修正：避免 filepath 为 None 时调用 os.path.exists
    if filepath is not None and os.path.exists(filepath):
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                word = line.strip()
                if word:
                    stopwords.add(word)
        print(f"✅ 成功加载外部停用词表，共 {len(stopwords)} 个词。")
    else:
        print(f"⚠️ 未找到 {filepath} 文件，将仅使用内置基础停用词。")
    
    # 补充我们在语料中发现的特定停用词（副词、代词、过渡词等）
    custom_stops = {
        "下面", "就让", "我们", "第一眼", "要么", "虽然", "之外", "近日", 
        "可谓", "同时", "加上", "使得", "经过", "仍能", "一款", "一身", 
        "此款", "多名", "各种", "一直", "不由", "轻轻", "绝不", "他们", 
        "各自", "的", "了", "在", "是", "我", "有", "和", "就", "不", "人", "都", "一"
    }
    stopwords.update(custom_stops)
    return stopwords

# ==========================================
# 3. 核心文本处理函数
# ==========================================
def process_text(text, stopwords):
    text = text.strip()
    if not text: return ""
    
    # 修复数字与英文单位断开的问题
    text = re.sub(r'(\d+)[ \t]+([a-zA-Z]+)', r'\1\2', text)
    
    words = pseg.cut(text)
    result = []
    
    allowed_pos = {'n', 'nr', 'ns', 'nt', 'nz', 'v', 'vn', 'a', 'eng', 'm'}
    
    for word, flag in words:
        word = word.strip()
        
        # 1. 基础过滤：为空或在停用词表中则跳过
        if not word or word in stopwords:
            continue
            
        # 2. 【核心修复 A】过滤掉所有长度为 1 的中文单字（如：年、月、日、说）
        # 保留纯英文字母（如 U 盘，X 战警）
        if len(word) < 2 and not re.match(r'^[a-zA-Z]+$', word):
            continue
            
        # 3. 【核心修复 B】过滤掉纯数字（如：0, 1, 2010, 99）
        # 搜索系统通常不需要对纯数字建立索引，但我们要保留字母与数字的组合(如 MP3, SA658)
        if re.match(r'^\d+$', word):
            continue
            
        # 4. 词性放行判断
        if flag in allowed_pos or re.match(r'^[a-zA-Z0-9]+$', word):
            result.append(word)
                
    return " ".join(result)

# ==========================================
# 4. 主干逻辑
# ==========================================
def main():
    input_file = "data.txt"
    output_file = "fenci_optimized_result.txt"
    
    # 加载停用词（如果您有本地 txt 词表，传入路径，例如 load_stopwords("stop_words.txt")）
    stopwords = load_stopwords("stopwords.txt")
    
    try:
        with open(input_file, 'r', encoding='utf-8') as fin, \
             open(output_file, 'w', encoding='utf-8') as fout:
            
            for line in fin:
                # 保留文档头部标识，如 【文档ID: 0  类别: 科技】
                if line.startswith("【文档ID"):
                    fout.write(line)
                    continue
                
                # 处理正文内容
                processed_line = process_text(line, stopwords)
                if processed_line:
                    fout.write(processed_line + "\n")
                    
        print(f"✅ 分词优化处理完成！结果已保存至 {output_file}")
        
    except FileNotFoundError:
        print(f"❌ 找不到输入文件 {input_file}，请检查路径。")

if __name__ == "__main__":
    main()