import re

def parse_page_query(question):
    """
    【雙語版】解析頁碼，支持中/英文/混合表述，兼容大小寫
    中文：第3頁、3頁、4-6頁、第4到6頁
    英文：page 3、Page 10、pages 4-6、page 4 to 6、10th page
    混合：第5page、page第6頁
    :return: 頁碼列表，無則返回[]
    """
    question = question.replace(" ", "").lower()  # 統一小寫+去空格，兼容大小寫
    # 英文匹配：pageX、pagesX-Y、pageXtoY、Xthpage
    en_range_pattern = r'page(s)?(\d+)[:\-to](\d+)'
    en_single_pattern = r'page(\d+)|(\d+)thpage'
    # 中文匹配：第X頁、X頁、X-Y頁、X到Y頁
    cn_range_pattern = r'第?(\d+)[:\-到至](\d+)頁'
    cn_single_pattern = r'第?(\d+)頁'

    page_nums = []
    # 優先匹配連續頁碼（英→中）
    en_range_match = re.search(en_range_pattern, question)
    if en_range_match:
        start = int(en_range_match.group(2))
        end = int(en_range_match.group(3))
        if start <= end:
            page_nums = list(range(start, end + 1))
        return page_nums
    cn_range_match = re.search(cn_range_pattern, question)
    if cn_range_match:
        start = int(cn_range_match.group(1))
        end = int(cn_range_match.group(2))
        if start <= end:
            page_nums = list(range(start, end + 1))
        return page_nums
    # 匹配單頁碼（英→中）
    en_single_match = re.search(en_single_pattern, question)
    if en_single_match:
        num = en_single_match.group(1) or en_single_match.group(2)
        page_nums = [int(num)]
        return page_nums
    cn_single_match = re.search(cn_single_pattern, question)
    if cn_single_match:
        page_nums = [int(cn_single_match.group(1))]
    return page_nums

def parse_chapter_query(question):
    """
    【雙語版】解析章節+最後一句，支持中/英文章節名，兼容大小寫
    中文章節：第2章、第二章、第3節、摘要
    英文章節：Chapter 1、chapter2、Abstract、Introduction、Conclusion、Section 3
    最後一句：最後一句、最後一句話、final sentence、last sentence、the last line
    :return: (章節名, 是否查询最後一句)，無則("", False)
    """
    question = question.replace(" ", "").lower()
    # 【最後一句】雙語關鍵詞匹配
    last_sent_keywords = [
        "最後一句", "最後一句話", "finalsentence", 
        "lastsentence", "thelastline", "lastline"
    ]
    is_last_sent = any(kw in question for kw in last_sent_keywords)

    # 【英文章節】匹配：chapterX、sectionX、abstract、introduction等固定標題
    en_chapter_pattern = r'(chapter|section)(\d+)'  # Chapter1/section3
    en_fixed_heading = r'(abstract|introduction|conclusion|summary|preface|appendix)'  # 英文固定章節名
    # 【中文章節】匹配：第X章、第X節、第二章
    cn_chapter_pattern = r'(第\d+[章節條節])|(第[一二三四五六七八九十百]+[章節條節])'

    chapter_name = ""
    # 1. 匹配英文固定章節名（Abstract/Introduction等，優先返回原大小寫）
    en_fixed_match = re.search(en_fixed_heading, question)
    if en_fixed_match:
        chapter_name = en_fixed_match.group(1).capitalize()  # 統一首字母大寫（Abstract/Introduction）
    # 2. 匹配英文數字章節（Chapter1→Chapter 1）
    en_num_match = re.search(en_chapter_pattern, question)
    if not chapter_name and en_num_match:
        type_ = en_num_match.group(1).capitalize()  # Chapter/Section
        num = en_num_match.group(2)
        chapter_name = f"{type_} {num}"
    # 3. 匹配中文章節
    cn_num_match = re.search(cn_chapter_pattern, question)
    if not chapter_name and cn_num_match:
        chapter_name = cn_num_match.group(1) or cn_num_match.group(2)

    # 兼容原問題中的章節名大小寫（如用户查Abstract，匹配PDF中的ABSTRACT/abstract）
    return chapter_name, is_last_sent

def extract_last_sentence(text):
    """
    【雙語版】提取最後一个完整句子，支持中/英文標點，自動過濾虛擬問題標記
    中文：。！？  英文：. ? !
    """
    if not text or len(text.strip()) < 1:
        return "未找到有效句子 / No valid sentence found"
    # 清理幹擾内容（虛擬問題標記、多餘空格）
    text = text.replace("【虛擬检索問題】：", "").strip()
    text = re.sub(r'\s+', ' ', text)  # 多个空格合并為一个，适配英文

    # 雙語分割符：中文+英文結束標點，保留標點
    bilingual_separators = r'([。！？.?!])'
    parts = re.split(bilingual_separators, text)
    # 重組完整句子（内容+標點）
    sentences = []
    for i in range(0, len(parts)-1, 2):
        content = parts[i].strip()
        symbol = parts[i+1]
        if content and symbol in ["。", "！", "？", ".", "?", "!"]:
            sentences.append(f"{content}{symbol}")
    # 處理最後一段無標點的情况（視為完整句子）
    if len(parts) % 2 != 0 and parts[-1].strip():
        sentences.append(parts[-1].strip())
    # 無有效句子時返回雙語提示
    return sentences[-1] if sentences else "未找到有效句子 / No valid sentence found"

def is_special_query(question):
    """
    【雙語版】判斷是否為特殊查詢：頁碼/章節最後一句（中/英文）
    :return: (is_page_query, is_chapter_last, page_nums, chapter_name)
    """
    page_nums = parse_page_query(question)
    chapter_name, is_last_sent = parse_chapter_query(question)
    is_page = len(page_nums) > 0
    is_chapter_last = chapter_name != "" and is_last_sent
    return is_page, is_chapter_last, page_nums, chapter_name