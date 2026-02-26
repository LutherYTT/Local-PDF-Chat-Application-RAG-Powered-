import pdfplumber
from langchain_text_splitters import RecursiveCharacterTextSplitter
import re
from core.question_generator import QuestionGenerator

class PDFProcessor:
    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,          
            chunk_overlap=200,       
            separators=["\n\n", "\n", "。", "！", "？", ".", "!", "?", " ", ""]  # 双语分隔符
        )
        self.meta_keywords = ["作者", "編者", "主編", "著者", "Author", "Editor", "Writer"]
        self.question_generator = QuestionGenerator()
        self.processed_chunks = []  
        self.en_fixed_headings = [
            "Abstract", "Introduction", "Conclusion", "Summary", 
            "Preface", "Appendix", "Method", "Result", "Discussion"
        ]

    def extract_meta_info(self, text):
        meta_info = {}
        author_pattern = r'(作者|编者|主编|著者|Author|Editor|Writer)\s*[:：]\s*([^\n]+)'
        author_match = re.search(author_pattern, text, re.I)
        if author_match:
            meta_info["author"] = author_match.group(2).strip()
        title_pattern = r'^([^\n]{1,50})$'
        title_match = re.search(title_pattern, text)
        if title_match and len(title_match.group(1).strip()) > 2:
            meta_info["title"] = title_match.group(1).strip()
        return meta_info

    # 檢測文字主語言（中/英）
    def detect_language(self, text):
        """
        偵測文字主語言：包含中文漢字則為zh，否則為en
        :param text: 待檢測文本
        :return: "zh"（中文）/ "en"（英文）
        """
        chinese_char_pattern = r'[\u4e00-\u9fff]'
        if re.search(chinese_char_pattern, text):
            return "zh"
        return "en"

    def load_pdf_with_pages(self, pdf_path, progress_callback=None):
        temp_chunks = []
        current_heading = "未分類章節 / Unclassified Chapter"
        doc_meta = {}  
        doc_language = "en"

        try:
            with pdfplumber.open(pdf_path) as pdf:
                total_pages = len(pdf.pages)
                # 第一步：提取元信息+檢測文檔主語言
                full_doc_text = ""
                if progress_callback:
                    progress_callback(0.05, f"提取文檔元信息（前{min(5, total_pages)}頁）")
                for page_idx in range(min(5, total_pages)):
                    page = pdf.pages[page_idx]
                    text = page.extract_text() or ""
                    full_doc_text += text + "\n"
                    page_meta = self.extract_meta_info(text)
                    doc_meta.update(page_meta)
                    if doc_meta.get("author") and doc_meta.get("title"):
                        break
                doc_language = self.detect_language(full_doc_text)

                # 第二步：逐頁處理文本（細化進度）
                total_process_steps = total_pages * 4  # 每頁4个核心步骤
                completed_steps = 0
                
                if progress_callback:
                    progress_callback(0.1, f"開始逐頁處理（共{total_pages}頁）")
                
                for page_idx, page in enumerate(pdf.pages):
                    page_num = page_idx + 1
                    
                    # 2.1 提取頁面文本
                    if progress_callback:
                        progress = 0.1 + (completed_steps / total_process_steps) * 0.75
                        progress_callback(progress, f"提取第{page_num}頁文本")
                    text = page.extract_text() or ""
                    header_text = page.crop((0, 0, page.width, page.height*0.1)).extract_text() or ""
                    footer_text = page.crop((0, page.height*0.9, page.width, page.height)).extract_text() or ""
                    full_page_text = f"{header_text}\n{text}\n{footer_text}".strip()
                    completed_steps += 1
                    if not full_page_text:
                        continue
                    
                    # 2.2 標題識別
                    if progress_callback:
                        progress = 0.1 + (completed_steps / total_process_steps) * 0.75
                        progress_callback(progress, f"識別第{page_num}頁章節標題")
                    lines = full_page_text.split("\n")
                    for line in lines:
                        clean_line = line.strip()
                        if self.is_heading(clean_line):
                            current_heading = clean_line
                    completed_steps += 1
                    
                    # 2.3 文本分區塊
                    if progress_callback:
                        progress = 0.1 + (completed_steps / total_process_steps) * 0.75
                        progress_callback(progress, f"拆分第{page_num}頁文本为區塊")
                    chunks = self.text_splitter.split_text(full_page_text)
                    completed_steps += 1
                    
                    # 2.4 產生虛擬問題並增強內容
                    for c in chunks:
                        if progress_callback:
                            progress = 0.1 + (completed_steps / total_process_steps) * 0.75
                            progress_callback(progress, f"為第{page_num}頁文本區塊生成虛擬問題")
                        meta_info = {
                            "doc_title": doc_meta.get("title", "未找到 / Not Found"),
                            "heading": current_heading,
                            "page": page_num,
                            "author": doc_meta.get("author", "未找到 / Not Found"),
                            "doc_language": doc_language
                        }
                        virtual_questions = self.question_generator.generate_virtual_questions(c, meta_info)
                        enhanced_content = f"{c}\n【虛擬檢索問題 / Virtual Query】：{' | '.join(virtual_questions)}"
                        chunk_meta = {
                            "content": enhanced_content,
                            "page": page_num,
                            "heading": current_heading,
                            "total_pages": total_pages,
                            "author": doc_meta.get("author", "未找到 / Not Found"),
                            "doc_title": doc_meta.get("title", "未找到 / Not Found"),
                            "virtual_questions": virtual_questions,
                            "doc_language": doc_language,
                            "chunk_idx": 0,
                            "total_chunks": 0
                        }
                        temp_chunks.append(chunk_meta)
                        completed_steps += 1
            
            # 全局索引賦值
            total_chunks = len(temp_chunks)
            if progress_callback:
                progress_callback(0.9, "統一賦值文本區塊索引")
            for idx, chunk in enumerate(temp_chunks):
                chunk["chunk_idx"] = idx
                chunk["total_chunks"] = total_chunks
            
            if progress_callback:
                progress_callback(1.0, "文本分區塊处理完成")
            
            self.processed_chunks = temp_chunks
            return self.processed_chunks

        except Exception as e:
            print(f"PDF解析錯誤 / PDF Parse Error: {e}")
            self.processed_chunks = []
            return []

    def is_heading(self, line):
        clean_line = line.strip()
        if len(clean_line) < 2 or len(clean_line) > 200:
            return False
        if any(kw in clean_line for kw in self.meta_keywords):
            return False
        line_lower = clean_line.lower()
        line_upper = clean_line.upper()

        if any(h.lower() == line_lower or h.upper() == line_upper for h in self.en_fixed_headings):
            return True
        if re.match(r'^[Cc]hapter\s*[0-9]+(\.[0-9]+)*', clean_line):
            return True
        if re.match(r'^[Ss]ection\s*[0-9]+(\.[0-9]+)*', clean_line):
            return True
        if re.match(r'^第[一二三四五六七八九十0-9]+[章节条节]', clean_line):
            return True
        if re.match(r'^[0-9]+\.[0-9\.]+[\s\-\_]*.*', clean_line):
            return True
        if clean_line.startswith(("#", "##", "###", "####")):
            return True
        if clean_line.isupper() and len(clean_line.split()) <= 10:
            return True
        return False