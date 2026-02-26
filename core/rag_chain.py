from openai import OpenAI
from dotenv import load_dotenv
import os
import re

# 載入配置
load_dotenv()

# 輕量檢測查詢語言（中/英）
def detect_query_language(query):
    chinese_char_pattern = r'[\u4e00-\u9fff]'
    if re.search(chinese_char_pattern, query):
        return "zh"
    return "en"

class RAGEngine:
    def __init__(self):
        self.base_url = "https://api.deepseek.com"
    
    def _get_client(self):
        """動態取得 OpenAI 用戶端（每次都讀最新的 API Key）"""
        return OpenAI(
            api_key=os.getenv("DEEPSEEK_API_KEY"),
            base_url=self.base_url
        )

    def build_prompt(self, question, contexts):
        """
        1. 偵測查詢語言和文件語言（增加容錯）
        2. 引導AI依查詢語言回答，跨語言時精準轉述
        3. 保留原有引用標註規則
        4. 修復：doc_language 鍵缺失的相容處理
        """
        # 檢測查詢語言（zh/en）
        query_lang = detect_query_language(question)
        
        # 獲取文檔語言（增加容錯，避免KeyError）
        doc_lang = "zh"  # 默认中文
        if contexts and isinstance(contexts, list) and len(contexts) > 0:
            # 使用 get 方法取值，避免KeyError
            doc_lang = contexts[0].get("doc_language", "zh")  
            # 額外容錯：確保語言值正常
            if doc_lang not in ["zh", "en"]:
                doc_lang = "zh"

        # 構建上下文字符串
        context_str = ""
        for i, ctx in enumerate(contexts):
            # 所有取值都用 get 方法，避免KeyError
            doc_title = ctx.get('doc_title', '未找到 / Not Found')
            heading = ctx.get('heading', '未分類章節 / Unclassified Chapter')
            page = ctx.get('page', 0)
            author = ctx.get('author', '未找到 / Not Found')
            content = ctx.get('content', '')
            
            context_str += f"""
【參考{i+1}】
文件標題：{doc_title}
章節：{heading}
頁碼：第 {page} 頁
作者：{author}
內容：{content}
"""
        # 跨語言專屬Prompt引導
        if query_lang == "zh" and doc_lang == "en":
            # 英文文檔+中文提問：要求AI用中文精準轉述英文内容
            system_prompt = """你是專業的跨語言PDF文檔問答助手，嚴格遵守以下規則：
1. 參考內容為英文，用戶提問為中文，**必須用純中文精準轉述參考內容的語義**，不得保留英文原文（專業用語可保留並備註英文原文）；
2. 優先使用配對使用者問題中「頁碼/章節」的參考內容回答，精確對應；
3. 僅使用參考內容回答，禁止編造任何訊息，答案簡潔準確；
4. 回答結束後，必須在末尾標註引用來源（格式：「引用：章節名 - 第X頁」）；
5. 若參考內容無相關信息，直接回答："未找到相關內容"。
【參考內容】：
{context_str}""".format(context_str=context_str)
        elif query_lang == "en" and doc_lang == "zh":
            # 中文文檔+英文提問：要求AI用純英文精準轉述中文內容
            system_prompt = """You are a professional cross-lingual PDF Q&A assistant, strictly follow these rules:
1. The reference content is in Chinese and the user's question is in English, **answer in pure English and accurately paraphrase the semantic of the Chinese reference content** (professional and technical terms may be retained, with the original English translation noted in the margin.);
2. Prioritize answering with reference content matching the "page/chapter" in the user's question;
3. Answer only based on the reference content, do not fabricate any information, and keep the answer concise and accurate;
4. Mark the citation source at the end of the answer (format: "Citation: Chapter Name - Page X");
5. If there is no relevant information in the reference content, answer directly: "No relevant content found".
【Reference Content】：
{context_str}""".format(context_str=context_str)
        else:
            # 單語言场景（中/英）
            system_prompt = """你是專業的PDF文件問答助手，嚴守以下規則：
1. 優先使用配對使用者問題中「頁碼/章節」的參考內容回答，精準對應；
2. 僅使用參考內容回答，禁止編造任何訊息，回答簡潔準確；
3. 回答語言要和用戶提問語言相同；
4. 回答結束後，必須在末尾標註引用來源（格式：「引用：章節名 - 第X頁」）；
5. 若參考內容無相關訊息，直接回答："未找到相關內容"。
【參考內容】：
{context_str}""".format(context_str=context_str)

        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question}
        ]

    def stream_query(self, question, contexts):
        """
        1. 增加 contexts 空值/格式容錯
        2. 原有流式邏輯不變，跨語言由Prompt引導
        """
        # 容錯：contexts 為空/非列表時的處理
        if not contexts or not isinstance(contexts, list):
            # 双語提示
            return iter(["未找到相關內容 / No relevant content found"])
        
        try:
            messages = self.build_prompt(question, contexts)
            client = self._get_client()
            stream = client.chat.completions.create(
                model="deepseek-chat",
                messages=messages,
                temperature=0.2,  # 低temperature保證轉述精準性
                stream=True
            )
            return stream
        except Exception as e:
            error_msg = f"API呼叫失敗 / API Call Error：{str(e)}"
            print(error_msg)
            return iter([error_msg])