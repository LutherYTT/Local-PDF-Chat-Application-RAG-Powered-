from openai import OpenAI
from dotenv import load_dotenv
import os
import re

# 載入配置
load_dotenv()

class QuestionGenerator:
    def __init__(self):
        self.base_url = "https://api.deepseek.com"
    
    def _get_client(self):
        """動態取得 OpenAI 用戶端（每次都讀最新的 API Key）"""
        return OpenAI(
            api_key=os.getenv("DEEPSEEK_API_KEY"),
            base_url=self.base_url
        )

    def generate_virtual_questions(self, text_chunk, meta_info, num_questions=8):
        """
        基於文本塊生成虛擬問題（作为檢索入口）
        :param text_chunk: PDF文本塊内容
        :param meta_info: 元信息（作者/標題/章節/頁碼）
        :param num_questions: 生成虛擬問題數量
        :return: 虛擬問題列表
        """
        if len(text_chunk.strip()) < 20:
            return []
        # 識別文本語言：含中文則生成中文問題，否則生成英文問題
        has_chinese = bool(re.search(r'[\u4e00-\u9fff]', text_chunk))
        if has_chinese:
            prompt = f"""你是專業的文檔問答優化助手，請基於以下文字內容，產生{num_questions}個符合用戶提問習慣的中文虛擬問題：
    1. 問題需涵蓋文字核心訊息；2. 簡潔易懂，符合日常提問邏輯；3. 每個問題獨立成行，僅傳回問題清單。
    【文本内容】：{text_chunk}
    【元信息】：文檔標題：{meta_info.get('doc_title', '未知')}，章節：{meta_info.get('heading', '未知')}，頁碼：第{meta_info.get('page', '未知')}頁"""
        else:
            prompt = f"""You are a professional document Q&A assistant. Generate {num_questions} natural English virtual questions based on the text below:
    1. Cover the core information of the text; 2. Simple and in line with daily query habits; 3. Return only the question list, one question per line.
    【Text Content】：{text_chunk}
    【Meta Info】：Title: {meta_info.get('doc_title', 'Unknown')}, Chapter: {meta_info.get('heading', 'Unknown')}, Page: {meta_info.get('page', 'Unknown')}"""
        try:
            client = self._get_client()
            response = client.chat.completions.create(
                model="deepseek-chat",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=2000
            )
            questions = response.choices[0].message.content.strip().split("\n")
            valid_questions = [q.strip() for q in questions if q.strip() and len(q.strip()) > 5]
            return valid_questions[:num_questions]
        except Exception as e:
            print(f"產生虛擬問題失敗 / Generate Virtual Questions Error: {e}")
            return []