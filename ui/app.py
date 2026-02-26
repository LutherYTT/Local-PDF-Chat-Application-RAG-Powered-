import customtkinter as ctk
from tkinter import filedialog, END, PhotoImage
import threading
import os
import time
import random
import re
from datetime import datetime
from functools import wraps
# 新增：用于讀寫.env文件的库
from dotenv import load_dotenv, set_key

from core.pdf_processor import PDFProcessor
from core.vector_store import VectorStoreManager
from core.rag_chain import RAGEngine
from core.file_cache import FileCacheManager

# 全域樣式配置
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

# ===================== 問題序號清洗工具函數 =====================
def clean_question_serial_number(question: str) -> str:
    if not question:
        return question
    serial_pattern = r'^\s*[(【[]?(\d+|[一二三四五六七八九十百千]+)[.、)\]\s]\s*'
    cleaned_question = re.sub(serial_pattern, '', question.strip())
    return cleaned_question.strip()

# ===================== 節流裝飾器 =====================
def throttle(ms):
    def decorator(func):
        last_call = 0
        lock = threading.Lock()
        
        @wraps(func)
        def wrapper(*args, **kwargs):
            nonlocal last_call
            force = kwargs.pop('force', False)
            now = time.time() * 1000
            
            with lock:
                if force or (now - last_call >= ms):
                    last_call = now
                    return func(*args, **kwargs)
        return wrapper
    return decorator

class MessageBubble(ctk.CTkFrame):
    def __init__(self, master, sender, message, **kwargs):
        super().__init__(master,** kwargs)
        self.sender = sender
        self.message = message
        
        if sender == "你":
            self.bg_color = "#2563eb"
            self.text_color = "#ffffff"
            self.anchor = "e"
            self.btn_fg = "#3b82f6"
            self.btn_hover = "#60a5fa"
        elif sender == "AI":
            self.bg_color = "#374151"
            self.text_color = "#ffffff"
            self.anchor = "w"
            self.btn_fg = "#4b5563"
            self.btn_hover = "#6b7280"
        else:
            self.bg_color = "#1f2937"
            self.text_color = "#9ca3af"
            self.anchor = "center"
            self.btn_fg = "#27272a"
            self.btn_hover = "#3f3f46"
        
        self.configure(fg_color=self.bg_color, corner_radius=12)
        
        self.content_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.content_frame.pack(padx=15, pady=10, fill="both", expand=True)
        
        self.text_label = ctk.CTkLabel(
            self.content_frame,
            text=message,
            text_color=self.text_color,
            font=("Microsoft YaHei", 13),
            wraplength=650,
            justify="left"
        )
        self.text_label.pack(side="left", fill="both", expand=True)
        
        self.copy_btn = ctk.CTkButton(
            self.content_frame,
            text="📋",
            width=35,
            height=35,
            font=("Microsoft YaHei", 12),
            fg_color=self.btn_fg,
            hover_color=self.btn_hover,
            corner_radius=6,
            command=self._copy_message
        )
        self.copy_btn.pack(side="right", padx=(10, 0), pady=5)
        
        self.copy_hint = ctk.CTkLabel(
            self,
            text="✅ 已複製",
            text_color="#00ff9d",
            font=("Microsoft YaHei", 10)
        )

    def _copy_message(self):
        try:
            self.clipboard_clear()
            self.clipboard_append(self.message)
            self.update()
            self.copy_hint.place(relx=0.5, rely=1.1, anchor="n")
            self.after(2000, lambda: self.copy_hint.place_forget())
        except Exception as e:
            self.copy_hint.configure(text="❌ 複製失敗", text_color="#ff3300")
            self.copy_hint.place(relx=0.5, rely=1.1, anchor="n")
            self.after(2000, lambda: self.copy_hint.place_forget())

class PDFChatApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        # 設置Icon
        icon_path = os.path.join(os.path.dirname(__file__), "assets", "icon.ico")
        if os.path.exists(icon_path):
            try:
                self.iconbitmap(icon_path)
            except Exception as e:
                print(f"設置Icon失敗：{str(e)}")
                pass  # 忽略錯誤，繼續使用預設Icon
        else:
            # 嘗試載入 PNG 作為備選
            png_path = os.path.join(os.path.dirname(__file__), "assets", "icon.png")
            if os.path.exists(png_path):
                try:
                    img = PhotoImage(file=png_path)
                    self.iconphoto(True, img)
                except Exception as e:
                    print(f"設置Icon失敗：{str(e)}")
                    pass

        self.title("Local PDF Chat Application")
        self.geometry("1200x800")
        self.resizable(True, True)

        self.is_model_loading = False
        self.model_loaded = False
        self.is_processing = False
        self.current_file = None
        self.all_virtual_questions = []

        # ===================== 初始化.env文件和API Key =====================
        self.env_path = os.path.join(os.getcwd(), ".env")  # .env檔在項目根目錄
        self._init_env_file()  # 初始化.env文件（不存在則創建）
        load_dotenv(self.env_path)  # 加載環境變量
        self.deepseek_api_key = os.getenv("DEEPSEEK_API_KEY", "")  # 讀取已保存的API Key

        self.pdf_processor = PDFProcessor()
        self.rag_engine = RAGEngine()
        self.file_cache = FileCacheManager()
        self.vector_store = None

        self._setup_ui()
        self._preload_model_async()

    # ===================== 初始化.env文件 =====================
    def _init_env_file(self):
        """如果.env檔案不存在則創建空文件"""
        if not os.path.exists(self.env_path):
            try:
                with open(self.env_path, "w", encoding="utf-8") as f:
                    f.write("# DeepSeek API Configuration\n")
                    f.write("DEEPSEEK_API_KEY=\n")
            except Exception as e:
                print(f"創建.env文件失敗：{str(e)}")

    def _setup_ui(self):
        self.sidebar = ctk.CTkFrame(self, width=300, corner_radius=0)
        self.sidebar.pack(side="left", fill="y", padx=0, pady=0)
        self.sidebar.pack_propagate(False)
        
        # 側邊欄標題
        ctk.CTkLabel(
            self.sidebar, 
            text="本地PDF知識庫", 
            font=("Microsoft YaHei", 22, "bold")
        ).pack(pady=(30, 20))
        
        # 上傳按鈕
        self.upload_btn = ctk.CTkButton(
            self.sidebar,
            text="📂 上傳PDF文檔",
            height=55,
            font=("Microsoft YaHei", 15),
            command=self._upload_pdf,
            state="disabled"
        )
        self.upload_btn.pack(pady=10, padx=20, fill="x")
        
        # 進度顯示區域
        self.progress_frame = ctk.CTkFrame(self.sidebar, fg_color="transparent")
        self.progress_frame.pack(pady=20, padx=20, fill="x")
        
        self.progress_label = ctk.CTkLabel(
            self.progress_frame,
            text="🔄 模型加載中，請稍候...",
            font=("Microsoft YaHei", 12),
            text_color="#ffcc00"
        )
        self.progress_label.pack(pady=(0, 10), anchor="w")
        
        self.progress_bar = ctk.CTkProgressBar(
            self.progress_frame,
            height=8,
            corner_radius=4
        )
        self.progress_bar.pack(fill="x")
        self.progress_bar.set(0)
        
        # 推薦問題區域
        self.recommend_frame = ctk.CTkFrame(self.sidebar, fg_color="transparent")
        self.recommend_frame.pack(pady=20, padx=20, fill="both", expand=True)
        
        self.recommend_title_frame = ctk.CTkFrame(self.recommend_frame, fg_color="transparent")
        self.recommend_title_frame.pack(pady=(0, 15), fill="x")
        self.recommend_title_frame.pack_propagate(False)
        self.recommend_title_frame.configure(height=30)
        
        self.recommend_title = ctk.CTkLabel(
            self.recommend_title_frame,
            text="💡 推薦問題",
            font=("Microsoft YaHei", 14, "bold"),
            text_color="#60a5fa"
        )
        self.recommend_title.pack(side="left", anchor="w")
        
        self.refresh_btn = ctk.CTkButton(
            self.recommend_title_frame,
            text="🔄",
            width=30,
            height=25,
            font=("Microsoft YaHei", 12),
            fg_color="#4b5563",
            hover_color="#6b7280",
            command=self._refresh_recommend_questions,
            state="disabled"
        )
        self.refresh_btn.pack(side="right", anchor="e")
        
        self.recommend_buttons_frame = ctk.CTkScrollableFrame(
            self.recommend_frame,
            fg_color="transparent",
            height=200
        )
        self.recommend_buttons_frame.pack(fill="both", expand=True)
        self.recommend_buttons = []
        self._update_recommend_buttons([])

        # ===================== 左側底部API Key配置區域 =====================
        self.api_key_frame = ctk.CTkFrame(self.sidebar, fg_color="#1f2937", corner_radius=8)
        self.api_key_frame.pack(side="bottom", fill="x", padx=20, pady=(10, 20))
        
        # API Key標題
        self.api_key_title = ctk.CTkLabel(
            self.api_key_frame,
            text="🔑 DeepSeek API Key",
            font=("Microsoft YaHei", 12, "bold"),
            text_color="#60a5fa"
        )
        self.api_key_title.pack(anchor="w", padx=12, pady=(10, 5))
        
        # API Key輸入框（密碼模式，隱藏輸入内容）
        self.api_key_entry = ctk.CTkEntry(
            self.api_key_frame,
            placeholder_text="sk-...",
            font=("Microsoft YaHei", 12),
            height=40,
            show="•"  # 隱藏輸入内容，保护隱私
        )
        self.api_key_entry.pack(fill="x", padx=12, pady=(0, 8))
        # 初始化時填充已保存的API Key
        if self.deepseek_api_key:
            self.api_key_entry.insert(0, self.deepseek_api_key)
        
        # 保存按鈕
        self.save_api_key_btn = ctk.CTkButton(
            self.api_key_frame,
            text="💾 保存API Key",
            height=35,
            font=("Microsoft YaHei", 12),
            fg_color="#10b981",
            hover_color="#059669",
            command=self._save_deepseek_api_key
        )
        self.save_api_key_btn.pack(fill="x", padx=12, pady=(0, 12))
        
        # ========== 右側聊天區 ==========
        self.chat_frame = ctk.CTkFrame(self, corner_radius=0, fg_color="#111827")
        self.chat_frame.pack(side="right", fill="both", expand=True)
        
        self.chat_scroll = ctk.CTkScrollableFrame(
            self.chat_frame,
            fg_color="transparent",
            corner_radius=0
        )
        self.chat_scroll.pack(padx=20, pady=20, fill="both", expand=True)
        
        # 輸入區域
        self.input_frame = ctk.CTkFrame(self.chat_frame, height=70, fg_color="#1f2937")
        self.input_frame.pack(padx=20, pady=(0, 20), fill="x")
        self.input_frame.pack_propagate(False)
        
        self.input_entry = ctk.CTkEntry(
            self.input_frame,
            placeholder_text="輸入你的問題（支援章節/頁碼檢索）...",
            font=("Microsoft YaHei", 14),
            height=45,
            state="disabled"
        )
        self.input_entry.pack(side="left", fill="x", expand=True, padx=(15, 10), pady=12)
        self.input_entry.bind("<Return>", lambda e: self._send_question())
        
        self.export_btn = ctk.CTkButton(
            self.input_frame,
            text="匯出對話",
            width=90,
            height=45,
            font=("Microsoft YaHei", 14),
            fg_color="#10b981",
            hover_color="#059669",
            command=self._export_chat_history,
            state="disabled"
        )
        self.export_btn.pack(side="right", padx=(0, 10), pady=12)
        
        self.send_btn = ctk.CTkButton(
            self.input_frame,
            text="發送",
            width=90,
            height=45,
            font=("Microsoft YaHei", 14),
            command=self._send_question,
            state="disabled"
        )
        self.send_btn.pack(side="right", padx=(0, 15), pady=12)
        
        self._insert_message("系統", "歡迎使用本地 PDF 聊天應用程式！模型加載完成後即可上傳文檔使用。")

    # ===================== 保存API Key到.env文件 =====================
    def _save_deepseek_api_key(self):
        """保存DeepSeek API Key到.env文件"""
        api_key = self.api_key_entry.get().strip()
        
        if not api_key:
            self._insert_message("系統", "⚠️ 請輸入有效的DeepSeek API Key")
            return
        
        try:
            # 寫入.env文件（自動處理文件不存在的情况）
            set_key(self.env_path, "DEEPSEEK_API_KEY", api_key)
            # 更新内存中的API Key
            self.deepseek_api_key = api_key
            # 刷新環境變量
            os.environ["DEEPSEEK_API_KEY"] = api_key
            
            # 給用户成功提示
            self._insert_message("系統", "✅ DeepSeek API Key已成功保存到.env文件！\n（以後啟動程式會自動載入）")
            
        except Exception as e:
            self._insert_message("系統", f"❌ 保存API Key失敗：{str(e)}")

    def _preload_model_async(self):
        def load_model_in_thread():
            self.is_model_loading = True
            try:
                self.after(0, lambda: self.progress_label.configure(text="🔄 正在加載嵌入模型與重排模型...", text_color="#ffcc00"))
                self.after(0, lambda: self.progress_bar.set(0.2))
                
                self.vector_store = VectorStoreManager()
                
                self.after(0, lambda: self.progress_bar.set(1.0))
                self._update_progress(0, 0, "✅ 模型加載完成，就緒", 1.0, "#00ff9d", force=True)
                
                self.model_loaded = True
                self.after(0, lambda: self.upload_btn.configure(state="normal"))
                self.after(0, lambda: self.input_entry.configure(state="normal"))
                self.after(0, lambda: self.send_btn.configure(state="normal"))
                self.after(0, lambda: self.export_btn.configure(state="normal"))
                
                # 提示用户是否已配置API Key
                if not self.deepseek_api_key:
                    self._insert_message("系統", "✅ 模型加載完成！\n⚠️ 請先在左側底部配置DeepSeek API Key後再提問。")
                else:
                    self._insert_message("系統", "✅ 模型加載完成！已檢測到已保存的API Key，可直接上傳文檔提問。")
            except Exception as e:
                self._update_progress(0, 0, f"❌ 模型加載失敗：{str(e)}", 0, "#ff3300", force=True)
                self._insert_message("系統", f"模型加載失敗：{str(e)}，請重啟程式。")
            finally:
                self.is_model_loading = False
        
        threading.Thread(target=load_model_in_thread, daemon=True).start()

    @throttle(100)
    def _update_progress(self, step, total_steps, step_name, progress_value, status_color="#00ff9d", force=False):
        self.after(0, lambda: self._do_update_progress(step, total_steps, step_name, progress_value, status_color))
    
    def _do_update_progress(self, step, total_steps, step_name, progress_value, status_color):
        self.progress_label.configure(
            text=f"({step}/{total_steps}) {step_name}" if total_steps > 0 else step_name,
            text_color=status_color
        )
        self.progress_bar.set(progress_value)

    @throttle(100)
    def _update_text_chunk_progress(self, progress, step_name):
        self.after(0, lambda: self._do_update_text_chunk_progress(progress, step_name))
    
    def _do_update_text_chunk_progress(self, progress, step_name):
        total_progress = 0.2 + (progress * 0.3)
        self.progress_label.configure(
            text=f"(2/5) 文本分塊：{step_name}",
            text_color="#ffcc00"
        )
        self.progress_bar.set(total_progress)

    def _insert_message(self, sender, message):
        self.after(0, lambda: self._do_insert_message(sender, message))
    
    def _do_insert_message(self, sender, message):
        bubble = MessageBubble(self.chat_scroll, sender, message)
        bubble.pack(pady=8, padx=10, anchor=bubble.anchor)
        self.chat_scroll.update_idletasks()
        self.chat_scroll._parent_canvas.yview_moveto(1.0)

    def _update_recommend_buttons(self, questions):
        for btn in self.recommend_buttons:
            btn.destroy()
        self.recommend_buttons.clear()
        
        if not questions:
            empty_label = ctk.CTkLabel(
                self.recommend_buttons_frame,
                text="暫無推薦問題\n（請先上傳PDF文檔）",
                font=("Microsoft YaHei", 11),
                text_color="#9ca3af",
                justify="center",
                wraplength=250
            )
            empty_label.pack(pady=10)
            self.recommend_buttons.append(empty_label)
            self.refresh_btn.configure(state="disabled")
            return
        
        self.refresh_btn.configure(state="normal")
        selected_questions = random.sample(questions, min(4, len(questions)))
        
        for q in selected_questions:
            btn_label = ctk.CTkLabel(
                self.recommend_buttons_frame,
                text=q,
                height=60,
                font=("Microsoft YaHei", 13),
                fg_color="#374151",
                text_color="#ffffff",
                corner_radius=6,
                wraplength=200,
                justify="left",
                padx=14,
                pady=5
            )
            btn_label.pack(pady=5, fill="x", padx=5)
            self.recommend_buttons.append(btn_label)
            btn_label.bind("<Button-1>", lambda e, q=q: self._quick_ask(q))

    def _refresh_recommend_questions(self):
        if not self.all_virtual_questions:
            return
        self._update_recommend_buttons(self.all_virtual_questions)

    def _quick_ask(self, question):
        if self.is_processing or not self.model_loaded:
            return
        self.input_entry.delete(0, END)
        self.input_entry.insert(0, question)
        self._send_question()

    def _upload_pdf(self):
        if self.is_processing or not self.model_loaded or self.is_model_loading:
            return
        
        file_path = filedialog.askopenfilename(
            filetypes=[("PDF文件", "*.pdf")],
            title="選擇PDF文檔"
        )
        if not file_path:
            return
        
        self.current_file = file_path
        self.is_processing = True
        self.upload_btn.configure(state="disabled")
        self.send_btn.configure(state="disabled")
        self.all_virtual_questions = []

        def reset_vector_index():
            self.vector_store.global_db = None
            self.vector_store.hierarchical_index.clear()
            self.vector_store.processed_chunks = []
        
        reset_vector_index()
        
        def process_pdf():
            total_steps = 5
            try:
                # 1. 檢查緩存
                self._update_progress(0, total_steps, "檢查文件緩存...", 0.05, "#ffcc00")
                file_hash = self.file_cache.calculate_file_hash(file_path)
                file_name = os.path.basename(file_path)
                
                if self.file_cache.is_cached(file_hash):
                    self._update_progress(1, total_steps, "緩存命中，正在加載...", 0.1, "#00ff9d")
                    time.sleep(0.3)
                    
                    chunks, vector_store_path = self.file_cache.load_cache(file_hash)
                    
                    if chunks:
                        # 清洗序号
                        all_questions = set()
                        for chunk in chunks:
                            virtual_qs = chunk.get("virtual_questions", []) or []
                            for q in virtual_qs:
                                q_stripped = q.strip() if q else ""
                                if not q_stripped or len(q_stripped) <= 5:
                                    continue
                                cleaned_q = clean_question_serial_number(q_stripped)
                                if cleaned_q and len(cleaned_q) > 5:
                                    all_questions.add(cleaned_q)
                        self.all_virtual_questions = list(all_questions)
                        self.after(0, lambda: self._update_recommend_buttons(self.all_virtual_questions))
                        
                        self.vector_store.set_processed_chunks(chunks)
                        
                        # 嘗試加載索引
                        index_loaded = False
                        if vector_store_path and os.path.exists(vector_store_path):
                            try:
                                load_success = self.vector_store.load_index_from_path(vector_store_path)
                                index_valid = (self.vector_store.global_db is not None) or (len(self.vector_store.hierarchical_index) > 0)
                                if load_success and index_valid:
                                    index_loaded = True
                                    self._insert_message("系統", f"✅ 索引加載成功！")
                            except Exception as e:
                                self._insert_message("系統", f"⚠️ 索引加載失敗，將重建索引：{str(e)}")
                        
                        if not index_loaded:
                            self._update_progress(3, total_steps, "重建向量索引中...", 0.6, "#ffcc00")
                            cache_index_path = os.path.join(self.file_cache.cache_dir, f"{file_hash}_index")
                            success = self.vector_store.build_hierarchical_index(chunks, save_path=cache_index_path)
                            if not success:
                                raise Exception("重建向量索引失敗")
                            
                            self.file_cache.save_cache(file_hash, file_name, chunks, cache_index_path)
                        
                        self._update_progress(5, total_steps, "緩存加載完成！", 1.0, "#00ff9d", force=True)
                        author = chunks[0].get("author", "未知") if chunks else "未知"
                        self._insert_message("系統", 
                            f"文檔「{file_name}」已從緩存加載！\n"
                            f"作者：{author}\n"
                            f"總塊數：{len(chunks)}\n"
                            f"文檔增強數：{len(self.all_virtual_questions)}"
                        )
                        return
                    else:
                        self._update_progress(1, total_steps, "緩存數據損壞，重新處理PDF...", 0.1, "#ffcc00")
                        time.sleep(0.3)
                
                # 緩存未命中：全新處理流程
                self._update_progress(1, total_steps, "PDF文件解析初始化", 0.1, "#ffcc00")
                time.sleep(0.2)
                
                self._update_progress(2, total_steps, "文本分塊初始化", 0.2, "#ffcc00")
                chunks = self.pdf_processor.load_pdf_with_pages(
                    file_path,
                    progress_callback=self._update_text_chunk_progress
                )
                if not chunks:
                    raise Exception("未提取到文本內容，請檢查PDF是否為掃描件")
                
                # 清洗序号
                all_questions = set()
                for chunk in chunks:
                    virtual_qs = chunk.get("virtual_questions", []) or []
                    for q in virtual_qs:
                        q_stripped = q.strip() if q else ""
                        if not q_stripped or len(q_stripped) <= 5:
                            continue
                        cleaned_q = clean_question_serial_number(q_stripped)
                        if cleaned_q and len(cleaned_q) > 5:
                            all_questions.add(cleaned_q)
                self.all_virtual_questions = list(all_questions)
                self.after(0, lambda: self._update_recommend_buttons(self.all_virtual_questions))
                
                self._update_progress(3, total_steps, "文檔增強（虛擬問題整合）", 0.6, "#ffcc00")
                time.sleep(0.2)
                
                self._update_progress(4, total_steps, "構建向量索引中...", 0.8, "#ffcc00")
                cache_index_path = os.path.join(self.file_cache.cache_dir, f"{file_hash}_index")
                
                success = self.vector_store.build_hierarchical_index(chunks, save_path=cache_index_path)
                if not success:
                    raise Exception("構建向量索引失敗")
                
                if self.vector_store.global_db is None:
                    raise Exception("構建索引後全局索引為空，請檢查文檔內容是否有效")
                
                self.vector_store.set_processed_chunks(chunks)
                
                self._update_progress(4, total_steps, "保存處理結果到緩存...", 0.9, "#ffcc00")
                self.file_cache.save_cache(file_hash, file_name, chunks, cache_index_path)
                
                self._update_progress(5, total_steps, "處理完成！", 1.0, "#00ff9d", force=True)
                author = chunks[0].get("author", "未知") if chunks else "未知"
                self._insert_message("系統", 
                    f"文檔「{file_name}」加載完成！\n"
                    f"作者：{author}\n"
                    f"總塊數：{len(chunks)}\n"
                    f"文檔增強數：{len(self.all_virtual_questions)}\n"
                    f"（已保存到緩存，下次上傳直接加載）"
                )
                
            except Exception as e:
                self._update_progress(0, total_steps, f"處理失敗：{str(e)}", 0, "#ff3300", force=True)
                self._insert_message("系統", f"處理失敗：{str(e)}")
            finally:
                self.is_processing = False
                self.after(100, lambda: self._update_progress(
                    0, 0, "✅ 就緒（等待上傳/提問）", 1.0, "#00ff9d", force=True
                ))
                self.after(0, lambda: self.upload_btn.configure(state="normal"))
                self.after(0, lambda: self.send_btn.configure(state="normal"))
        
        threading.Thread(target=process_pdf, daemon=True).start()

    def _send_question(self):
        question = self.input_entry.get().strip()
        if not question or self.is_processing or not self.model_loaded:
            return
        
        # 提問前檢查是否已配置API Key
        if not self.deepseek_api_key:
            self._insert_message("系統", "⚠️ 請先在左側底部配置並保存DeepSeek API Key後再提問！")
            return
        
        self.input_entry.delete(0, END)
        self._insert_message("你", question)
        self.is_processing = True
        self.send_btn.configure(state="disabled")
        self.upload_btn.configure(state="disabled")
        
        def stream_answer():
            total_steps = 5
            try:
                self._update_progress(1, total_steps, "問題向量化中...", 0.15, "#ffcc00")
                time.sleep(0.1)
                
                self._update_progress(2, total_steps, "向量檢索中...", 0.35, "#ffcc00")
                
                index_valid = (self.vector_store.global_db is not None) or (len(self.vector_store.hierarchical_index) > 0)
                if not index_valid:
                    if self.current_file:
                        file_hash = self.file_cache.calculate_file_hash(self.current_file)
                        cache_index_path = os.path.join(self.file_cache.cache_dir, f"{file_hash}_index")
                        if os.path.exists(cache_index_path):
                            load_success = self.vector_store.load_index_from_path(cache_index_path)
                            if not load_success:
                                raise Exception("向量索引未加載，重新加載也失敗，請重新上傳PDF文檔")
                            index_valid = (self.vector_store.global_db is not None) or (len(self.vector_store.hierarchical_index) > 0)
                            if not index_valid:
                                raise Exception("索引加載後仍無效，請重新上傳PDF文檔")
                        else:
                            raise Exception("向量索引文件不存在，請重新上傳PDF文檔")
                    else:
                        raise Exception("未上傳任何PDF文檔，請先上傳文檔後再提問")
                
                contexts = self.vector_store.search_with_rerank(question)
                if not contexts:
                    self._insert_message("AI", "未找到相關內容，請嘗試調整問題表述或重新上傳PDF")
                    self._update_progress(5, total_steps, "檢索完成", 1.0, "#ffcc00", force=True)
                    return
                
                self._update_progress(3, total_steps, "結果重排中...", 0.55, "#ffcc00")
                time.sleep(0.1)
                
                self._update_progress(4, total_steps, "上下文增強中...", 0.75, "#ffcc00")
                time.sleep(0.1)
                
                self._update_progress(5, total_steps, "AI生成回答中...", 0.95, "#ffcc00")
                stream = self.rag_engine.stream_query(question, contexts)
                
                self._insert_message("AI", "")
                full_answer = ""
                for chunk in stream:
                    if hasattr(chunk, 'choices') and chunk.choices[0].delta.content:
                        content = chunk.choices[0].delta.content
                        full_answer += content
                        self.after(0, lambda: self._update_last_ai_bubble(full_answer))
                    elif isinstance(chunk, str):
                        full_answer = chunk
                        self.after(0, lambda: self._update_last_ai_bubble(full_answer))
                
                self._update_progress(5, total_steps, "回答完成", 1.0, "#00ff9d", force=True)
                
            except Exception as e:
                self._update_progress(0, total_steps, f"問答失敗：{str(e)}", 0, "#ff3300", force=True)
                self._insert_message("AI", f"問答失敗：{str(e)}")
            finally:
                self.is_processing = False
                self.after(100, lambda: self._update_progress(
                    0, 0, "✅ 就緒（等待上傳/提問）", 1.0, "#00ff9d", force=True
                ))
                self.after(0, lambda: self.send_btn.configure(state="normal"))
                self.after(0, lambda: self.upload_btn.configure(state="normal"))
        
        threading.Thread(target=stream_answer, daemon=True).start()

    def _update_last_ai_bubble(self, new_content):
        children = self.chat_scroll.winfo_children()
        if children:
            last_bubble = children[-1]
            if isinstance(last_bubble, MessageBubble) and last_bubble.sender == "AI":
                last_bubble.message = new_content
                last_bubble.text_label.configure(text=new_content)
                self.chat_scroll.update_idletasks()
                self.chat_scroll._parent_canvas.yview_moveto(1.0)

    def _export_chat_history(self):
        children = self.chat_scroll.winfo_children()
        bubbles = [child for child in children if isinstance(child, MessageBubble)]
        
        if not bubbles:
            self._insert_message("系統", "📭 暫無對話記錄可匯出")
            return
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        default_filename = f"PDF聊天記錄_{timestamp}.txt"
        
        file_path = filedialog.asksaveasfilename(
            defaultextension=".txt",
            filetypes=[("文本文件", "*.txt"), ("Markdown文件", "*.md"), ("所有文件", "*.*")],
            initialfile=default_filename,
            title="匯出對話記錄"
        )
        if not file_path:
            return
        
        try:
            lines = []
            lines.append("=" * 50)
            lines.append(f"匯出時間：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            if self.current_file:
                lines.append(f"關聯文檔：{os.path.basename(self.current_file)}")
            else:
                lines.append("關聯文檔：無")
            lines.append("=" * 50)
            lines.append("")
            
            for bubble in bubbles:
                sender = bubble.sender
                message = bubble.message.strip()
                if message:
                    lines.append(f"【{sender}】{message}")
                    lines.append("")
            
            content = "\n".join(lines)
            
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(content)
            
            self._insert_message("系統", f"✅ 對話已成功匯出到：{os.path.basename(file_path)}")
        except Exception as e:
            self._insert_message("系統", f"❌ 匯出失敗：{str(e)}")

if __name__ == "__main__":
    app = PDFChatApp()
    app.mainloop()