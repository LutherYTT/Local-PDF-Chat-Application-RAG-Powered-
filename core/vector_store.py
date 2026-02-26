from langchain_community.vectorstores import FAISS
from langchain_core.embeddings import Embeddings
import os
import hashlib
import json
from sentence_transformers import SentenceTransformer, CrossEncoder
import numpy as np
import re
from collections import defaultdict
from core.utils import is_special_query, parse_page_query, parse_chapter_query

class CustomEmbeddings(Embeddings):
    def __init__(self, model_name="paraphrase-multilingual-MiniLM-L12-v2"):
        super().__init__()
        # 多語言模型：適配中英混合文件（作者名+中文查詢）
        self.model = SentenceTransformer(model_name, device="cpu")
        self.model_name = model_name
    
    def embed_documents(self, texts):
        embeddings = self.model.encode(texts, normalize_embeddings=True)
        return embeddings.tolist()
    
    def embed_query(self, text):
        embedding = self.model.encode(text, normalize_embeddings=True)
        return embedding.tolist()
    
    def __call__(self, text):
        return self.embed_query(text)

def get_string_hash(s: str) -> str:
    """生成字串的MD5 hash值"""
    return hashlib.md5(s.encode('utf-8')).hexdigest()

class HierarchicalVectorStore:
    def __init__(self, persist_path="./data/faiss_index"):
        self.persist_path = persist_path
        self.embeddings = CustomEmbeddings()
        # 重排模型
        self.reranker = CrossEncoder("cross-encoder/ms-marco-TinyBERT-L-2-v2", device="cpu")
        self.hierarchical_index = defaultdict(lambda: defaultdict(dict))
        self.global_db = None
        self.hash_mapping = {}
        self.mapping_file = os.path.join(persist_path, "hash_mapping.json")
        self._load_hash_mapping()
        self.processed_chunks = []
        self.adjacent_num = 2  # 上下文窗口
        # 平衡加權參數：避免單一維度（頁碼/章節）過度壓制語意相關內容
        self.page_chapter_weight = 6.0  
        self.semantic_base_weight = 2.0  # 語意匹配基礎分

    def load_index_from_path(self, index_path):
        """從指定路徑載入已儲存的向量索引"""
        try:
            # 載入全域索引
            global_index_path = os.path.join(index_path, "global_index")
            if os.path.exists(global_index_path):
                self.global_db = FAISS.load_local(
                    global_index_path, 
                    self.embeddings, 
                    allow_dangerous_deserialization=True
                )
            
            # 載入層次化索引
            mapping_file = os.path.join(index_path, "hierarchical_mapping.json")
            if os.path.exists(mapping_file):
                with open(mapping_file, "r", encoding="utf-8") as f:
                    index_mapping = json.load(f)
                
                for doc_hash in index_mapping:
                    self.hierarchical_index[doc_hash] = {}
                    for heading_hash in index_mapping[doc_hash]:
                        index_dir = os.path.join(index_path, index_mapping[doc_hash][heading_hash])
                        if os.path.exists(index_dir):
                            db = FAISS.load_local(
                                index_dir, 
                                self.embeddings, 
                                allow_dangerous_deserialization=True
                            )
                            # 反向查找原始標題和章節名
                            doc_title = self.hash_mapping.get(doc_hash, "未知文檔")
                            heading = self.hash_mapping.get(heading_hash, "未知章節")
                            self.hierarchical_index[doc_title][heading] = db
            
            return True
        except Exception as e:
            print(f"載入向量索引失敗: {e}")
            return False
    
    def _load_hash_mapping(self):
        if os.path.exists(self.mapping_file):
            try:
                with open(self.mapping_file, "r", encoding="utf-8") as f:
                    self.hash_mapping = json.load(f)
                # 校驗hash映射表格式
                if not isinstance(self.hash_mapping, dict):
                    raise ValueError("哈希映射表不是字典格式")
            except Exception as e:
                print(f"加载哈希映射表失敗: {e}")
                self.hash_mapping = {}
                # 自動重建空的哈希映射表
                self._save_hash_mapping()
        else:
            # 文件不存在時自動創建空文件
            self.hash_mapping = {}
            self._save_hash_mapping()
            print(f"⚠️ Hash映射表不存在，已自動創建: {self.mapping_file}")
    
    def _save_hash_mapping(self):
        """保存Hash映射表"""
        try:
            os.makedirs(self.persist_path, exist_ok=True)
            with open(self.mapping_file, "w", encoding="utf-8") as f:
                json.dump(self.hash_mapping, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"保存Hash映射表失敗: {e}")
    
    def _get_safe_path(self, original_name: str) -> str:
        """生成安全的Hash路徑名（避免特殊字符）"""
        if not original_name:
            original_name = "empty_name"
        # 查找已存在的哈希值
        for hash_val, name in self.hash_mapping.items():
            if name == original_name:
                return hash_val
        # 生成新哈希
        hash_val = get_string_hash(original_name)
        self.hash_mapping[hash_val] = original_name
        self._save_hash_mapping()
        return hash_val

    def set_processed_chunks(self, chunks):
        """設置已處理的文本塊（用於上下文擴展）"""
        self.processed_chunks = chunks
    
    def get_adjacent_chunks(self, target_chunk):
        """獲取目標塊的相鄰塊（擴大上下文窗口）"""
        if not self.processed_chunks:
            return [target_chunk]
        target_idx = target_chunk.get("chunk_idx", -1)
        total_chunks = target_chunk.get("total_chunks", 0)
        if target_idx == -1 or total_chunks == 0:
            return [target_chunk]
        start_idx = max(0, target_idx - self.adjacent_num)
        end_idx = min(total_chunks - 1, target_idx + self.adjacent_num)
        adjacent_chunks = [self.processed_chunks[idx] for idx in range(start_idx, end_idx + 1)]
        return adjacent_chunks

    def build_hierarchical_index(self, chunks_with_meta, save_path=None):
        """
        建立層次化向量索引（按文檔標題→章節分層）
        :param chunks_with_meta: 帶元資料的文字塊列表
        :param save_path: 索引保存路徑（預設使用初始化的persist_path）
        :return: 建置成功回傳True，失敗回傳False
        """
        if save_path is None:
            save_path = self.persist_path
        try:
            os.makedirs(self.persist_path, exist_ok=True)
            hierarchical_groups = defaultdict(lambda: defaultdict(list))
            
            # 按文檔標題和章節分組
            for chunk in chunks_with_meta:
                doc_title = chunk["doc_title"]
                heading = chunk["heading"]
                hierarchical_groups[doc_title][heading].append(chunk)
            
            # 過濾無效塊（空/過短內容）
            def filter_valid_chunks(chunks):
                return [c for c in chunks if len(c["content"].strip()) > 10]
            
            # 構建分層索引
            for doc_title in hierarchical_groups:
                doc_hash = self._get_safe_path(doc_title)
                for heading in hierarchical_groups[doc_title]:
                    heading_hash = self._get_safe_path(heading)
                    chunks = filter_valid_chunks(hierarchical_groups[doc_title][heading])
                    if not chunks:
                        continue
                    # 提取文本和元數據
                    texts = [c["content"] for c in chunks]
                    metadatas = [
                        {
                            "page": c["page"],
                            "heading": c["heading"],
                            "author": c["author"],
                            "doc_title": c["doc_title"],
                            "virtual_questions": c["virtual_questions"],
                            "chunk_idx": c["chunk_idx"],
                            "total_chunks": c["total_chunks"],
                            "content_hash": get_string_hash(c["content"])
                        }
                        for c in chunks
                    ]
                    # 構建FAISS索引
                    db = FAISS.from_texts(texts, self.embeddings, metadatas=metadatas)
                    self.hierarchical_index[doc_title][heading] = db
                    # 保存索引到本地
                    save_dir = os.path.join(self.persist_path, f"{doc_hash}_{heading_hash}_index")
                    os.makedirs(save_dir, exist_ok=True)
                    db.save_local(save_dir)
            
            # 構建全局索引（包含所有有效塊）
            all_chunks = filter_valid_chunks(chunks_with_meta)
            all_texts = [c["content"] for c in all_chunks]
            all_metadatas = [
                {
                    "page": c["page"],
                    "heading": c["heading"],
                    "author": c["author"],
                    "doc_title": c["doc_title"],
                    "virtual_questions": c["virtual_questions"],
                    "chunk_idx": c["chunk_idx"],
                    "total_chunks": c["total_chunks"],
                    "content_hash": get_string_hash(c["content"])
                }
                for c in all_chunks
            ]
            self.global_db = FAISS.from_texts(all_texts, self.embeddings, metadatas=all_metadatas)
            global_save_path = os.path.join(self.persist_path, "global_index")
            os.makedirs(global_save_path, exist_ok=True)
            self.global_db.save_local(global_save_path)
            
            # 保存分層索引映射關係
            index_mapping = {}
            for doc_title in self.hierarchical_index:
                doc_hash = self._get_safe_path(doc_title)
                index_mapping[doc_hash] = {}
                for heading in self.hierarchical_index[doc_title]:
                    heading_hash = self._get_safe_path(heading)
                    index_mapping[doc_hash][heading_hash] = f"{doc_hash}_{heading_hash}_index"
            
            with open(os.path.join(self.persist_path, "hierarchical_mapping.json"), "w", encoding="utf-8") as f:
                json.dump(index_mapping, f, ensure_ascii=False, indent=2)
            
            return True
        except Exception as e:
            print(f"構建層次化索引失敗: {e}")
            import traceback
            traceback.print_exc()
            return False

    def enhanced_keyword_match(self, query, docs, top_k=5):
        """
        增強關鍵字配對（融合語意、頁碼、章節、作者、虛擬問題等維度）
        :param query: 查詢語句
        :param docs: 初始語意匹配的文檔列表
        :param top_k: 最終傳回的文件數量
        :return: (doc_scores: 文件分數映射, final_matched: 篩選後的文檔清單)
        """
        matched_docs = []
        query_lower = query.lower()
        # 解析頁碼/章節
        page_nums = parse_page_query(query)
        chapter_name, _ = parse_chapter_query(query)
        chapter_name_lower = chapter_name.lower() if chapter_name else ""

        # 第一步：計算每個文檔的綜合得分（語義+加權）
        doc_scores = {}  # 得分映射：content_hash -> (score, doc)
        for doc in docs:
            score = self.semantic_base_weight  # 語義匹配基礎分（保底）
            content = doc.page_content.lower()
            meta = doc.metadata
            doc_page = meta.get("page", 0)
            doc_heading_lower = meta.get("heading", "").lower()
            content_hash = meta.get("content_hash", "")

            # 1. 頁碼/章節加權（可控優先级）
            if page_nums and doc_page in page_nums:
                score += self.page_chapter_weight
            if chapter_name and chapter_name_lower in doc_heading_lower:
                score += self.page_chapter_weight

            # 2. 元數據匹配（作者/標題）
            if meta.get("author") and meta["author"].strip():
                # 只要元數據有作者，無論是否匹配query，都加基礎分
                score += 1.5
                if meta["author"].lower() in query_lower:
                    score += 2.0
            if meta.get("doc_title") and meta["doc_title"].lower() in query_lower:
                score += 1.5

            # 3. 虛擬問題匹配
            virtual_questions = meta.get("virtual_questions", [])
            for q in virtual_questions:
                if q.lower() in query_lower or query_lower in q.lower():
                    score += 1.8

            # 4. 内容關鍵詞匹配（兼容中英文）
            # 提取query中的核心詞（長度≥2）
            query_keywords = [kw for kw in re.findall(r'\w+', query_lower) if len(kw) >= 2]
            for kw in query_keywords:
                if kw in content:
                    score += 0.5
            # 英文名字匹配（作者名通常是「名+姓」）
            english_names = re.findall(r'[A-Z][a-z]+\s+[A-Z][a-z]+', content)
            for name in english_names:
                if name.lower() in query_lower:
                    score += 3.0

            # 保存得分（按内容Hash去重，保留最高分）
            if content_hash in doc_scores:
                if score > doc_scores[content_hash][0]:
                    doc_scores[content_hash] = (score, doc)
            else:
                doc_scores[content_hash] = (score, doc)

        # 第二步：排序+截斷
        sorted_docs = sorted(doc_scores.values(), key=lambda x: x[0], reverse=True)
        final_matched = [doc for (score, doc) in sorted_docs[:top_k]]

        # 第三步：兜底召回（保證數量）
        if len(final_matched) < top_k:
            # 補充語義相關但未加權的塊
            for doc in docs:
                content_hash = doc.metadata.get("content_hash", "")
                if content_hash not in doc_scores and doc not in final_matched:
                    final_matched.append(doc)
                    if len(final_matched) == top_k:
                        break

        return doc_scores, final_matched

    def hierarchical_search(self, query, top_k=5):
        """
        層次化擷取（優先標題/章節→全域擷取→加權重排→上下文擴展）
        :param query: 查詢語句
        :param top_k: 最終傳回的文字塊數量
        :return: 帶有上下文的文字塊列表
        """
        if not self.global_db and not self.hierarchical_index and not self.processed_chunks:
            return []
        
        query_lower = query.lower()
        matched_docs = []
        matched_doc_titles = []
        expand_factor = 5  # 擴容因子：提升召回率

        # 1. 層次化語義檢索（優先標題/章節）
        for doc_title in self.hierarchical_index:
            if doc_title.lower() in query_lower:
                matched_doc_titles.append(doc_title)
        
        if matched_doc_titles:
            for doc_title in matched_doc_titles:
                matched_headings = []
                for heading in self.hierarchical_index[doc_title]:
                    if heading.lower() in query_lower:
                        matched_headings.append(heading)
                
                if matched_headings:
                    # 匹配到章節：針對性檢索
                    for heading in matched_headings:
                        db = self.hierarchical_index[doc_title][heading]
                        docs = db.similarity_search(query, k=top_k*expand_factor)
                        matched_docs.extend(docs)
                else:
                    # 無匹配章節：均分召回數量到所有章節
                    heading_count = len(self.hierarchical_index[doc_title])
                    k_per_heading = max(1, top_k*expand_factor // heading_count)
                    for heading, db in self.hierarchical_index[doc_title].items():
                        docs = db.similarity_search(query, k=k_per_heading)
                        matched_docs.extend(docs)
        else:
            # 無匹配標題：全局檢索
            if self.global_db:
                matched_docs = self.global_db.similarity_search(query, k=top_k*expand_factor)

        # 2. 加權關鍵詞匹配
        doc_scores, matched_docs = self.enhanced_keyword_match(query, matched_docs, top_k=top_k*2)

        # 3. 去重（按内容Hash）
        seen_hash = set()
        unique_docs = []
        for doc in matched_docs:
            content_hash = doc.metadata.get("content_hash", "")
            if content_hash not in seen_hash:
                seen_hash.add(content_hash)
                unique_docs.append(doc)

        # 4. 重排（reranker）：融合加權得分和reranker得分
        final_ranked = []
        if len(unique_docs) > 0:
            pairs = [[query, doc.page_content] for doc in unique_docs]
            scores = self.reranker.predict(pairs)
            # 融合加權得分和reranker得分（避免reranker完全覆蓋加權）
            fused_scores = []
            for i, (score, doc) in enumerate(zip(scores, unique_docs)):
                # 加權得分（標準化） + reranker得分（標準化）
                content_hash = doc.metadata.get("content_hash", "")
                weighted_score = doc_scores.get(content_hash, (0,))[0]
                norm_weighted = weighted_score / (self.page_chapter_weight * 2)  # 標準化到0-1
                norm_rerank = (score - min(scores)) / (max(scores) - min(scores) + 1e-6)  # 標準化到0-1
                fused = 0.6 * norm_weighted + 0.4 * norm_rerank  # 加權得分占60%，reranker占40%
                fused_scores.append((fused, doc))
            # 按融合得分排序
            fused_scores = sorted(fused_scores, key=lambda x: x[0], reverse=True)
            final_ranked = [item[1] for item in fused_scores[:top_k]]
        else:
            final_ranked = []

        # 5. 上下文增强（擴大相鄰塊）
        target_chunk_list = []
        for doc in final_ranked:
            target_chunk = {
                "content": doc.page_content,
                "page": doc.metadata["page"],
                "heading": doc.metadata["heading"],
                "author": doc.metadata["author"],
                "doc_title": doc.metadata["doc_title"],
                "virtual_questions": doc.metadata["virtual_questions"],
                "chunk_idx": doc.metadata["chunk_idx"],
                "total_chunks": doc.metadata["total_chunks"]
            }
            target_chunk_list.append(target_chunk)
        
        # 去重+合併相鄰塊
        all_context_chunks = []
        seen_idx = set()
        for chunk in target_chunk_list:
            adjacent_chunks = self.get_adjacent_chunks(chunk)
            for ac in adjacent_chunks:
                if ac["chunk_idx"] not in seen_idx:
                    seen_idx.add(ac["chunk_idx"])
                    all_context_chunks.append(ac)

        return all_context_chunks

# 兼容原有接口
class VectorStoreManager(HierarchicalVectorStore):
    def __init__(self, persist_path="./data/faiss_index"):
        super().__init__(persist_path)
    
    def create_index(self, chunks_with_meta):
        """兼容原有創建索引接口"""
        return self.build_hierarchical_index(chunks_with_meta)
    
    def search_with_rerank(self, query, top_k=15, return_k=5):
        """兼容原有檢索接口"""
        return self.hierarchical_search(query, top_k=return_k)


if __name__ == "__main__":
    # 測試數據
    test_chunks = [
        {
            "content": "張三，男，1980年生，研究方向為人工智慧",
            "page": 5,
            "heading": "第1章 作者介紹",
            "author": "張三",
            "doc_title": "人工智慧研究手冊",
            "virtual_questions": ["張三的研究方向是什麼？", "張三的出生年份？"],
            "chunk_idx": 0,
            "total_chunks": 3
        },
        {
            "content": "李四，女，1985年生，研究方向為自然語言處理",
            "page": 6,
            "heading": "第1章 作者介紹",
            "author": "李四",
            "doc_title": "人工智慧研究手冊",
            "virtual_questions": ["李四的研究方向是什麼？", "李四的出生年份？"],
            "chunk_idx": 1,
            "total_chunks": 3
        },
        {
            "content": "王五，男，1990年生，研究方向為計算機視覺",
            "page": 7,
            "heading": "第1章 作者介紹",
            "author": "王五",
            "doc_title": "人工智慧研究手冊",
            "virtual_questions": ["王五的研究方向是什麼？", "王五的出生年份？"],
            "chunk_idx": 2,
            "total_chunks": 3
        }
    ]
    
    # 初始化向量存儲
    vs_manager = VectorStoreManager(persist_path="./test_faiss_index")
    # 構建索引
    vs_manager.create_index(test_chunks)
    vs_manager.set_processed_chunks(test_chunks)
    # 檢索測試
    result = vs_manager.search_with_rerank("張三 研究方向", return_k=1)
    print("檢索结果：")
    for chunk in result:
        print(f"内容：{chunk['content']}")
        print(f"頁碼：{chunk['page']}")
        print(f"作者：{chunk['author']}")
        print("-" * 50)