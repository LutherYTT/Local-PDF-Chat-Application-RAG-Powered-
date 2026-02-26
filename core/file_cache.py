import os
import hashlib
import json
import pickle
from pathlib import Path

class FileCacheManager:
    """文件緩存管理器：負責計算文件Hash、保存/加載處理結果"""
    def __init__(self, cache_dir="./data/cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.index_file = self.cache_dir / "cache_index.json"
        self._load_index()

    def _load_index(self):
        """加載緩存索引"""
        if self.index_file.exists():
            with open(self.index_file, "r", encoding="utf-8") as f:
                self.cache_index = json.load(f)
        else:
            self.cache_index = {}

    def _save_index(self):
        """保存緩存索引"""
        with open(self.index_file, "w", encoding="utf-8") as f:
            json.dump(self.cache_index, f, ensure_ascii=False, indent=2)

    def calculate_file_hash(self, file_path):
        """計算文件的MD5 Hash值（用于識別相同文件）"""
        hash_md5 = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()

    def is_cached(self, file_hash):
        """檢查文件是否已緩存"""
        return file_hash in self.cache_index

    def save_cache(self, file_hash, file_name, chunks, vector_store_path):
        """保存處理結果到緩存"""
        # 保存分块數據
        chunks_file = self.cache_dir / f"{file_hash}_chunks.pkl"
        with open(chunks_file, "wb") as f:
            pickle.dump(chunks, f)
        
        # 更新緩存索引
        self.cache_index[file_hash] = {
            "file_name": file_name,
            "chunks_path": str(chunks_file),
            "vector_store_path": vector_store_path,
            "timestamp": os.path.getmtime(chunks_file)
        }
        self._save_index()

    def load_cache(self, file_hash):
        """从緩存加載處理結果"""
        if not self.is_cached(file_hash):
            return None, None
        
        cache_info = self.cache_index[file_hash]
        # 加載分块數據
        with open(cache_info["chunks_path"], "rb") as f:
            chunks = pickle.load(f)
        
        return chunks, cache_info["vector_store_path"]