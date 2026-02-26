import os
import sys
from ui.app import PDFChatApp

def main():
    # 確保data目錄存在
    data_dirs = [
        "./data",
        "./data/faiss_index",
        "./data/uploads"
    ]
    for dir_path in data_dirs:
        try:
            # 使用os.makedirs並設定exist_ok=True，避免重複創報錯
            os.makedirs(dir_path, exist_ok=True)
            print(f"目錄已創建/存在: {dir_path}")
        except Exception as e:
            print(f"創建目錄失敗 {dir_path}: {e}")
            sys.exit(1)
    
    # 啟動應用
    app = PDFChatApp()
    app.mainloop()

if __name__ == "__main__":
    main()