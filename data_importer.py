import pandas as pd
import json
import csv
from typing import List, Dict, Any
from pathlib import Path

from data_upserter import PineconeDataUpserter

class FileDataImporter:
    def __init__(self, index_name="rag-documents"):
        self.upserter = PineconeDataUpserter(index_name)
    
    def import_from_csv(self, csv_file: str, encoding='utf-8'):
        """Import ข้อมูลจาก CSV file"""
        print(f"Importing from CSV: {csv_file}")
        
        df = pd.read_csv(csv_file, encoding=encoding)
        documents = []
        
        for idx, row in df.iterrows():
            doc = {
                'id': f"csv_{idx}",
                'title': row.get('title', f"Document {idx}"),
                'content': row.get('content', ''),
                'source_url': row.get('source_url', ''),
                'type': row.get('type', 'csv_import'),
                'created_at': str(pd.Timestamp.now())
            }
            documents.append(doc)
        
        self.upserter.upsert_documents(documents)
        return len(documents)
    
    def import_from_json(self, json_file: str, encoding='utf-8'):
        """Import ข้อมูลจาก JSON file"""
        print(f"Importing from JSON: {json_file}")
        
        with open(json_file, 'r', encoding=encoding) as f:
            data = json.load(f)
        
        # Handle different JSON structures
        if isinstance(data, list):
            documents = data
        elif isinstance(data, dict) and 'documents' in data:
            documents = data['documents']
        else:
            documents = [data]
        
        # Ensure required fields
        for i, doc in enumerate(documents):
            if 'id' not in doc:
                doc['id'] = f"json_{i}"
            if 'created_at' not in doc:
                doc['created_at'] = str(pd.Timestamp.now())
        
        self.upserter.upsert_documents(documents)
        return len(documents)
    
    def import_from_txt(self, txt_file: str, title: str = None, encoding='utf-8'):
        """Import ข้อมูลจาก text file"""
        print(f"Importing from TXT: {txt_file}")
        
        with open(txt_file, 'r', encoding=encoding) as f:
            content = f.read()
        
        doc = {
            'id': f"txt_{Path(txt_file).stem}",
            'title': title or Path(txt_file).stem,
            'content': content,
            'source_url': f"file://{txt_file}",
            'type': 'text_file',
            'created_at': str(pd.Timestamp.now())
        }
        
        self.upserter.upsert_documents([doc])
        return 1
    
    def import_from_folder(self, folder_path: str, file_types=['.txt', '.md']):
        """Import ข้อมูลจาก folder ทั้งหมด"""
        print(f"Importing from folder: {folder_path}")
        
        folder = Path(folder_path)
        documents = []
        
        for file_path in folder.rglob('*'):
            if file_path.is_file() and file_path.suffix.lower() in file_types:
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    doc = {
                        'id': f"folder_{file_path.stem}",
                        'title': file_path.stem.replace('_', ' ').replace('-', ' '),
                        'content': content,
                        'source_url': f"file://{file_path}",
                        'type': 'folder_import',
                        'created_at': str(pd.Timestamp.now())
                    }
                    documents.append(doc)
                
                except Exception as e:
                    print(f"Error reading {file_path}: {e}")
        
        if documents:
            self.upserter.upsert_documents(documents)
        
        return len(documents)

# ตัวอย่างการใช้งาน
if __name__ == "__main__":
    importer = FileDataImporter()
    
    # สร้างไฟล์ตัวอย่าง
    sample_data = [
        {
            'title': 'Python Basics',
            'content': 'Python เป็นภาษาโปรแกรมมิ่งที่เรียนรู้ง่าย...',
            'source_url': 'https://example.com/python',
            'type': 'tutorial'
        },
        {
            'title': 'Machine Learning',
            'content': 'Machine Learning เป็นสาขาหนึ่งของ AI...',
            'source_url': 'https://example.com/ml',
            'type': 'tutorial'
        }
    ]
    
    # บันทึกเป็น JSON
    with open('sample_data.json', 'w', encoding='utf-8') as f:
        json.dump(sample_data, f, ensure_ascii=False, indent=2)
    
    # Import จาก JSON
    count = importer.import_from_json('sample_data.json')
    print(f"Imported {count} documents")