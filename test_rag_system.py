import json
from pathlib import Path
from data_importer import FileDataImporter
from vector_search import VectorSearcher

def create_sample_documents():
    """สร้างเอกสารตัวอย่างสำหรับทดสอบ"""
    documents = [
        {
            "title": "Python Programming Fundamentals",
            "content": """
            Python เป็นภาษาโปรแกรมมิ่งระดับสูงที่ถูกพัฒนาโดย Guido van Rossum ในปี 1991. 
            มันเป็นภาษาที่อ่านง่ายและเขียนง่าย เหมาะสำหรับผู้เริ่มต้นและผู้เชี่ยวชาญ.
            Python รองรับการเขียนโปรแกรมแบบ object-oriented, procedural, และ functional.
            มี library มากมายที่ช่วยในการพัฒนาแอปพลิเคชันต่างๆ เช่น Django สำหรับ web development,
            NumPy และ Pandas สำหรับ data science, TensorFlow และ PyTorch สำหรับ machine learning.
            """,
            "source_url": "https://docs.python.org/",
            "type": "documentation"
        },
        {
            "title": "Machine Learning with Python",
            "content": """
            Machine Learning เป็นสาขาหนึ่งของ Artificial Intelligence ที่เน้นการสร้างอัลกอริทึม
            ที่สามารถเรียนรู้จากข้อมูลโดยไม่ต้องเขียนโปรแกรมแบบ explicit.
            ใน Python มี libraries สำคัญสำหรับ ML เช่น scikit-learn สำหรับ classical ML,
            TensorFlow และ Keras สำหรับ deep learning, และ Pandas สำหรับ data manipulation.
            ประเภทของ ML ได้แก่ Supervised Learning (มีข้อมูล label), Unsupervised Learning (ไม่มี label),
            และ Reinforcement Learning (เรียนรู้จาก reward/punishment).
            """,
            "source_url": "https://scikit-learn.org/",
            "type": "tutorial"
        },
        {
            "title": "Pinecone Vector Database และ RAG Architecture",
            "content": """
            Pinecone เป็น managed vector database ที่เหมาะสำหรับการสร้าง RAG systems.
            มันช่วยให้เราสามารถทำ semantic search ได้ โดยใช้ similarity measures เช่น cosine similarity.
            Pinecone v7.x รองรับ Serverless architecture ที่จ่ายตามการใช้งานจริง.
            RAG (Retrieval-Augmented Generation) เป็น architecture ที่รวม information retrieval
            กับ text generation เพื่อให้ AI สามารถตอบคำถามด้วยข้อมูลที่ accurate และ up-to-date.
            ขั้นตอนของ RAG ประกอบด้วย: 1) Indexing (แปลงเอกสารเป็น embeddings), 
            2) Retrieval (ค้นหาข้อมูลที่เกี่ยวข้อง), 3) Generation (สร้างคำตอบจาก LLM).
            """,
            "source_url": "https://pinecone.io/learn/rag/",
            "type": "documentation"
        }
    ]
    
    # บันทึกเป็นไฟล์ JSON
    with open('test_documents.json', 'w', encoding='utf-8') as f:
        json.dump(documents, f, ensure_ascii=False, indent=2)
    
    return documents

def test_import_and_search():
    """ทดสอบการ import และค้นหา"""
    print("=== Testing RAG System with Pinecone v7.x ===\n")
    
    # 1. สร้างเอกสารตัวอย่าง
    print("1. Creating sample documents...")
    docs = create_sample_documents()
    print(f"Created {len(docs)} sample documents\n")
    
    # 2. Import เข้า Pinecone
    print("2. Importing documents to Pinecone...")
    importer = FileDataImporter()
    count = importer.import_from_json('test_documents.json')
    print(f"Imported {count} documents\n")
    
    # 3. ทดสอบการค้นหา
    print("3. Testing vector search...")
    searcher = VectorSearcher()
    
    test_queries = [
        "Python programming language features",
        "machine learning algorithms in Python",
        "การทำ semantic search ด้วย vector database",
        "RAG architecture explanation",
        "Pinecone serverless architecture"
    ]
    
    for query in test_queries:
        print(f"\n--- Query: {query} ---")
        results = searcher.search(query, top_k=2)
        
        for i, result in enumerate(results, 1):
            print(f"{i}. Score: {result['score']:.3f}")
            print(f"   Title: {result['title']}")
            print(f"   Content: {result['content'][:100]}...")
            print(f"   Source: {result['source_url']}")
    
    # 4. ทดสอบการค้นหาด้วย filter
    print(f"\n4. Testing filtered search...")
    filtered_results = searcher.search_with_filters(
        "Python", 
        document_type="documentation", 
        top_k=3
    )
    print(f"Found {len(filtered_results)} documentation results for 'Python'")
    
    # 5. ดูสถิติ index
    print(f"\n5. Index statistics:")
    stats = searcher.get_index_stats()
    print(f"Total vectors: {stats.get('total_vector_count', 0)}")
    print(f"Namespaces: {list(stats.get('namespaces', {}).keys())}")

if __name__ == "__main__":
    test_import_and_search()