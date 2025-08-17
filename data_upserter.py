import uuid
from typing import List, Dict, Any
from tqdm import tqdm

from pinecone_client import PineconeClient
from embedding_model import EmbeddingModel
from text_chunker import TextChunker

class PineconeDataUpserter:
    def __init__(self, index_name="rag-documents"):
        self.client = PineconeClient()
        self.index = self.client.get_index(index_name)
        self.embedder = EmbeddingModel()
        self.chunker = TextChunker(chunk_size=512, overlap=50)
    
    def prepare_vectors(self, document: Dict[str, Any]) -> List[Dict]:
        """เตรียม vectors สำหรับ upsert (v7.x format)"""
        vectors = []
        
        # แบ่ง content เป็น chunks
        chunks = self.chunker.chunk_by_sentences(document['content'])
        
        # สร้าง embeddings สำหรับแต่ละ chunk
        embeddings = self.embedder.batch_encode(chunks)
        
        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            vector_id = f"{document.get('id', str(uuid.uuid4()))}_{i}"
            
            # Metadata สำหรับ filtering และแสดงผล
            metadata = {
                'title': document['title'],
                'content': chunk,
                'chunk_index': i,
                'total_chunks': len(chunks),
                'source_url': document.get('source_url', ''),
                'document_type': document.get('type', 'general'),
                'created_at': document.get('created_at', ''),
            }
            
            # Pinecone v7.x vector format: tuple (id, values, metadata)
            vectors.append((vector_id, embedding, metadata))
        
        return vectors
    
    def upsert_document(self, document: Dict[str, Any]):
        """Upsert เอกสารเดียว"""
        vectors = self.prepare_vectors(document)
        
        # Upsert เป็น batch (Pinecone รองรับ max 100 vectors ต่อ batch)
        batch_size = 100
        for i in range(0, len(vectors), batch_size):
            batch = vectors[i:i + batch_size]
            self.index.upsert(vectors=batch)
        
        print(f"Upserted: {document['title']} ({len(vectors)} chunks)")
        return len(vectors)
    
    def upsert_documents(self, documents: List[Dict[str, Any]]):
        """Upsert หลายเอกสาร"""
        total_chunks = 0
        
        for doc in tqdm(documents, desc="Upserting documents"):
            chunks_count = self.upsert_document(doc)
            total_chunks += chunks_count
        
        print(f"Total chunks upserted: {total_chunks}")
        
        # รอให้ index update (Serverless เร็วกว่า)
        import time
        time.sleep(3)
        
        # ตรวจสอบสถานะ
        stats = self.index.describe_index_stats()
        print(f"Index stats: {stats}")
    
    def delete_by_filter(self, filter_dict: Dict[str, Any]):
        """ลบ vectors ที่ตรงกับ filter"""
        self.index.delete(filter=filter_dict)
        print(f"Deleted vectors with filter: {filter_dict}")

# ทดสอบ
if __name__ == "__main__":
    upserter = PineconeDataUpserter()
    
    # ข้อมูลตัวอย่าง
    sample_docs = [
        {
            'id': 'doc1',
            'title': 'Python Programming Basics',
            'content': '''Python เป็นภาษาโปรแกรมมิ่งระดับสูงที่ถูกพัฒนาโดย Guido van Rossum. 
            มันเป็นภาษาที่อ่านง่ายและเขียนง่าย เหมาะสำหรับผู้เริ่มต้น. 
            Python สามารถใช้ในการพัฒนา web applications, data analysis, machine learning, 
            และ artificial intelligence. Syntax ของ Python เรียบง่ายและเน้นความชัดเจน.''',
            'source_url': 'https://example.com/python-basics',
            'type': 'tutorial'
        }
    ]
    
    upserter.upsert_documents(sample_docs)