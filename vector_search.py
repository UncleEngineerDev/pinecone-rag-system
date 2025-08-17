from typing import List, Dict, Any, Optional
from pinecone_client import PineconeClient
from embedding_model import EmbeddingModel

class VectorSearcher:
    def __init__(self, index_name="rag-documents"):
        self.client = PineconeClient()
        self.index = self.client.get_index(index_name)
        self.embedder = EmbeddingModel()
    
    def search(self, 
               query: str, 
               top_k: int = 5, 
               filter_dict: Optional[Dict[str, Any]] = None,
               include_metadata: bool = True) -> List[Dict]:
        """ค้นหา vectors ที่คล้ายกับ query"""
        
        # แปลง query เป็น embedding
        query_embedding = self.embedder.encode_single(query)
        
        # Query Pinecone v7.x
        results = self.index.query(
            vector=query_embedding,
            top_k=top_k,
            filter=filter_dict,
            include_metadata=include_metadata
        )
        
        # จัดรูปแบบผลลัพธ์
        search_results = []
        for match in results['matches']:
            result = {
                'id': match['id'],
                'score': match['score'],
                'content': match.get('metadata', {}).get('content', ''),
                'title': match.get('metadata', {}).get('title', ''),
                'source_url': match.get('metadata', {}).get('source_url', ''),
                'chunk_index': match.get('metadata', {}).get('chunk_index', 0),
                'metadata': match.get('metadata', {})
            }
            search_results.append(result)
        
        return search_results
    
    def search_with_filters(self, 
                          query: str, 
                          document_type: Optional[str] = None,
                          title_contains: Optional[str] = None,
                          top_k: int = 5) -> List[Dict]:
        """ค้นหาพร้อม filters"""
        
        filter_dict = {}
        
        if document_type:
            filter_dict['document_type'] = {'$eq': document_type}
        
        if title_contains:
            filter_dict['title'] = {'$in': [title_contains]}
        
        return self.search(query, top_k, filter_dict)
    
    def get_similar_chunks(self, 
                          document_id: str, 
                          chunk_index: int, 
                          top_k: int = 3) -> List[Dict]:
        """หา chunks ที่คล้ายกับ chunk ที่กำหนด"""
        
        # ดึง embedding ของ chunk ที่ต้องการ
        vector_id = f"{document_id}_{chunk_index}"
        
        try:
            fetch_result = self.index.fetch([vector_id])
            if vector_id in fetch_result['vectors']:
                original_vector = fetch_result['vectors'][vector_id]['values']
                
                # ค้นหา similar vectors
                results = self.index.query(
                    vector=original_vector,
                    top_k=top_k + 1,  # +1 เพราะจะได้ตัวเองด้วย
                    include_metadata=True
                )
                
                # กรอง original vector ออก
                similar_chunks = []
                for match in results['matches']:
                    if match['id'] != vector_id:
                        similar_chunks.append({
                            'id': match['id'],
                            'score': match['score'],
                            'content': match.get('metadata', {}).get('content', ''),
                            'title': match.get('metadata', {}).get('title', ''),
                        })
                
                return similar_chunks[:top_k]
        
        except Exception as e:
            print(f"Error finding similar chunks: {e}")
            return []
    
    def get_index_stats(self) -> Dict:
        """ดูสถิติของ index"""
        return self.index.describe_index_stats()

# ตัวอย่างการใช้งาน
if __name__ == "__main__":
    searcher = VectorSearcher()
    
    # ทดสอบการค้นหา
    queries = [
        "Python programming language",
        "machine learning algorithms", 
        "การเรียนรู้ของเครื่อง",
        "ภาษาไทย programming"
    ]
    
    for query in queries:
        print(f"\nQuery: {query}")
        results = searcher.search(query, top_k=3)
        
        for i, result in enumerate(results, 1):
            print(f"{i}. [{result['score']:.3f}] {result['title']}")
            print(f"   {result['content'][:100]}...")
    
    # ดูสถิติ index
    stats = searcher.get_index_stats()
    print(f"\nIndex stats: {stats}")