from sentence_transformers import SentenceTransformer
import numpy as np

class EmbeddingModel:
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        """
        all-MiniLM-L6-v2: 384 dimensions, รองรับภาษาไทย, เร็ว
        all-mpnet-base-v2: 768 dimensions, คุณภาพสูงกว่า
        paraphrase-multilingual-MiniLM-L12-v2: 384 dimensions, หลายภาษา
        """
        print(f"Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.model_name = model_name
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        print(f"Embedding dimension: {self.embedding_dim}")
    
    def encode(self, texts):
        """แปลงข้อความหลายอันเป็น embeddings"""
        if isinstance(texts, str):
            texts = [texts]
        
        embeddings = self.model.encode(
            texts, 
            convert_to_tensor=False,
            show_progress_bar=True
        )
        return embeddings
    
    def encode_single(self, text):
        """แปลงข้อความเดียวเป็น embedding"""
        embedding = self.model.encode([text], convert_to_tensor=False)[0]
        return embedding.tolist()  # แปลงเป็น list สำหรับ Pinecone
    
    def batch_encode(self, texts, batch_size=32):
        """แปลง texts เป็น embeddings แบบ batch"""
        all_embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            embeddings = self.encode(batch)
            all_embeddings.extend(embeddings.tolist())
        
        return all_embeddings

# ทดสอบการใช้งาน
if __name__ == "__main__":
    embedder = EmbeddingModel()
    
    # ทดสอบ
    texts = [
        "Python เป็นภาษาโปรแกรมมิ่งที่เรียนรู้ง่าย",
        "Machine learning ใช้ในการสร้าง AI",
        "Pinecone เป็น vector database ที่มีประสิทธิภาพ"
    ]
    
    embeddings = embedder.encode(texts)
    print(f"Number of embeddings: {len(embeddings)}")
    print(f"Embedding shape: {embeddings[0].shape}")
    print(f"Sample embedding (first 5 values): {embeddings[0][:5]}")