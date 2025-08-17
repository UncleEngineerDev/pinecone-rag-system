from pinecone_client import PineconeClient
import time

def create_rag_index():
    """สร้าง Serverless index สำหรับ RAG project"""
    client = PineconeClient()
    
    index_name = "rag-documents"
    dimension = 384  # all-MiniLM-L6-v2 embedding dimension
    
    # สร้าง Serverless index
    client.create_serverless_index(
        index_name=index_name,
        dimension=dimension
    )
    
    # รอให้ index พร้อมใช้งาน
    print("Waiting for index to be ready...")
    time.sleep(30)  # Serverless index เร็วกว่า Pod
    
    # ตรวจสอบสถานะ index
    index = client.get_index(index_name)
    stats = index.describe_index_stats()
    print(f"Index stats: {stats}")
    
    # ดูรายละเอียด index
    index_info = client.describe_index(index_name)
    print(f"Index info: {index_info}")
    
    return index

if __name__ == "__main__":
    index = create_rag_index()
    print("Serverless index created successfully!")