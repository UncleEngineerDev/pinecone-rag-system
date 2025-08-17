from pinecone import Pinecone, ServerlessSpec, CloudProvider, AwsRegion
import os
from dotenv import load_dotenv

load_dotenv()

class PineconeClient:
    def __init__(self):
        self.api_key = os.getenv('PINECONE_API_KEY')
        
        if not self.api_key:
            raise ValueError("PINECONE_API_KEY not found in environment variables")
        
        # Initialize Pinecone v7.x
        self.pc = Pinecone(api_key=self.api_key)
        print("Connected to Pinecone v7.x")
    
    def list_indexes(self):
        """แสดงรายการ indexes ทั้งหมด"""
        indexes = self.pc.list_indexes()
        return [index.name for index in indexes]
    
    def create_serverless_index(self, index_name, dimension=384):
        """สร้าง Serverless index ใหม่"""
        if index_name not in self.list_indexes():
            self.pc.create_index(
                name=index_name,
                dimension=dimension,
                metric='cosine',
                spec=ServerlessSpec(
                    cloud=CloudProvider.AWS,
                    region=AwsRegion.US_EAST_1
                )
            )
            print(f"Created serverless index: {index_name}")
        else:
            print(f"Index {index_name} already exists")
    
    def get_index(self, index_name):
        """เชื่อมต่อกับ index"""
        return self.pc.Index(index_name)
    
    def delete_index(self, index_name):
        """ลบ index"""
        if index_name in self.list_indexes():
            self.pc.delete_index(index_name)
            print(f"Deleted index: {index_name}")
    
    def describe_index(self, index_name):
        """ดูรายละเอียด index"""
        return self.pc.describe_index(index_name)

# ทดสอบการเชื่อมต่อ
if __name__ == "__main__":
    client = PineconeClient()
    print("Available indexes:", client.list_indexes())