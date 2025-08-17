import re
from typing import List

class TextChunker:
    def __init__(self, chunk_size=512, overlap=50):
        self.chunk_size = chunk_size
        self.overlap = overlap
    
    def chunk_by_sentences(self, text: str) -> List[str]:
        """แบ่ง chunk โดยใช้ประโยค (เหมาะกับเอกสารภาษาไทย)"""
        # แบ่งเป็นประโยค (รองรับภาษาไทย)
        sentences = re.split(r'[.!?।]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            # ถ้าเพิ่มประโยคนี้แล้วยาวเกิน chunk_size
            if len(current_chunk + sentence) > self.chunk_size:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                    current_chunk = sentence + " "
                else:
                    # ประโยคเดียวยาวเกิน chunk_size
                    chunks.append(sentence[:self.chunk_size])
                    current_chunk = ""
            else:
                current_chunk += sentence + " "
        
        # เพิ่ม chunk สุดท้าย
        if current_chunk.strip():
            chunks.append(current_chunk.strip())
        
        return chunks
    
    def chunk_by_characters(self, text: str) -> List[str]:
        """แบ่ง chunk โดยใช้จำนวนตัวอักษร พร้อม overlap"""
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + self.chunk_size
            chunk = text[start:end]
            chunks.append(chunk)
            start = end - self.overlap  # overlap เพื่อไม่ให้สูญเสียบริบท
        
        return [chunk.strip() for chunk in chunks if chunk.strip()]
    
    def chunk_by_paragraphs(self, text: str) -> List[str]:
        """แบ่ง chunk โดยใช้ paragraph"""
        paragraphs = text.split('\n\n')
        chunks = []
        current_chunk = ""
        
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
                
            if len(current_chunk + para) > self.chunk_size:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                
                # ถ้า paragraph เดียวยาวเกิน chunk_size
                if len(para) > self.chunk_size:
                    chunks.extend(self.chunk_by_sentences(para))
                    current_chunk = ""
                else:
                    current_chunk = para + "\n\n"
            else:
                current_chunk += para + "\n\n"
        
        if current_chunk.strip():
            chunks.append(current_chunk.strip())
        
        return chunks

# ทดสอบ
if __name__ == "__main__":
    chunker = TextChunker(chunk_size=200, overlap=50)
    
    text = """
    Python เป็นภาษาโปรแกรมมิ่งระดับสูงที่มีความสามารถหลากหลาย. 
    มันถูกออกแบบมาให้อ่านง่ายและเขียนง่าย. 
    Python ถูกใช้ในหลายสาขา เช่น web development, data science, และ artificial intelligence.
    
    Machine Learning เป็นสาขาหนึ่งของ AI ที่กำลังได้รับความนิยม.
    มีหลาย libraries ที่ช่วยในการพัฒนา ML models เช่น scikit-learn, TensorFlow, และ PyTorch.
    """
    
    chunks = chunker.chunk_by_paragraphs(text)
    for i, chunk in enumerate(chunks):
        print(f"Chunk {i+1}: {chunk}\n---")