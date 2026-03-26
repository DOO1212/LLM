import pandas as pd
import chromadb
from chromadb.utils import embedding_functions
from sentence_transformers import SentenceTransformer
import torch

# 1. 아까 저장한 청킹 파일 불러오기
print("1. 청킹된 데이터를 불러오는 중...")
df = pd.read_csv('data/processed_chunks.csv')
documents = df['chunk'].tolist()
ids = [f"id_{i}" for i in range(len(documents))]

# 2. 로컬 벡터 DB 설정 (폴더 생성 및 연결)
# 이 폴더에 모든 지식이 저장됩니다.
client = chromadb.PersistentClient(path="./my_inventory_db")
collection = client.get_or_create_collection(name="inventory")

# 3. 임베딩 모델 로드 (한국어 성능이 검증된 모델)
# RTX 6000의 GPU(CUDA)를 사용하도록 설정합니다.
print("2. 임베딩 모델 로드 중 (GPU 가속 활용)...")
device = "cuda" if torch.cuda.is_available() else "cpu"
model = SentenceTransformer('jhgan/ko-sroberta-multitask', device=device)

# 4. 문장을 벡터로 변환 및 DB 저장
print(f"3. {len(documents)}건의 데이터를 벡터화하여 DB에 저장 중입니다. 잠시만 기다려주세요...")

# 데이터를 500개씩 나눠서 저장 (안정성을 위해)
batch_size = 500
for i in range(0, len(documents), batch_size):
    batch_docs = documents[i:i + batch_size]
    batch_ids = ids[i:i + batch_size]
    
    # 문장을 숫자로 변환
    batch_embeddings = model.encode(batch_docs).tolist()
    
    # DB에 추가
    collection.add(
        ids=batch_ids,
        embeddings=batch_embeddings,
        documents=batch_docs
    )
    print(f"   - {i + len(batch_docs)}개 완료...")

print("\n✅ 모든 데이터가 'my_inventory_db' 폴더에 안전하게 저장되었습니다!")