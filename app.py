import streamlit as st
import chromadb
from sentence_transformers import SentenceTransformer
import torch
import math

# 페이지 레이아웃 설정
st.set_page_config(page_title="재고 페이징 검색", layout="wide")
st.title("📦 재고 통합 검색 (페이징 모드)")

@st.cache_resource
def load_resources():
    client = chromadb.PersistentClient(path="./my_inventory_db")
    collection = client.get_collection(name="inventory")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer('jhgan/ko-sroberta-multitask', device=device)
    return collection, model

collection, model = load_resources()

# 1. 사이드바 설정 (한 페이지에 몇 개씩 볼지)
items_per_page = st.sidebar.number_input("페이지당 항목 수", min_value=5, max_value=50, value=10)

# 2. 검색창
query = st.text_input("🔍 검색어를 입력하세요 (예: 인천 창고 패딩)", "")

if query:
    # 검색 수행 (최대 5,000건)
    query_embedding = model.encode([query]).tolist()
    results = collection.query(
        query_embeddings=query_embedding,
        n_results=5000
    )
    
    docs = results['documents'][0]
    total_items = len(docs)
    
    if total_items > 0:
        # 3. 페이징 계산
        total_pages = math.ceil(total_items / items_per_page)
        
        # 세션 스테이트를 이용해 현재 페이지 기억
        if 'current_page' not in st.session_state:
            st.session_state.current_page = 1
            
        st.write(f"✅ 총 {total_items}건의 검색 결과가 있습니다. (전체 {total_pages} 페이지)")

        # 4. 현재 페이지에 해당하는 데이터만 슬라이싱
        start_idx = (st.session_state.current_page - 1) * items_per_page
        end_idx = start_idx + items_per_page
        current_docs = docs[start_idx:end_idx]

        # 데이터 출력
        for i, doc in enumerate(current_docs):
            st.info(f"📍 결과 {start_idx + i + 1}: {doc}")

        # 5. 페이지 이동 버튼 (하단)
        col1, col2, col3, col4, col5 = st.columns([1, 1, 2, 1, 1])
        
        with col2:
            if st.button("이전 페이지") and st.session_state.current_page > 1:
                st.session_state.current_page -= 1
                st.rerun()

        with col3:
            st.write(f"**{st.session_state.current_page} / {total_pages}**")

        with col4:
            if st.button("다음 페이지") and st.session_state.current_page < total_pages:
                st.session_state.current_page += 1
                st.rerun()
    else:
        st.warning("검색 결과가 없습니다.")