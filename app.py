import os
import pandas as pd
import chromadb
import streamlit as st
import requests
import json
from chromadb.utils import embedding_functions

# --- [1] 시스템 및 DB 초기화 ---

# 1. DB 경로 및 클라이언트 설정
db_path = "/home/yuhan/LLM_DB/inventory_db"
client = chromadb.PersistentClient(path=db_path)

# 2. BGE-M3 임베딩 엔진 설정 (GPU 사용)
korean_ef = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="BAAI/bge-m3", 
    device="cuda"
)

# 3. 컬렉션 연결 (기본 생성)
collection = client.get_or_create_collection(
    name="apparel_master_kr", 
    embedding_function=korean_ef
)

# [유틸리티] 가격 문자열을 안전하게 정수(int)로 변환하는 함수
def safe_int_convert(val):
    try:
        if pd.isna(val):
            return 0
        # 콤마, 원, 공백 등을 제거하고 숫자만 남깁니다.
        clean_val = str(val).replace(',', '').replace('원', '').split('.')[0]
        num_only = ""
        for char in clean_val:
            if char.isdigit():
                num_only = num_only + char
        
        if num_only != "":
            return int(num_only)
        else:
            return 0
    except:
        return 0

# --- [2] 100개 파일 지능형 통합 색인 로직 ---

st.sidebar.title("🛠️ 데이터 관리")
if st.sidebar.button("🚀 전체 데이터 초기화 및 신규 색인"):
    with st.spinner("100개 파일을 정밀 수사하며 DB를 새로 구축 중..."):
        
        # 기존 데이터 완전 삭제 (데이터 타입 동기화를 위해 필수)
        try:
            client.delete_collection(name="apparel_master_kr")
        except:
            pass
            
        # 신규 컬렉션 생성 (HNSW 고성능 옵션 적용)
        collection = client.create_collection(
            name="apparel_master_kr",
            embedding_function=korean_ef,
            metadata={"hnsw:space": "cosine", "hnsw:M": 16, "hnsw:construction_ef": 200}
        )

        data_folder = "DATA"
        price_synonyms = ["단가", "가격", "금액", "Price", "판매가"]
        all_docs = []
        all_metas = []
        all_ids = []
        total_count = 0

        # 폴더 내 모든 엑셀 파일 처리
        for filename in os.listdir(data_folder):
            # 1. 확장자가 .xlsx인지 확인합니다.
            if filename.endswith(".xlsx"):
                
                # 2. [추가] 엑셀 임시 파일(~$로 시작)인지 확인합니다.
                if filename.startswith("~$"):
                    st.sidebar.info(f"⏭️ 임시 파일 제외: {filename}")
                    continue # 이 파일은 건너뛰고 다음 파일로 넘어갑니다.

                file_path = os.path.join(data_folder, filename)
                try:
                    df = pd.read_excel(file_path)
                    
                    # 가격 컬럼 찾기
                    found_price_col = None
                    for col in df.columns:
                        for syn in price_synonyms:
                            if syn in str(col):
                                found_price_col = col
                                break
                        if found_price_col:
                            break

                    for i, row in df.iterrows():
                        # 1. 모든 컬럼 합쳐서 검색 텍스트 생성
                        combined_text = ""
                        for val in row.values:
                            combined_text = combined_text + str(val) + " "
                        all_docs.append(combined_text.strip())

                        # 2. 가격을 정수형으로 변환 (필터링의 핵심)
                        p_val = 0
                        if found_price_col:
                            p_val = safe_int_convert(row[found_price_col])
                        
                        all_metas.append({
                            "price": p_val, 
                            "source_file": filename
                        })
                        all_ids.append(f"doc_{total_count}")
                        total_count = total_count + 1
                    
                    st.sidebar.write(f"✅ {filename} 적재 완료")
                except Exception as e:
                    st.sidebar.error(f"❌ {filename} 오류: {e}")

        # GPU 배치 적재 (5000개 단위)
        if all_docs:
            for start in range(0, len(all_docs), 5000):
                end = start + 5000
                collection.add(
                    documents=all_docs[start:end],
                    metadatas=all_metas[start:end],
                    ids=all_ids[start:end]
                )
            st.sidebar.success(f"🎊 총 {total_count}개 품목 색인 완료!")

# --- [3] 메인 채팅 및 동적 검색 로직 ---

st.title("📦 로컬 재고 관리 AI (RTX 5000 Ada)")
user_input = st.chat_input("질문을 입력하세요 (예: 이천 4만원 이하 패딩)")

# 세션 상태 초기화
if "final_matches" not in st.session_state:
    st.session_state.final_matches = []
if "current_page" not in st.session_state:
    st.session_state.current_page = 1
if "last_page_size" not in st.session_state:
    st.session_state.last_page_size = 50

if user_input:
    st.chat_message("user").write(user_input)
    
    with st.spinner("AI가 질문을 분석하고 재고를 찾는 중..."):
        ai_data = None
        results = None
        search_query = ""
        final_filter = None
        price_limit = 999999999 # 기본값 (무제한)

        # 1. AI 호출 (Ollama - Gemma2 27B)
        try:
            with open("prompt.txt", "r", encoding="utf-8") as f:
                system_prompt = f.read()
            
            ai_payload = {
                "model": "gemma2:27b", 
                "prompt": system_prompt + "\n질문: " + user_input + "\nJSON:", 
                "stream": False, 
                "format": "json",
                "options": {"temperature": 0.1}
            }
            ai_response = requests.post("http://localhost:11434/api/generate", json=ai_payload)
            ai_data = ai_response.json()
        except Exception as e:
            st.error(f"AI 서버 연결 실패: {e}")

        # 2. AI 응답 해석 및 동적 필터 추출
        if ai_data and 'response' in ai_data:
            try:
                parsed_json = json.loads(ai_data['response'])
                
                # 키워드 추출 (이천, 패딩 등)
                raw_keywords = parsed_json.get("keywords", [])
                if len(raw_keywords) > 0:
                    search_query = ""
                    for word in raw_keywords:
                        search_query = search_query + word + " "
                else:
                    search_query = user_input

                # 가격 제한선 동적 추출
                ai_filters = parsed_json.get("filters", [])
                if len(ai_filters) > 0:
                    f_item = ai_filters[0]
                    if "price" in f_item:
                        final_filter = f_item
                        p_info = f_item["price"]
                        if isinstance(p_info, dict):
                            # $lte 연산자 등에서 숫자만 추출
                            for val in p_info.values():
                                price_limit = int(val)
                        else:
                            price_limit = int(p_info)

                # 3. DB 검색 실행
                results = collection.query(
                    query_texts=[search_query.strip()],
                    n_results=500,
                    where=final_filter
                )

                # 4. 하이브리드 필터링 (키워드 + 가격 이중 검증)
                temp_list = []
                if results and results.get('documents'):
                    check_list = search_query.strip().split()
                    
                    idx = 0
                    for doc in results['documents'][0]:
                        dist = results['distances'][0][idx]
                        meta = results['metadatas'][0][idx]
                        current_price = meta.get('price', 0)
                        
                        # 키워드 포함 여부 확인 루프
                        is_match = True
                        for k in check_list:
                            if k.lower() not in doc.lower():
                                is_match = False
                                break
                        
                        # 모든 조건(키워드 + 동적 가격제한) 충족 시 추가
                        if is_match and current_price <= price_limit and dist <= 0.8:
                            temp_list.append({
                                "doc": doc, 
                                "meta": meta, 
                                "dist": dist
                            })
                        idx = idx + 1
                
                # 결과 세션 저장 및 페이지 리셋
                st.session_state.final_matches = temp_list
                st.session_state.current_page = 1

            except Exception as e:
                st.error(f"분석 결과 처리 중 오류: {e}")

# --- [4] 화면 출력 및 페이징 로직 ---

if len(st.session_state.final_matches) > 0:
    
    # 상단 컨트롤바
    st.divider()
    c_info, c_select = st.columns([2, 3])
    
    with c_info:
        st.write(f"🔍 총 **{len(st.session_state.final_matches)}**개의 검색 결과")
        
    with c_select:
        # 10, 30, 50개 선택 버튼
        current_idx = [10, 30, 50].index(st.session_state.last_page_size)
        selected_size = st.radio(
            "페이지당 보기", 
            [10, 30, 50], 
            index=current_idx,
            horizontal=True
        )

    # 개수 설정 변경 시 리셋
    if selected_size != st.session_state.last_page_size:
        st.session_state.last_page_size = selected_size
        st.session_state.current_page = 1
        st.rerun()

    # 데이터 구간 계산
    PAGE_SIZE = selected_size
    total_items = len(st.session_state.final_matches)
    total_pages = (total_items - 1) // PAGE_SIZE + 1
    
    start_idx = (st.session_state.current_page - 1) * PAGE_SIZE
    end_idx = start_idx + PAGE_SIZE
    page_items = st.session_state.final_matches[start_idx:end_idx]
    
    # 결과 출력
    with st.chat_message("assistant"):
        st.write(f"📄 **{st.session_state.current_page} / {total_pages}** 페이지")
        
        for item in page_items:
            doc = item['doc']
            meta = item['meta']
            dist = item['dist']
            price_fmt = format(meta.get('price', 0), ',')
            
            st.info(f"""
            📦 **품목**: {doc}
            💰 **가격**: {price_fmt}원 (정확도: {dist:.4f})
            📂 **출처**: {meta.get('source_file')} | 📈 **재고**: {meta.get('stock', 0)}개
            """)

        # 하단 페이지 이동 버튼
        st.divider()
        b1, b2, b3 = st.columns([1, 2, 1])
        with b1:
            if st.session_state.current_page > 1:
                if st.button("⬅️ 이전"):
                    st.session_state.current_page = st.session_state.current_page - 1
                    st.rerun()
        with b2:
            st.write(f"현재 **{st.session_state.current_page}** 페이지")
        with b3:
            if st.session_state.current_page < total_pages:
                if st.button("다음 ➡️"):
                    st.session_state.current_page = st.session_state.current_page + 1
                    st.rerun()

elif user_input:
    st.chat_message("assistant").warning(f"'{user_input}'에 대한 결과를 찾을 수 없습니다.")