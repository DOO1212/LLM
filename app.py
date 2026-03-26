import streamlit as st
import chromadb
from sentence_transformers import SentenceTransformer
import requests
import json
import torch

# [설정] 화면 구성
st.set_page_config(page_title="RTX 고성능 AI 비서", layout="wide")
st.title("🚀 Gemma 2 27B 기반 지능형 재고 관리")

@st.cache_resource
def load_resources():
    client = chromadb.PersistentClient(path="./my_inventory_db")
    collection = client.get_collection(name="inventory")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer('jhgan/ko-sroberta-multitask', device=device)
    return collection, model

collection, model = load_resources()

# [스트리밍 함수] Ollama의 응답을 한 글자씩 가져옵니다.
def ask_gemma_stream(system_prompt, user_prompt):
    url = "http://localhost:11434/api/generate"
    data = {
        "model": "gemma2:27b", # 27B 모델 지정
        "prompt": f"<start_of_turn>user\n{system_prompt}\n\n질문: {user_prompt}<end_of_turn>\n<start_of_turn>model\n",
        "stream": True # 스트리밍 활성화
    }
    
    response = requests.post(url, json=data, stream=True)
    for line in response.iter_lines():
        if line:
            body = json.loads(line)
            yield body.get("response", "") # 한 단어씩 반환

# 채팅 UI 구현
if prompt := st.chat_input("인천 창고 패딩 4만원 이하 다 알려줘"):
    st.chat_message("user").markdown(prompt)
    
    # 1. 벡터 검색 (의미 분석)
    with st.spinner("RTX GPU가 데이터를 정밀 분석 중..."):
        query_embedding = model.encode([prompt]).tolist()
        results = collection.query(query_embeddings=query_embedding, n_results=150)
        context_docs = "\n".join(results['documents'][0])

    # 2. 시스템 프롬프트 (Gemma 2 전용 형식)
    system_instructions = f"""
    당신은 한국 기업의 전문 재고 관리 비서입니다. 
    반드시 한국어로만 답변하고, 제공된 [재고 정보]를 바탕으로 사실만 말하세요.
    가격 조건이 있다면 정확히 필터링해서 보여주세요.

    [재고 정보]
    {context_docs}
    """

    # 3. 실시간 스트리밍 답변 출력
    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        full_response = ""
        
        for chunk in ask_gemma_stream(system_instructions, prompt):
            full_response += chunk
            response_placeholder.markdown(full_response + "▌") # 커서 효과
        response_placeholder.markdown(full_response)