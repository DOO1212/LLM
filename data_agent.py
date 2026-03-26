# 1. 필수 라이브러리 임포트
import pandas as pd
from langchain_community.chat_models import ChatOllama
from langchain_experimental.agents import create_pandas_dataframe_agent

# 2. 데이터 불러오기
# 대용량 처리에 유리한 Parquet 포맷의 데이터를 판다스 데이터프레임으로 로드함
print("데이터를 불러오는 중입니다... 📊")
df = pd.read_parquet('/mnt/d/LLM/convert/data.parquet')

# 3. 로컬 LLM 연결 (Llama 3.1)
# temperature=0으로 설정하여 창의성보다는 팩트 기반의 정확한 분석 결과를 유도함
llm = ChatOllama(model="llama3.1", temperature=0)

# 4. 데이터 분석 에이전트(Agent) 생성
print("Llama 3.1 분석 에이전트를 준비 중입니다...\n")
agent = create_pandas_dataframe_agent(
    llm,
    df,
    verbose=True,                # 터미널에 LLM의 사고 과정(생성 및 실행한 코드) 출력
    allow_dangerous_code=True,   # LLM이 직접 코드를 실행할 수 있도록 권한 허용 (로컬 전용)
    handle_parsing_errors=True,  # 코드 문법 에러 발생 시 죽지 않고 LLM에게 재시도 기회 제공
    max_iterations=10            # 최대 사고 횟수를 10번으로 제한하여 무한 루프 방지
)

# 5. 자연어 질문 정의
question = "데이터가 총 몇 줄인지 알려주고, 각 컬럼(열)의 이름들이 무엇인지 한국어로 정리해줘."

print(f"질문: {question}")
print("-" * 50)

# 6. 에이전트 실행 및 예외 처리
try:
    # 에이전트가 질문을 분석하고 코드를 스스로 실행하여 최종 답변 도출
    response = agent.invoke(question)
    print("\n최종 답변 🎯:")
    print(response['output'])
except Exception as e:
    # 실행 중 발생하는 예기치 못한 에러 안전하게 출력
    print(f"\n앗, 에러 발생!: {e}")