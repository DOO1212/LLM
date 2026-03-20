import json
import torch
import os
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# --- 환경 설정 및 모델 로드 ---
LABELS = ["재고", "생산", "재무", "규율", "기타"]
model_id = "Qwen/Qwen2.5-7B-Instruct"

# (bnb_config 및 모델 로드 로직은 기존 사용자 설정 유지)
# tokenizer = AutoTokenizer.from_pretrained(model_id)
# model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=bnb_config, ...)

def load_dynamic_examples(path="clarified_training_dataset.jsonl"):
    """수집된 학습 데이터를 읽어와서 프롬프트용 예시 문장을 만듭니다."""
    if not os.path.exists(path):
        # 학습 데이터가 없을 때의 기본값
        return "- 질문: \"사내 포털 사용법 알려줘\" -> 결과: \"기타\""
    
    examples = []
    try:
        with open(path, "r", encoding="utf-8") as f:
            lines = f.readlines()
            # 최신 교정 데이터 5개만 추출 (모델 혼란 방지 및 속도)
            for line in lines[-5:]:
                data = json.loads(line)
                examples.append(f"- 질문: \"{data['text']}\" -> 결과: \"{data['label']}\"")
    except Exception as e:
        print(f"[Warning] 예시 로드 중 오류: {e}")
        return "- 질문: \"부서 공지 어디서 봐\" -> 결과: \"기타\""
    
    return "\n".join(examples)

def build_prompt(query):
    """예시 데이터의 영향력을 극대화한 프롬프트"""
    dynamic_examples = load_dynamic_examples()
    
    return f"""당신은 사내 질문 분류 전문가입니다.
[필독 지침]
1. 아래 제공된 [최우선 참고 예시]에 현재 질문과 유사한 사례가 있다면, **가이드라인보다 예시의 결과를 무조건 우선하여 분류**하세요.
2. 예시에 질문이 있다면, 그 질문의 라벨을 그대로 따라야 합니다.

[최우선 참고 예시]
{dynamic_examples}

[카테고리 가이드라인]
- 재고: 수량 확인, 입출고 현황, 자산 가산
- 생산: 공정, 설비 가동, 작업 지침
- 재무: 보너스, 매출액, 비용 정산, 예산
- 규율: 재택근무(집에서 일하기), 근태, 휴가, 정책
- 기타: 사내 포털, 공지, 게시판, 부서 안내 등
질문: "{query}"

결과를 아래 JSON 형식으로만 응답하세요. 예시와 일치한다면 해당 카테고리에 1.0을 부여하세요.
{{"재고": 0.0, "생산": 0.0, "재무": 0.0, "규율": 0.0, "기타": 0.0}}
JSON:"""

def classify_query(query: str) -> dict:
    prompt = build_prompt(query)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    # === 여기부터 추가 ===
    print("\n" + "="*50)
    print(" [DEBUG] 모델에게 전달되는 실제 프롬프트:")
    print(prompt) 
    print("="*50 + "\n")
    # === 여기까지 추가 ===

    outputs = model.generate(
        **inputs,
        max_new_tokens=64,
        do_sample=False,
        temperature=0, # 확정적인 답변을 위해 0으로 설정
        stop_strings=["}", "}\n"],
        tokenizer=tokenizer
    )

    generated_text = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    
    # JSON 형태 보정
    full_response = generated_text.strip()
    if not full_response.startswith("{"):
        full_response = "{" + full_response
    if not full_response.endswith("}"):
        full_response = full_response + "}"

    try:
        # JSON 파싱 (사용자님이 쓰시는 extract_best_json_object 함수가 있다고 가정)
        # 없으면 json.loads(full_response)로 대체 가능
        parsed = json.loads(full_response)
        
        # 라벨이 누락된 경우 0.0으로 채워주는 안전장치
        probs = {label: float(parsed.get(label, 0.0)) for label in LABELS}
        return probs
    except Exception as e:
        print(f"[Parsing Error] {e} | Raw: {full_response}")
        return {"재고": 0.0, "생산": 0.0, "재무": 0.0, "규율": 0.0, "기타": 1.0}