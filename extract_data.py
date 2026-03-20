import json


def _normalize_label(label: str | None) -> str:
    raw = (label or "").strip()
    if raw == "회사기타":
        return "기타"
    return raw


def _explicit_domain_label(query: str) -> str | None:
    q = (query or "").replace(" ", "")
    if any(k in q for k in ["재고", "입고", "출고", "반품", "발주", "품목", "자재", "창고", "안전재고"]):
        return "재고"
    if any(k in q for k in ["생산", "공정", "라인", "가동", "설비", "불량", "수율", "계획"]):
        return "생산"
    if any(k in q for k in ["재무", "매출", "비용", "예산", "손익", "미수금", "원가", "정산"]):
        return "재무"
    if any(k in q for k in ["규정", "규율", "근태", "출근", "퇴근", "연차", "휴가", "보안"]):
        return "규율"
    return None


def run_extraction():
    log_path = "router_logs.jsonl"
    output_path = "clarified_training_dataset.jsonl"
    seed_path = "data/random_questions_1000.jsonl"
    dataset = []
    
    # 1) 기본 학습셋(랜덤 생성 1000개) 우선 적재
    try:
        with open(seed_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                text = (row.get("question") or "").strip()
                label = _normalize_label(row.get("label"))
                if text and label:
                    dataset.append({"text": text, "label": label})
    except FileNotFoundError:
        pass

    # 2) 사용자 교정/명확화 로그를 추가 반영
    try:
        with open(log_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                if row.get("note") in ("사용자 수동 교정", "사용자 선택 명확화"):
                    text = (row.get("user_query") or "").strip()
                    label = _normalize_label(row.get("final_label"))
                    if text and label:
                        explicit = _explicit_domain_label(text)
                        # 명시 도메인 질문이 기타로 교정되어 학습 오염되는 것을 방지
                        if explicit and label == "기타":
                            continue
                        dataset.append({"text": text, "label": label})
    except FileNotFoundError:
        if not dataset:
            print("❌ 로그 파일이 아직 없습니다. 질문을 먼저 진행해주세요!")

    # 중복 제거
    unique_data = { (d['text'], d['label']): d for d in dataset }.values()

    with open(output_path, "w", encoding="utf-8") as f:
        for item in unique_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
            
    print(f"✅ 성공적으로 {len(unique_data)}개의 정답 데이터를 추출했습니다!")
    print(f"👉 생성된 파일: {output_path}")

if __name__ == "__main__":
    run_extraction()