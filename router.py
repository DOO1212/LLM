import json
import uuid
import os
import re
from datetime import datetime
from typing import Optional
from dataclasses import dataclass

# 임베딩 기반 분류기 임포트 (3B LLM 대체)
from router_embed import classify_query

# 임계치 설정
CONF_THRESHOLD = 0.99
GAP_THRESHOLD = 0.25
LOG_PATH = "router_logs.jsonl"
DATASET_PATH = "clarified_training_dataset.jsonl"

# 유사도 임계치: 이 값 이상이면 LLM 없이 교정 데이터를 직접 사용
SIMILARITY_THRESHOLD = 0.85


def _normalize_label(label: str | None) -> str:
    raw = (label or "").strip()
    return "기타" if raw == "회사기타" else raw


def _explicit_domain_label(query: str) -> str | None:
    q = (query or "").replace(" ", "")
    if any(k in q for k in ["재고", "입고", "출고", "반품", "발주", "품목", "자재", "창고", "제품", "단가"]):
        return "재고"
    if any(k in q for k in ["생산", "공정", "라인", "가동", "설비", "불량"]):
        return "생산"
    if any(k in q for k in ["재무", "매출", "비용", "예산", "손익", "미수금"]):
        return "재무"
    if any(k in q for k in ["규정", "연차", "재택근무", "근태", "보안", "퇴사", "입사"]):
        return "규율"
    if any(k in q for k in ["공지", "게시판", "포털", "사내", "부서"]):
        return "기타"
    return None


def _domain_signal_strength(query: str, label: str) -> int:
    q = (query or "").replace(" ", "")
    keyword_map = {
        "재고": ["재고", "입고", "출고", "반품", "발주", "품목", "자재", "창고", "안전재고", "제품", "단가"],
        "생산": ["생산", "공정", "라인", "가동", "설비", "불량", "수율", "실적"],
        "재무": ["재무", "매출", "매입", "비용", "예산", "손익", "원가", "정산"],
        "규율": ["규정", "규율", "근태", "출근", "퇴근", "연차", "휴가", "보안"],
        "기타": ["공지", "게시판", "포털", "사내", "부서", "양식"],
    }
    keywords = keyword_map.get(label, [])
    return sum(1 for kw in keywords if kw in q)


def _normalize(text: str) -> str:
    """공백·특수문자를 제거하고 소문자로 정규화합니다."""
    return re.sub(r"\s+", "", text).lower()


def _similarity(a: str, b: str) -> float:
    """두 문자열의 문자 단위 Jaccard 유사도를 반환합니다 (0.0 ~ 1.0)."""
    na, nb = _normalize(a), _normalize(b)
    if na == nb:
        return 1.0
    set_a = set(na)
    set_b = set(nb)
    if not set_a or not set_b:
        return 0.0
    # 포함 관계도 고려 (짧은 쪽이 긴 쪽 안에 있으면 높은 유사도)
    if na in nb or nb in na:
        shorter = min(len(na), len(nb))
        longer  = max(len(na), len(nb))
        return shorter / longer
    intersection = len(set_a & set_b)
    union        = len(set_a | set_b)
    return intersection / union


def lookup_correction(query: str) -> Optional[str]:
    """
    교정 데이터셋에서 현재 질문과 가장 유사한 항목을 찾아 라벨을 반환합니다.
    유사도가 SIMILARITY_THRESHOLD 미만이면 None을 반환합니다.
    """
    if not os.path.exists(DATASET_PATH):
        return None

    best_label = None
    best_score = 0.0
    top_matches: list[tuple[float, str]] = []

    try:
        with open(DATASET_PATH, "r", encoding="utf-8") as f:
            for line in f:
                item = json.loads(line.strip())
                score = _similarity(query, item["text"])
                if score > best_score:
                    best_score = score
                    best_label = _normalize_label(item["label"])
                    top_matches = [(score, best_label)]
                elif abs(score - best_score) < 1e-9:
                    top_matches.append((score, _normalize_label(item["label"])))
    except Exception as e:
        print(f"[lookup_correction 오류] {e}")
        return None

    if best_score >= SIMILARITY_THRESHOLD:
        # 동일/동점 매치가 여러 라벨로 충돌하면 명시 도메인 라벨을 우선합니다.
        if top_matches:
            labels = [label for _, label in top_matches]
            unique_labels = {l for l in labels if l}
            explicit = _explicit_domain_label(query)
            if len(unique_labels) > 1 and explicit in unique_labels:
                best_label = explicit

        # 명시적 업무 도메인 질문인데 교정 데이터가 기타로 오염된 경우는 무시
        explicit = _explicit_domain_label(query)
        if explicit in ("재고", "생산", "재무", "규율") and best_label == "기타":
            return None
        print(f"[교정 데이터 히트] 유사도={best_score:.2f} → '{best_label}' (LLM 건너뜀)")
        return best_label

    return None

@dataclass
class RoutingResult:
    probs: dict
    top1_label: str
    top1_score: float
    top2_label: str
    top2_score: float
    route_status: str
    clarification_question: Optional[str]
    decision_debug: Optional[dict] = None


def _is_ambiguous(top1_score: float, top2_score: float) -> bool:
    confidence_gap = top1_score - top2_score
    return (top1_score < CONF_THRESHOLD) and (confidence_gap < GAP_THRESHOLD)

def map_execution_target(label: str):
    mapping = {
        "재고": "inventory_api",
        "생산": "production_api",
        "재무": "finance_api",
        "규율": "policy_search",
        "기타": None,
    }
    return mapping.get(label)

def handle_non_business_question():
    message = "본 챗봇은 재고, 생산, 재무, 사내 규정 관련 업무만 처리할 수 있습니다."
    print(f"\n[안내] {message}")

def append_jsonl(path: str, row: dict):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")

def save_router_log(session_id, user_query, routing_result, user_choice=None, final_label=None, note=None):
    confidence_gap = round(routing_result.top1_score - routing_result.top2_score, 4)
    execution_target = map_execution_target(final_label) if final_label else None

    log_row = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "request_id": str(uuid.uuid4()),
        "session_id": session_id,
        "user_query": user_query,
        "probs": routing_result.probs,
        "top1_label": routing_result.top1_label,
        "top1_score": routing_result.top1_score,
        "top2_label": routing_result.top2_label,
        "top2_score": routing_result.top2_score,
        "confidence_gap": confidence_gap,
        "route_status": routing_result.route_status,
        "final_label": final_label,
        "execution_target": execution_target,
        "note": note
    }
    append_jsonl(LOG_PATH, log_row)

def make_clarification(query: str, top1: str, top2: str):
    return "질문의 카테고리를 선택해주세요."

def router(query: str) -> RoutingResult:
    """
    1단계: 교정 데이터셋에서 유사 질문을 먼저 검색합니다.
           히트하면 LLM을 건너뛰고 즉시 확정합니다.
    2단계: 히트 없으면 7B LLM에게 분류를 요청합니다.
    """
    LABELS = ["재고", "생산", "재무", "규율", "기타"]

    # ── 1단계: 교정 데이터 직접 조회 (LLM보다 항상 우선) ──
    corrected_label = lookup_correction(query)
    if corrected_label:
        probs = {l: 0.0 for l in LABELS}
        probs[corrected_label] = 1.0
        second_labels = [l for l in LABELS if l != corrected_label]
        return RoutingResult(
            probs=probs,
            top1_label=corrected_label,
            top1_score=1.0,
            top2_label=second_labels[0],
            top2_score=0.0,
            route_status="resolved",
            clarification_question=None,
            decision_debug={
                "source": "dataset_lookup",
                "best_sim": 1.0,
                "note": "교정 데이터 직접 히트",
            },
        )

    # ── 2단계: 교정 데이터 없음 → LLM 호출 ──
    try:
        probs, embed_debug = classify_query(query, return_debug=True)
    except Exception as e:
        print(f"[LLM 호출 에러] {e}")
        probs = {l: 0.0 for l in LABELS}
        probs["기타"] = 1.0
        embed_debug = {"source": "error_fallback", "error": str(e)}

    sorted_probs = sorted(probs.items(), key=lambda x: x[1], reverse=True)
    top1_label, top1_score = sorted_probs[0]
    top2_label, top2_score = sorted_probs[1]
    if _is_ambiguous(top1_score, top2_score):
        return RoutingResult(
            probs=probs,
            top1_label=top1_label,
            top1_score=top1_score,
            top2_label=top2_label,
            top2_score=top2_score,
            route_status="needs_clarification",
            clarification_question=make_clarification(query, top1_label, top2_label),
            decision_debug=embed_debug,
        )

    return RoutingResult(
        probs=probs,
        top1_label=top1_label,
        top1_score=top1_score,
        top2_label=top2_label,
        top2_score=top2_score,
        route_status="resolved",
        clarification_question=None,
        decision_debug=embed_debug,
    )

def process_query(session_id: str, query: str, clarified_label: Optional[str] = None):
    routing = router(query)

    if not routing:
        return {"status": "error", "final_label": None}

    if clarified_label:
        clarified_label = _normalize_label(clarified_label)
        valid_labels = ["재고", "생산", "재무", "규율", "기타"]
        if clarified_label not in valid_labels:
            return {"status": "error", "final_label": None, "message": "유효하지 않은 라벨입니다."}

        final_label = clarified_label
        save_router_log(
            session_id,
            query,
            routing,
            final_label=final_label,
            note="사용자 선택 명확화",
        )
        return {
            "status": "resolved",
            "final_label": final_label,
            "execution_target": map_execution_target(final_label),
            "score": float(routing.probs.get(final_label, routing.top1_score)),
            "clarified_label_applied": True,
            "decision_debug": {
                "source": "user_clarification",
                "selected_label": final_label,
                "base": routing.decision_debug or {},
            },
        }

    if routing.route_status == "resolved":
        final_label = routing.top1_label
        explicit = _explicit_domain_label(query)
        if explicit in ("재고", "생산", "재무", "규율") and final_label == "기타":
            final_label = explicit
        # 분류 점수가 박빙인데 도메인 키워드가 강하면 명시 도메인으로 보정
        confidence_gap = float(routing.top1_score - routing.top2_score)
        if explicit in ("재고", "생산", "재무", "규율") and final_label != explicit:
            signal = _domain_signal_strength(query, explicit)
            top1_signal = _domain_signal_strength(query, final_label)
            if signal >= 2 and (confidence_gap < 0.12 or top1_signal == 0):
                final_label = explicit
        
        note = "교정 데이터 히트" if routing.top1_score == 1.0 and routing.top2_score == 0.0 else "자동 확정"
        if explicit in ("재고", "생산", "재무", "규율") and final_label == explicit and routing.top1_label == "기타":
            note = "도메인 의도 강제 보정"
        elif explicit in ("재고", "생산", "재무", "규율") and final_label == explicit and routing.top1_label != explicit:
            note = "도메인 키워드 보정"
        save_router_log(session_id, query, routing, final_label=final_label, note=note)
        return {
            "status": "resolved",
            "final_label": final_label,
            "execution_target": map_execution_target(final_label),
            "score": routing.top1_score,
            "decision_debug": routing.decision_debug,
        }
    
    # 명확화 요청이어도 로그에는 남김 (모든 질문 로그 보장)
    save_router_log(
        session_id, query, routing,
        final_label=routing.top1_label,
        note="명확화 요청(저신뢰도)",
    )
    return {
        "status": "needs_clarification",
        "question": routing.clarification_question,
        "candidates": [label for label, _ in sorted(routing.probs.items(), key=lambda x: x[1], reverse=True)],
        "scores": {
            label: score for label, score in sorted(routing.probs.items(), key=lambda x: x[1], reverse=True)
        },
        "decision_debug": routing.decision_debug,
    }

if __name__ == "__main__":
    session = "terminal-test"
    print("=== 7B Router Test Mode (종료: exit) ===")
    
    while True:
        user_input = input("\n질문: ").strip()
        if user_input.lower() in ["exit", "quit", "종료"]: break
        if not user_input: continue
        
        # 1. 라우팅 실행
        res = process_query(session, user_input)
        print(f"결과: {res}")

        # 2. 분류 오류 버튼(수동 교정) 기능 추가
        # 결과가 'resolved'(확정)이거나 'blocked'(차단)일 때만 피드백을 받습니다.
        if res.get("status") in ["resolved", "blocked"]:
            feedback = input("\n[피드백] 분류가 틀렸나요? (교정하려면 라벨명 입력 / 아니면 Enter): ").strip()
            
            # 유효한 라벨을 입력했을 경우 수동 교정 로그 저장
            valid_labels = ["재고", "생산", "재무", "규율", "기타"]
            if feedback in valid_labels:
                # 위에서 정의한 로그 저장 함수 호출
                save_router_log(
                    session_id=session,
                    user_query=user_input,
                    routing_result=router(user_input), # 현재 상태 다시 로드
                    final_label=feedback,
                    note="사용자 수동 교정" # ★ 이 메모가 학습 데이터 추출의 키워드입니다.
                )
                print(f"✅ 정답이 '{feedback}'(으)로 교정되어 로그에 기록되었습니다.")