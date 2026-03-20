import json
import os
import re
import numpy as np
from sentence_transformers import SentenceTransformer
from app.services.query_engine import infer_label_hint

LABELS = ["재고", "생산", "재무", "규율", "기타"]
DATASET_PATH = "clarified_training_dataset.jsonl"
KNN_TOP_K = 5
KNN_MIN_SIM = 0.62

# 카테고리별 대표 예시 문장 (zero-shot 폴백용 프로토타입)
# 새로운 질문이 들어왔을 때 교정 데이터에 유사한 게 없으면 이것과 비교
CATEGORY_PROTOTYPES = {
    "재고": [
        "재고 수량 알려줘",
        "입출고 현황 확인해줘",
        "자산이 얼마나 있어",
        "재고 몇 개 남았어",
        "창고 재고 현황",
    ],
    "생산": [
        "공정 상태 어때",
        "설비 가동률 알려줘",
        "작업 지침 어떻게 돼",
        "생산량 얼마야",
        "오늘 생산 목표 달성했어",
    ],
    "재무": [
        "이번달 보너스 얼마야",
        "급여 언제 나와",
        "매출액 알려줘",
        "비용 정산 어떻게 해",
        "예산 확인하고 싶어",
    ],
    "규율": [
        "재택근무 가능해",
        "집에서 일해도 돼",
        "휴가 신청하려면",
        "근태 확인해줘",
        "사내 규정 알려줘",
        "연차 며칠 남았어",
    ],
    "기타": [
        "사내 시스템 접속이 안돼",
        "부서 공지 어디서 확인해",
        "사내 포털 사용법 알려줘",
        "회사 서류 양식 어디 있어",
        "회사 관련 일반 문의",
    ],
}

# 모델: 환경변수 EMBEDDING_MODEL (예: Qwen/Qwen3-Embedding-8B, 기본 KURE-v1)
_EMBEDDING_MODEL_ID = os.environ.get("EMBEDDING_MODEL", "nlpai-lab/KURE-v1")
_IS_QWEN3_EMBEDDING = "Qwen3" in _EMBEDDING_MODEL_ID or "Qwen3-Embedding" in _EMBEDDING_MODEL_ID

def _load_embedding_model():
    kwargs = {}
    if _IS_QWEN3_EMBEDDING:
        kwargs["model_kwargs"] = {"device_map": "auto"}
        kwargs["tokenizer_kwargs"] = {"padding_side": "left"}
        if os.environ.get("USE_FLASH_ATTENTION", "").lower() in ("1", "true", "yes"):
            kwargs["model_kwargs"]["attn_implementation"] = "flash_attention_2"
    return SentenceTransformer(_EMBEDDING_MODEL_ID, **kwargs)

print(f"[임베딩 모델 로딩] {_EMBEDDING_MODEL_ID} ...")
_model = _load_embedding_model()
print("[임베딩 모델 로딩 완료]")

def _encode_query(text_or_list, **kw):
    if _IS_QWEN3_EMBEDDING:
        kw["prompt_name"] = "query"
    kw.setdefault("convert_to_numpy", True)
    return _model.encode(text_or_list, **kw)

def _encode_docs(text_or_list, **kw):
    kw.setdefault("convert_to_numpy", True)
    return _model.encode(text_or_list, **kw)

# 카테고리 프로토타입 임베딩 사전 계산 (문서용)
_proto_embeddings: dict[str, np.ndarray] = {}
for label, sentences in CATEGORY_PROTOTYPES.items():
    embs = _encode_docs(sentences)
    _proto_embeddings[label] = np.mean(embs, axis=0)  # 평균 벡터


def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    norm = np.linalg.norm(a) * np.linalg.norm(b)
    if norm < 1e-10:
        return 0.0
    return float(np.dot(a, b) / norm)


def _load_training_data() -> list[dict]:
    if not os.path.exists(DATASET_PATH):
        return []
    raw_data = []
    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                row = json.loads(line)
                if row.get("label") == "회사기타":
                    row["label"] = "기타"
                raw_data.append(row)

    # 같은 질문(text)에 상충 라벨이 섞인 경우(학습 오염) 정규화합니다.
    # - 명시적 도메인 키워드가 있으면 해당 도메인 라벨을 우선
    # - 그렇지 않으면 마지막 라벨(가장 최근 교정)을 사용
    grouped: dict[str, list[dict]] = {}
    for row in raw_data:
        text = str(row.get("text", "")).strip()
        if not text:
            continue
        key = _normalize_query_key(text)
        grouped.setdefault(key, []).append({"text": text, "label": str(row.get("label", "")).strip()})

    merged: list[dict] = []
    for items in grouped.values():
        latest = items[-1]
        labels = [it["label"] for it in items if it["label"]]
        explicit = _explicit_domain_label(latest["text"])
        if explicit:
            if explicit in labels:
                chosen = explicit
            else:
                # 명시 도메인인데 기타로만 학습된 경우 오염으로 간주하고 도메인 라벨 채택
                chosen = explicit
        else:
            chosen = labels[-1] if labels else latest["label"]
        merged.append({"text": latest["text"], "label": chosen})
    return merged


def _normalize_query_key(text: str) -> str:
    return re.sub(r"\s+", "", (text or "").strip().lower())


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
    if any(k in q for k in ["공지", "게시판", "포털", "부서", "조직도", "양식"]):
        return "기타"
    return None


def _softmax_normalize(scores: dict[str, float]) -> dict[str, float]:
    """점수를 softmax로 정규화하여 합이 1이 되도록 합니다."""
    vals = np.array(list(scores.values()), dtype=float)
    # temperature scaling으로 분포를 더 뾰족하게
    vals = vals * 5.0
    vals = vals - np.max(vals)
    exp_vals = np.exp(vals)
    softmax_vals = exp_vals / exp_vals.sum()
    return dict(zip(scores.keys(), softmax_vals.tolist()))


def classify_query(query: str, return_debug: bool = False):
    """
    한국어 임베딩 기반 질문 분류.

    1단계: 교정 데이터(clarified_training_dataset.jsonl)에서 의미적으로
           가장 유사한 예시를 찾아 해당 라벨을 사용합니다. (kNN, 임계치 0.75)
    2단계: 교정 데이터에 유사한 예시가 없으면 카테고리 프로토타입과
           코사인 유사도를 비교하여 zero-shot 분류합니다.
    """
    query_emb = _encode_query(query)
    debug_info = {"source": "prototype", "top_k": [], "best_sim": 0.0}
    label_hint = infer_label_hint(query)

    # ── 1단계: 교정 데이터 top-k 가중치 투표 ──
    training_data = _load_training_data()
    if training_data:
        texts  = [d["text"]  for d in training_data]
        labels = [d["label"] for d in training_data]
        train_embs = _encode_docs(texts)

        sims = np.array([_cosine_sim(query_emb, e) for e in train_embs], dtype=float)
        if len(sims) > 0:
            top_k = min(KNN_TOP_K, len(sims))
            top_idx = np.argsort(sims)[::-1][:top_k]
            best_i = int(top_idx[0])
            best_sim = float(sims[best_i])

            # 상위 유사 예시의 라벨별 가중 합산
            # - 유사도 제곱: 높은 유사도 예시에 더 큰 비중
            # - 최근성 가중치: 최근 데이터일수록 최대 +15%
            label_scores = {l: 0.0 for l in LABELS}
            n = max(len(training_data), 1)
            for idx in top_idx:
                sim = float(max(sims[idx], 0.0))
                recency = 1.0 + 0.15 * (idx / n)
                weight = (sim ** 2) * recency
                label = labels[idx]
                if label in label_scores:
                    label_scores[label] += weight

            ranked = sorted(label_scores.items(), key=lambda x: x[1], reverse=True)
            top_label, top_score = ranked[0]
            second_score = ranked[1][1] if len(ranked) > 1 else 0.0
            gap = top_score - second_score

            if best_sim >= KNN_MIN_SIM:
                top_k_debug = []
                for idx in top_idx:
                    top_k_debug.append(
                        {
                            "text": texts[int(idx)],
                            "label": labels[int(idx)],
                            "sim": round(float(sims[int(idx)]), 4),
                        }
                    )
                print(
                    f"[임베딩 top-k] best='{texts[best_i]}' sim={best_sim:.2f} "
                    f"vote={top_label} score={top_score:.3f} gap={gap:.3f}"
                )
                debug_info = {
                    "source": "topk_vote",
                    "best_sim": round(best_sim, 4),
                    "top_k": top_k_debug,
                    "vote_scores": {k: round(float(v), 4) for k, v in label_scores.items()},
                    "vote_gap": round(float(gap), 4),
                }
                if label_hint and label_hint.get("label") in label_scores:
                    boost = float(label_hint.get("boost", 0.0))
                    label_scores[label_hint["label"]] += boost
                    debug_info["hint_applied"] = label_hint
                    ranked = sorted(label_scores.items(), key=lambda x: x[1], reverse=True)
                    debug_info["vote_gap"] = round(float(ranked[0][1] - ranked[1][1]), 4)
                probs = _softmax_normalize(label_scores)
                debug_info["normalized_scores"] = {k: round(float(v), 4) for k, v in probs.items()}
                if return_debug:
                    return probs, debug_info
                return probs

    # ── 2단계: 카테고리 프로토타입 비교 (zero-shot) ──
    raw_scores = {
        label: _cosine_sim(query_emb, proto_emb)
        for label, proto_emb in _proto_embeddings.items()
    }
    if label_hint and label_hint.get("label") in raw_scores:
        raw_scores[label_hint["label"]] += float(label_hint.get("boost", 0.0))
    probs = _softmax_normalize(raw_scores)

    best_label = max(probs, key=probs.get)
    print(f"[임베딩 프로토타입] 분류 결과: {best_label} (점수: {probs[best_label]:.2f})")

    debug_info = {
        "source": "prototype",
        "best_sim": round(float(raw_scores.get(best_label, 0.0)), 4),
        "prototype_scores": {k: round(float(v), 4) for k, v in raw_scores.items()},
    }
    if label_hint:
        debug_info["hint_applied"] = label_hint
    if return_debug:
        return probs, debug_info
    return probs
