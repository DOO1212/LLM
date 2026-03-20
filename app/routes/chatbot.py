import json
import os
import re
from collections import Counter
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from flask import Blueprint, jsonify, request, session

from app.auth import login_required
from app.db import get_cursor
from app.services.audit_service import log_action
from app.services.chat_context import get_context, set_context
from extract_data import run_extraction


chatbot_bp = Blueprint("chatbot", __name__)


@dataclass
class RoutingResult:
    probs: dict
    top1_label: str
    top1_score: float
    top2_label: str
    top2_score: float
    route_status: str
    clarification_question: str | None


def _load_data_reader():
    from data_reader import get_watch_dir, query_data, rescan, get_all_files_columns

    return get_watch_dir, query_data, rescan, get_all_files_columns


def _generate_answer(query, label, table_data):
    from answer_llm import generate_answer

    return generate_answer(query, label, table_data)


def _process_query(session_id, query, clarified_label=None):
    from router import process_query

    return process_query(session_id, query, clarified_label=clarified_label)


def _save_router_log(session_id, user_query, routing_result, final_label, note):
    from router import save_router_log

    return save_router_log(
        session_id=session_id,
        user_query=user_query,
        routing_result=routing_result,
        final_label=final_label,
        note=note,
    )


def _compute_structured_answer(query, label, table_data):
    from app.services.query_engine import compute_structured_answer

    return compute_structured_answer(query, label, table_data)


def _compute_table_operations_answer(query, label, table_data):
    from app.services.table_operations import run_table_operations

    return run_table_operations(query, label, table_data)


def _is_context_followup(query: str) -> bool:
    q = (query or "").strip()
    compact = q.replace(" ", "")

    # 도메인 키워드가 명확한 질문은 후속질문으로 보지 않습니다.
    explicit_domain_keywords = [
        "재고", "입고", "출고", "반품",
        "생산", "공정", "가동",
        "재무", "매출", "비용", "예산",
        "규정", "공지", "게시판",
        "제품", "단가",
    ]
    if any(k in compact for k in explicit_domain_keywords):
        return False

    keywords = [
        "그거",
        "이거",
        "저거",
        "방금",
        "아까",
        "이전",
        "지난",
    ]
    # 매우 짧고 지시어 성격이 강한 질문만 후속질문으로 간주
    if len(compact) <= 3:
        return True
    return any(k in compact for k in keywords)


def _has_explicit_domain_intent(query: str) -> bool:
    compact = (query or "").replace(" ", "")
    explicit_domain_keywords = [
        "재고", "입고", "출고", "반품",
        "생산", "공정", "가동",
        "재무", "매출", "비용", "예산",
        "규정", "공지", "게시판",
        "제품", "단가",
    ]
    return any(k in compact for k in explicit_domain_keywords)


def _to_number(value):
    text = str(value).strip().replace(",", "")
    if not text:
        return None
    if re.fullmatch(r"-?\d+(\.\d+)?", text):
        try:
            return float(text)
        except ValueError:
            return None
    return None


def _contains_any(q: str, aliases: tuple[str, ...]) -> bool:
    return any(a in q for a in aliases)


def _has_explicit_time_window(query: str) -> bool:
    q = (query or "").replace(" ", "")
    aliases = (
        "오늘", "금일", "당일",
        "어제", "전일",
        "내일", "익일", "명일",
        "이번주", "금주", "당주", "이번주간",
        "지난주", "저번주", "전주", "직전주",
        "다음주", "담주", "차주", "익주",
        "이번달", "금월", "당월",
        "지난달", "저번달", "전월", "직전월",
        "다음달", "차월", "익월",
        "올해", "금년", "당해",
        "작년", "전년",
        "내년", "익년", "명년",
    )
    return _contains_any(q, aliases)


def _time_window_text(query: str) -> str | None:
    q = query.replace(" ", "")
    today_aliases = ("오늘", "금일", "당일")
    yesterday_aliases = ("어제", "전일")
    tomorrow_aliases = ("내일", "익일", "명일")
    this_week_aliases = ("이번주", "금주", "당주", "이번주간")
    last_week_aliases = ("지난주", "저번주", "전주", "직전주")
    next_week_aliases = ("다음주", "담주", "차주", "익주")
    this_month_aliases = ("이번달", "금월", "당월")
    last_month_aliases = ("지난달", "저번달", "전월", "직전월")
    next_month_aliases = ("다음달", "차월", "익월")
    this_year_aliases = ("올해", "금년", "당해")
    last_year_aliases = ("작년", "전년")
    next_year_aliases = ("내년", "익년", "명년")
    if _contains_any(q, today_aliases):
        return "오늘"
    if _contains_any(q, yesterday_aliases):
        return "어제"
    if _contains_any(q, tomorrow_aliases):
        return "내일"
    if _contains_any(q, this_week_aliases):
        return "이번 주"
    if _contains_any(q, last_week_aliases):
        return "지난 주"
    if _contains_any(q, next_week_aliases):
        return "다음 주"
    if _contains_any(q, this_month_aliases):
        return "이번 달"
    if _contains_any(q, last_month_aliases):
        return "지난 달"
    if _contains_any(q, next_month_aliases):
        return "다음 달"
    if _contains_any(q, this_year_aliases):
        return "올해"
    if _contains_any(q, last_year_aliases):
        return "작년"
    if _contains_any(q, next_year_aliases):
        return "내년"
    return None


def _default_action_by_label(label: str) -> str:
    if label == "재고":
        return "수량 급감/미달 품목을 우선 점검하고 발주 우선순위를 확정하세요."
    if label == "생산":
        return "지연 또는 저실적 라인을 우선 점검하고 인력/설비/자재 원인을 확인하세요."
    if label == "재무":
        return "변동 폭이 큰 항목부터 근거 전표를 점검하고 예산 대비 차이를 확인하세요."
    return "핵심 수치를 검토한 뒤 후속 조치 담당자와 일정을 확정하세요."


def _is_attendance_query(query: str) -> bool:
    q = (query or "").replace(" ", "")
    keywords = [
        "출근",
        "퇴근",
        "근태",
        "출근시간",
        "퇴근시간",
        "지각",
        "조퇴",
        "근무시간",
        "출퇴근",
    ]
    return any(k in q for k in keywords)


def _is_notice_query(query: str) -> bool:
    q = (query or "").replace(" ", "")
    keywords = ["공지", "공지사항", "사내공지", "게시판공지", "안내사항"]
    return any(k in q for k in keywords)


def _notice_board_answer(_query: str) -> str:
    """게시판 제거 후 공지 질의에 대한 안내."""
    return "게시판 기능이 제거되어 공지사항 조회를 제공하지 않습니다. 필요한 안내는 담당 부서에 문의해 주세요."


def _attendance_policy_answer(label: str) -> str:
    if label == "규율":
        return (
            "근태(출근/퇴근) 문의로 확인되었습니다.\n"
            "현재 포털에서는 근태 모듈을 운영하지 않아 출근시간 조회/기록 기능은 제공하지 않습니다.\n"
            "필요 시 인사/총무 규정 문서 또는 담당 부서에 확인해 주세요."
        )
    return (
        "현재 포털에서는 근태(출근/퇴근) 기능을 사용하지 않습니다.\n"
        "지금은 대시보드, 챗봇/분류, 전자결재 기능만 운영 중입니다."
    )


def _is_garbled_answer_text(text: str | None) -> bool:
    if not text:
        return True
    low = text.lower()
    if "filefile" in low or "modulefile" in low:
        return True
    # 정상 문장에 비해 특수문자/깨진 문자 비율이 너무 높은 경우
    cleaned = re.sub(r"[A-Za-z0-9가-힣\s\.,:;!\?\-\(\)\[\]\/]", "", text)
    if len(text) > 40 and (len(cleaned) / max(len(text), 1)) > 0.2:
        return True
    return False


def _is_structured_report_query(query: str) -> bool:
    q = (query or "").replace(" ", "")
    keywords = [
        "현황",
        "조회",
        "요약",
        "보고",
        "리스트",
        "목록",
        "개수",
        "갯수",
        "건수",
        "수량",
        "합계",
        "총",
        "비교",
        "추이",
        "예정",
    ]
    return any(k in q for k in keywords)


def _format_trust_template(
    query: str,
    label: str,
    table_data: dict | None,
    base_answer: str | None,
) -> str | None:
    """
    답변을 보고/검토에 적합한 고정 포맷으로 정리합니다.
    """
    if not table_data:
        return base_answer

    summary = str(table_data.get("summary") or "").strip()
    filename = str(table_data.get("filename") or "데이터 파일")
    columns = [str(c) for c in (table_data.get("columns") or []) if str(c).strip()]
    rows = table_data.get("rows") or []
    structured_query = _is_structured_report_query(query)

    summary_parts = [p.strip() for p in summary.split("|") if p.strip()]
    summary_line = summary_parts[0] if summary_parts else f"{label} 조회 결과를 요약했습니다."

    metrics = []
    for part in summary_parts:
        if part not in metrics:
            metrics.append(part)
        if len(metrics) >= 3:
            break
    if f"총 {len(rows)}개 항목 조회" not in metrics:
        metrics.insert(0, f"총 {len(rows)}개 항목 조회")

    # 수치형 컬럼이 있으면 핵심수치에 합계 정보를 보강합니다.
    if columns and rows and len(metrics) < 3:
        numeric_metric = None
        for idx, col in enumerate(columns):
            nums = []
            for row in rows:
                if idx >= len(row):
                    continue
                val = _to_number(row[idx])
                if val is not None:
                    nums.append(val)
            if nums:
                numeric_metric = f"{col} 합계 {sum(nums):,.0f}"
                break
        if numeric_metric and numeric_metric not in metrics:
            metrics.append(numeric_metric)

    lines = [line.strip() for line in (base_answer or "").splitlines() if line.strip()]
    detail_lines = [line for line in lines if line.startswith("- ")]
    if detail_lines and len(metrics) < 3:
        metrics.append(f"상세 품목 미리보기 {len(detail_lines)}개")

    while len(metrics) < 3:
        metrics.append("추가 수치 없음")

    evidence_cols = ", ".join(columns[:4]) if columns else "주요 컬럼 정보 없음"

    result_lines = [
        f"요약: {summary_line}",
        "핵심수치:",
    ]
    for m in metrics[:3]:
        result_lines.append(f"- {m}")

    result_lines.append(f"근거: {filename} 기준, {evidence_cols} 컬럼")

    if detail_lines:
        result_lines.append("상세내역(일부):")
        for item in detail_lines[:5]:
            result_lines.append(item)
        tail = next((line for line in lines if line.startswith("외 ")), None)
        if tail:
            result_lines.append(tail)
    elif structured_query:
        result_lines.append("상세내역(일부):")
        result_lines.append("- 상세 항목은 조회 결과 표를 확인하세요.")

    result_lines.append(f"권장액션: {_default_action_by_label(label)}")
    return "\n".join(result_lines)


def _deterministic_answer(query: str, label: str, table_data: dict | None) -> str | None:
    """
    특정 질의(개수/총합)는 LLM 대신 규칙 기반으로 답변하여 안정성을 높입니다.
    """
    if not table_data or not table_data.get("rows"):
        return None

    # 하드코딩 분기를 늘리지 않기 위해 구조화 엔진을 먼저 사용합니다.
    structured = _compute_structured_answer(query, label, table_data)
    if structured:
        return structured

    # 표 연산 레지스트리(백엔드 집계·필터). LLM보다 우선합니다.
    table_ops = _compute_table_operations_answer(query, label, table_data)
    if table_ops:
        return table_ops

    q = query.replace(" ", "")
    columns = table_data.get("columns", [])
    rows = table_data.get("rows", [])
    summary = table_data.get("summary", "")

    # 재무: "이번달 순이익이 얼마야?" / "손익 얼마?" 등
    if label == "재무" and ("순이익" in q or "손익" in q) and ("얼마" in q or "몇" in q or "합계" in q):
        col_idx = next((i for i, c in enumerate(columns) if "손익" in str(c) or "순이익" in str(c)), None)
        if col_idx is not None:
            nums = []
            for row in rows:
                if col_idx < len(row):
                    v = _to_number(row[col_idx])
                    if v is not None:
                        nums.append(v)
            if nums:
                total = sum(nums)
                time_part = _time_window_text(query) or "조회 기간"
                return f"{time_part} 순이익(손익)은 {total:,.0f}원입니다."
        # summary에 "손익 합계: xxx"가 있으면 파싱해서 사용
        m = re.search(r"손익\s*합계\s*:\s*([-\d,.]+)", summary)
        if m:
            try:
                val = float(m.group(1).replace(",", ""))
                time_part = _time_window_text(query) or "조회 기간"
                return f"{time_part} 순이익(손익)은 {val:,.0f}원입니다."
            except ValueError:
                pass

    asks_total = any(k in q for k in ["총수량", "전체수량", "수량", "물량", "입고량", "출고량", "합계", "총합"])
    asks_count = any(k in q for k in ["개수", "몇개", "몇 개", "항목수", "건수", "갯수", "카운트"])

    if asks_total:
        candidate_cols = [
            i for i, c in enumerate(columns)
            if any(k in str(c) for k in ["수량", "개수", "입고", "출고", "재고", "물량"])
        ]
        best_sum = None
        best_name = None
        for idx in candidate_cols:
            nums = []
            for row in rows:
                if idx >= len(row):
                    continue
                v = _to_number(row[idx])
                if v is not None:
                    nums.append(v)
            if nums:
                s = sum(nums)
                if best_sum is None or s > best_sum:
                    best_sum = s
                    best_name = columns[idx]
        if best_sum is not None:
            msg = f"{summary}\n'{best_name}' 기준 합계는 {best_sum:,.0f}입니다."
            if asks_count:
                msg += f"\n조회된 항목 개수는 총 {len(rows)}개입니다."
            return msg

    if asks_count:
        return f"{summary}\n조회된 항목 개수는 총 {len(rows)}개입니다."

    asks_incoming_items = (
        label == "재고"
        and "입고" in q
        and any(k in q for k in ["예정", "품목", "목록", "리스트", "무엇", "뭐"])
    )
    if asks_incoming_items:
        name_idx = None
        date_idx = None
        for i, col in enumerate(columns):
            c = str(col)
            if name_idx is None and any(k in c for k in ["품목", "제품", "자재", "코드", "명칭", "이름"]):
                name_idx = i
            if date_idx is None and any(k in c for k in ["입고일", "예정일", "일자", "날짜", "date"]):
                date_idx = i

        window_text = _time_window_text(query)
        title = f"{window_text} 입고 예정 품목:" if window_text else "입고 예정 품목:"
        lines = [summary, title]
        preview = rows[:12]
        for row in preview:
            name = row[name_idx] if name_idx is not None and name_idx < len(row) else row[0]
            if date_idx is not None and date_idx < len(row):
                lines.append(f"- {name} ({row[date_idx]})")
            else:
                lines.append(f"- {name}")
        if len(rows) > len(preview):
            lines.append(f"외 {len(rows) - len(preview)}개")
        return "\n".join(lines)

    if label == "재고" and "입고" in q and asks_count:
        return f"{summary}\n입고 예정 품목 개수는 총 {len(rows)}개입니다."
    return None


def _extract_focus_tokens(query: str) -> list[str]:
    """
    질문에서 품목/코드/핵심 키워드 후보를 추출합니다.
    후속 질문의 토큰이 너무 빈약할 때 직전 질문을 보강하기 위한 용도입니다.
    """
    stopwords = {
        "얼마야",
        "알려줘",
        "보여줘",
        "확인",
        "어떻게",
        "얼마나",
        "오늘",
        "현재",
        "전체",
        "데이터",
        "조회",
        "그거",
        "이거",
        "저거",
        "방금",
        "아까",
        "이전",
        "지난",
        "결과",
        "상태",
        "현황",
    }
    tokens = re.findall(r"[A-Za-z0-9가-힣][\w가-힣x\-\.]*", query)
    return [t for t in tokens if t not in stopwords and len(t) >= 2]


def load_learning_stats():
    dataset_path = "clarified_training_dataset.jsonl"
    log_path = "router_logs.jsonl"

    stats = {"total_corrections": 0, "learned_examples": 0, "label_counts": {}}

    if os.path.exists(log_path):
        with open(log_path, "r", encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                if row.get("note") == "사용자 수동 교정":
                    stats["total_corrections"] += 1

    if os.path.exists(dataset_path):
        with open(dataset_path, "r", encoding="utf-8") as f:
            for line in f:
                item = json.loads(line)
                stats["learned_examples"] += 1
                label = _normalize_label_name(item.get("label", "알 수 없음"))
                stats["label_counts"][label] = stats["label_counts"].get(label, 0) + 1

    return stats


def load_learning_quality_stats():
    log_path = "router_logs.jsonl"
    dataset_path = "clarified_training_dataset.jsonl"

    total_questions = 0
    correction_count = 0
    clarified_count = 0
    blocked_count = 0
    top1_scores = []
    final_labels = []

    if os.path.exists(log_path):
        with open(log_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except Exception:
                    continue

                total_questions += 1
                note = str(row.get("note", ""))
                final_label = _normalize_label_name(str(row.get("final_label", "")).strip())
                if final_label:
                    final_labels.append(final_label)
                if note == "사용자 수동 교정":
                    correction_count += 1
                if note == "사용자 선택 명확화":
                    clarified_count += 1

                try:
                    top1_scores.append(float(row.get("top1_score", 0.0)))
                except Exception:
                    pass

    learned_examples = 0
    if os.path.exists(dataset_path):
        with open(dataset_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    learned_examples += 1

    avg_confidence = (sum(top1_scores) / len(top1_scores)) if top1_scores else 0.0
    correction_rate = (correction_count / total_questions) if total_questions else 0.0
    reflection_rate = (learned_examples / correction_count) if correction_count else 1.0
    label_counter = Counter(final_labels[-200:])

    return {
        "total_questions": total_questions,
        "correction_count": correction_count,
        "clarified_count": clarified_count,
        "blocked_count": blocked_count,
        "learned_examples": learned_examples,
        "avg_confidence": round(avg_confidence, 4),
        "correction_rate": round(correction_rate, 4),
        "reflection_rate": round(reflection_rate, 4),
        "label_distribution_recent": dict(label_counter),
    }


def _normalize_learning_text(text: str) -> str:
    return re.sub(r"\s+", "", (text or "").strip().lower())


def _extract_time_filter_note(summary: str | None) -> str | None:
    text = (summary or "").strip()
    if not text.startswith("기간 필터:"):
        return None
    parts = [p.strip() for p in text.split("|") if p.strip()]
    return parts[0] if parts else None


def _build_data_evidence(table_data: dict | None) -> dict | None:
    if not table_data:
        return None
    columns = table_data.get("columns") or []
    rows = table_data.get("rows") or []
    summary = str(table_data.get("summary") or "")
    return {
        "filename": table_data.get("filename"),
        "row_count": len(rows),
        "column_count": len(columns),
        "columns": columns[:12],
        "time_filter_note": _extract_time_filter_note(summary),
        "time_filter_blocked": bool(table_data.get("time_filter_blocked")),
        "summary": summary,
    }


def _append_query_trace(row: dict):
    try:
        with open("query_trace.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    except Exception:
        pass


def _normalize_label_name(label: str | None) -> str:
    raw = (label or "").strip()
    return "기타" if raw == "회사기타" else raw


def _is_correction_learned(query: str, label: str) -> tuple[bool, str | None]:
    dataset_path = "clarified_training_dataset.jsonl"
    if not os.path.exists(dataset_path):
        return False, None

    target_query = _normalize_learning_text(query)
    target_label = _normalize_label_name(label)

    try:
        with open(dataset_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                row_query = _normalize_learning_text(str(row.get("text", "")))
                row_label = _normalize_label_name(str(row.get("label", "")).strip())
                if row_query == target_query and row_label == target_label:
                    return True, row_label
    except Exception:
        return False, None
    return False, None


def _get_learned_label_for_query(query: str) -> str | None:
    dataset_path = "clarified_training_dataset.jsonl"
    if not os.path.exists(dataset_path):
        return None

    target_query = _normalize_learning_text(query)
    try:
        with open(dataset_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                row_query = _normalize_learning_text(str(row.get("text", "")))
                if row_query == target_query:
                    label = _normalize_label_name(str(row.get("label", "")).strip())
                    if label:
                        return label
    except Exception:
        return None
    return None


@chatbot_bp.post("/ask")
@login_required
def ask():
    data = request.json or {}
    query = (data.get("query") or "").strip()
    clarified_label = (data.get("clarified_label") or "").strip() or None
    if not query:
        return jsonify({"error": "query는 필수입니다."}), 400

    # 공지사항 질문은 게시판 없이 안내 메시지만 반환합니다.
    if _is_notice_query(query):
        employee_id = session.get("employee_id")
        answer = _notice_board_answer(query)
        result = {
            "status": "resolved",
            "final_label": "기타",
            "execution_target": None,
            "score": 1.0,
            "answer": answer,
        }
        set_context(employee_id, query, "기타", answer)
        log_action(
            employee_id=employee_id,
            action_type="chatbot.ask",
            target_type="query",
            payload={
                "query": query,
                "effective_query": query,
                "final_label": "기타",
                "status": "resolved",
                "notice_fast_path": True,
            },
        )
        return jsonify(res=result)

    # 분류 없이 조회만 하므로 clarified_label 사용하지 않음 (항상 전체 조회)
    clarified_label = None
    learned_label_applied = False
    used_context_label = None

    employee_id = session.get("employee_id")
    context = get_context(employee_id)
    is_followup = _is_context_followup(query)

    # 후속 질문에서 품목/핵심 토큰이 비어 있으면 직전 질문 맥락을 보강해 조회 정확도를 높입니다.
    query_tokens = _extract_focus_tokens(query)
    context_expanded = False
    effective_query = query
    has_explicit_time = _has_explicit_time_window(query)
    if is_followup and context and not query_tokens and context.last_query and not has_explicit_time:
        effective_query = f"{context.last_query} {query}"
        context_expanded = True

    result = _process_query(employee_id, effective_query, clarified_label=clarified_label)

    # 분류 선택 없이 진행: needs_clarification이어도 resolved로 처리해 바로 조회·답변
    was_low_confidence = result.get("status") == "needs_clarification"
    if was_low_confidence:
        result["status"] = "resolved"
        result["final_label"] = "기타"

    label = result.get("final_label")
    table_data = None
    data_evidence = None
    if result.get("status") == "resolved":
        _, query_data_fn, _, _ = _load_data_reader()
        table_data = query_data_fn(None, effective_query)
        result["data"] = table_data
        if table_data and table_data.get("label"):
            result["final_label"] = table_data["label"]
            label = table_data["label"]
        data_evidence = _build_data_evidence(table_data)
        _append_query_trace(
            {
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "query": query,
                "effective_query": effective_query,
                "final_label": label,
                "status": result.get("status"),
                "context_query_expanded": context_expanded,
                "data_evidence": data_evidence,
            }
        )

    if result.get("status") == "resolved":
        try:
            # 조회 결과가 있을 때는 표시(저신뢰도 라우팅이어도 데이터가 있으면 막지 않음)
            rows = (table_data or {}).get("rows") if table_data else None
            has_data = bool(rows and isinstance(rows, (list, tuple)) and len(rows) > 0)
            # 이해하지 못한 경우: 조회 결과가 없을 때만 안내 (was_low_confidence 단독으로는 답변 덮어쓰지 않음)
            if not has_data and not _is_attendance_query(query) and not _is_notice_query(query):
                result["answer"] = (
                    "질문 의도를 정확히 파악하기 어렵습니다. 아래처럼 조금 더 구체적으로 적어 주시면 도움을 드리기 쉬워요.\n\n"
                    "· 재무: 이번달 매출 얼마야? / 이번달 손익 / 예산 대비 비용\n"
                    "· 재고: 재고 현황 알려줘 / 품목별 재고 수량\n"
                    "· 생산: 이번주 생산량 / 가동률·불량 현황\n"
                    "· 규정/근태: 근태 규정 / 연차 신청 방법\n"
                    "· 기타: 담당 부서 문의 등"
                )
            elif table_data and table_data.get("rows") and not table_data.get("time_filter_blocked"):
                result["answer"] = "관련 데이터를 가져왔습니다. 아래 조회 데이터 목록을 확인해 주세요."
            else:
                # 근태 질문은 LLM 생성 대신 정책 안내를 고정 응답으로 반환합니다.
                if _is_attendance_query(query):
                    answer = _attendance_policy_answer(label or "기타")
                elif table_data and table_data.get("time_filter_blocked"):
                    answer = table_data.get("summary")
                else:
                    answer = _deterministic_answer(query, label, table_data)
                    if not answer:
                        answer = _generate_answer(effective_query, label, table_data)
                if _is_garbled_answer_text(answer):
                    if table_data:
                        answer = (
                            f"{table_data.get('summary', '조회 결과가 있습니다.')}\n"
                            "답변 생성 품질이 불안정해 요약 결과만 표시했습니다."
                        )
                    else:
                        answer = (
                            "질문을 정확히 이해하지 못했을 수 있습니다. "
                            "예: '이번달 매출', '재고 현황', '근태 규정'처럼 구체적으로 다시 적어 주시겠어요?"
                        )
                result["answer"] = answer
        except Exception:
            result["answer"] = None

        # learned_label 자동적용은 사용자 명시 선택과 구분합니다.
        if learned_label_applied:
            result["clarified_label_applied"] = False
            result["learned_label_applied"] = True

        set_context(employee_id, query, label, result.get("answer") or "")
        if used_context_label:
            result["context_applied"] = True
            result["context_label"] = used_context_label
        if context_expanded:
            result["context_query_expanded"] = True

        # 명확화 선택으로 확정된 라벨도 즉시 학습 데이터에 반영
        if result.get("clarified_label_applied"):
            run_extraction()

    log_action(
        employee_id=employee_id,
        action_type="chatbot.ask",
        target_type="query",
        payload={
            "query": query,
            "effective_query": effective_query,
            "final_label": result.get("final_label"),
            "status": result.get("status"),
            "clarified_label": clarified_label,
            "used_context_label": used_context_label,
            "context_query_expanded": context_expanded,
            "data_evidence": data_evidence,
        },
    )
    return jsonify(res=result)


FEEDBACK_LOGS_PATH = "feedback_logs.jsonl"


def _append_feedback_log(employee_id, query: str, answer: str | None, rating: str, reason: str | None = None):
    row = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "employee_id": employee_id,
        "query": query,
        "answer_snippet": (answer[:500] if answer else None),
        "rating": rating,
    }
    if reason and reason.strip():
        row["reason"] = reason.strip()[:500]
    with open(FEEDBACK_LOGS_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _remove_last_feedback_log(employee_id, query: str, rating: str) -> bool:
    """해당 사용자·질문·평가에 대한 가장 최근 로그 한 줄 제거. 제거했으면 True."""
    if not os.path.isfile(FEEDBACK_LOGS_PATH):
        return False
    with open(FEEDBACK_LOGS_PATH, "r", encoding="utf-8") as f:
        lines = [ln.strip() for ln in f.readlines() if ln.strip()]
    rows = []
    for ln in lines:
        try:
            rows.append(json.loads(ln))
        except json.JSONDecodeError:
            continue
    # 마지막에서부터 첫 번째로 일치하는 항목 제거
    for i in range(len(rows) - 1, -1, -1):
        r = rows[i]
        if r.get("employee_id") == employee_id and r.get("query") == query and r.get("rating") == rating:
            rows.pop(i)
            break
    else:
        return False
    with open(FEEDBACK_LOGS_PATH, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return True


@chatbot_bp.post("/chat-feedback")
@login_required
def chat_feedback():
    data = request.json or {}
    query = (data.get("query") or "").strip()
    answer = (data.get("answer") or "").strip() or None
    rating = (data.get("rating") or "").strip().lower()
    cancel_rating = (data.get("cancel_rating") or "").strip().lower() or None
    employee_id = session.get("employee_id")

    if rating == "cancel":
        if cancel_rating not in ("like", "dislike"):
            return jsonify({"error": "취소 시 cancel_rating은 like 또는 dislike입니다."}), 400
        removed = _remove_last_feedback_log(employee_id, query, cancel_rating)
        return jsonify(status="success", rating="cancel", removed=removed)

    if rating not in ("like", "dislike"):
        return jsonify({"error": "rating은 like, dislike 또는 cancel입니다."}), 400
    reason = (data.get("reason") or "").strip() or None if rating == "dislike" else None
    _append_feedback_log(employee_id, query, answer, rating, reason=reason)
    return jsonify(status="success", rating=rating)


@chatbot_bp.post("/correct")
@login_required
def correct():
    data = request.json or {}
    query = (data.get("query") or "").strip()
    correct_label = (data.get("correct_label") or "").strip()
    if not query or not correct_label:
        return jsonify({"error": "query와 correct_label은 필수입니다."}), 400

    dummy_result = RoutingResult(
        probs={},
        top1_label=correct_label,
        top1_score=1.0,
        top2_label="None",
        top2_score=0.0,
        route_status="resolved",
        clarification_question=None,
    )

    employee_id = session.get("employee_id")
    _save_router_log(employee_id, query, dummy_result, correct_label, "사용자 수동 교정")

    run_extraction()
    stats = load_learning_stats()
    log_action(
        employee_id=employee_id,
        action_type="chatbot.correct",
        target_type="query",
        payload={"query": query, "correct_label": correct_label},
    )
    return jsonify(
        status="success",
        learned_examples=stats["learned_examples"],
        label_counts=stats["label_counts"],
    )


@chatbot_bp.get("/files")
@login_required
def list_scanned_files():
    """현재 스캔된 파일 목록을 반환합니다 (재스캔 없이 현재 인덱스 기준)."""
    get_watch_dir, _, _, get_all_files_columns = _load_data_reader()
    all_entries = get_all_files_columns()
    all_files = [{"filename": e["filename"], "label": e["label"] or "기타"} for e in all_entries]
    return jsonify(
        watch_dir=get_watch_dir(),
        all_files=all_files,
        count=len(all_files),
    )


@chatbot_bp.post("/rescan")
@login_required
def rescan_files():
    get_watch_dir, _, rescan, get_all_files_columns = _load_data_reader()
    index = rescan()
    found = {label: os.path.basename(path) for label, path in index.items()}
    # data 폴더 내 CSV/Excel 전부 스캔 결과 (파일별 filename, label)
    all_entries = get_all_files_columns()
    all_files = [{"filename": e["filename"], "label": e["label"] or "기타"} for e in all_entries]
    log_action(
        employee_id=session.get("employee_id"),
        action_type="data.rescan",
        target_type="data",
        payload={"by_label": found, "all_count": len(all_files)},
    )
    return jsonify(
        status="success",
        watch_dir=get_watch_dir(),
        files=found,
        all_files=all_files,
        count=len(all_files),
    )


@chatbot_bp.get("/stats")
@login_required
def stats():
    return jsonify(load_learning_stats())


@chatbot_bp.get("/stats/quality")
@login_required
def quality_stats():
    return jsonify(load_learning_quality_stats())


@chatbot_bp.post("/learning/verify")
@login_required
def verify_learning():
    data = request.json or {}
    query = (data.get("query") or "").strip()
    label = (data.get("label") or "").strip()
    if not query or not label:
        return jsonify({"error": "query와 label은 필수입니다."}), 400

    learned, matched_label = _is_correction_learned(query, label)
    stats = load_learning_stats()
    return jsonify(
        status="success",
        learned=learned,
        matched_label=matched_label,
        learned_examples=stats.get("learned_examples", 0),
    )


@chatbot_bp.get("/audit")
@login_required
def my_audit():
    employee_id = session.get("employee_id")
    with get_cursor() as cur:
        cur.execute(
            "SELECT * FROM audit_logs WHERE employee_id = ? ORDER BY id DESC LIMIT 100",
            (employee_id,),
        )
        rows = cur.fetchall()
    return jsonify(rows=rows)

