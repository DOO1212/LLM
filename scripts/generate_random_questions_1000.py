import csv
import json
import random
from pathlib import Path


SEED = 20260313
TARGET_COUNTS = {
    "재고": 180,
    "생산": 170,
    "재무": 170,
    "규율": 160,
    "기타": 160,
}


def unique_texts(items):
    seen = set()
    out = []
    for text in items:
        t = " ".join(str(text).split())
        if not t or t in seen:
            continue
        seen.add(t)
        out.append(t)
    return out


def candidates_inventory():
    times = ["오늘", "이번 주", "이번 달", "금일", "주간", "월간"]
    actions = ["현황", "수량", "리스트", "요약", "집계", "상세"]
    items = ["볼트 M4x10", "볼트 M5", "너트 M4", "와셔 8mm", "모터 A라인", "센서 B형", "비상정지버튼"]
    stock_keys = ["재고", "안전재고", "입고 예정", "출고 예정", "반품 재고", "불용 재고", "재고 평가금액"]
    templates = [
        "{time} {item} {stock} {action} 알려줘",
        "{item} {stock} 지금 얼마나 있어?",
        "{time} {stock} 기준으로 {item} 데이터 보여줘",
        "{item} 최소발주량 총 금액 계산해줘",
        "{stock} 중에서 {item} 관련 항목만 뽑아줘",
        "{time} 기준 {item} 재고 변동 추이 알려줘",
    ]
    out = []
    for tpl in templates:
        for time in times:
            for item in items:
                for stock in stock_keys:
                    for action in actions:
                        out.append(tpl.format(time=time, item=item, stock=stock, action=action))
    return unique_texts(out)


def candidates_production():
    lines = ["1라인", "2라인", "3라인", "A라인", "B라인", "C라인"]
    metrics = ["생산 실적", "가동률", "불량률", "목표 달성률", "비가동 시간", "작업 지연"]
    times = ["오늘", "어제", "이번 주", "이번 달", "주간", "월간"]
    reasons = ["설비 점검", "자재 부족", "인력 공백", "품질 이슈", "계획 변경"]
    templates = [
        "{time} {line} {metric} 보여줘",
        "{line} {metric} 왜 낮은지 분석해줘",
        "{time} 생산라인별 {metric} 비교해줘",
        "{line} 비가동 사유가 {reason} 맞아?",
        "{line} 기준으로 {time} 생산 요약 만들어줘",
    ]
    out = []
    for tpl in templates:
        for time in times:
            for line in lines:
                for metric in metrics:
                    for reason in reasons:
                        out.append(tpl.format(time=time, line=line, metric=metric, reason=reason))
    return unique_texts(out)


def candidates_finance():
    times = ["오늘", "이번 주", "이번 달", "지난 달", "분기", "연간"]
    metrics = ["매출", "비용", "손익", "예산 집행률", "미수금", "원가"]
    depts = ["영업", "생산", "구매", "관리", "품질", "물류"]
    actions = ["현황", "요약", "집계", "상세", "비교", "추이"]
    templates = [
        "{time} {dept} {metric} {action} 알려줘",
        "{dept} {metric}가 지난 달 대비 얼마나 변했어?",
        "{time} 기준 {metric} 리포트 만들어줘",
        "{dept} 예산 대비 {metric} 차이 분석해줘",
        "{metric} 기준 상위 5개 항목 보여줘",
    ]
    out = []
    for tpl in templates:
        for time in times:
            for dept in depts:
                for metric in metrics:
                    for action in actions:
                        out.append(tpl.format(time=time, dept=dept, metric=metric, action=action))
    return unique_texts(out)


def candidates_policy():
    topics = ["연차", "재택근무", "출장", "보안", "복장", "근태", "퇴사", "입사"]
    actions = ["규정 알려줘", "신청 절차 알려줘", "승인 기준 알려줘", "주의사항 요약해줘", "양식 위치 알려줘"]
    roles = ["사원", "대리", "과장", "팀장", "신입"]
    times = ["오늘", "이번 주", "이번 달", "현재"]
    templates = [
        "{topic} {action}",
        "{role} 기준 {topic} {action}",
        "{times} 적용되는 {topic} 규정 변경사항 있어?",
        "{topic} 위반 시 조치 기준 알려줘",
    ]
    out = []
    for tpl in templates:
        for topic in topics:
            for action in actions:
                for role in roles:
                    for times in times:
                        out.append(
                            tpl.format(topic=topic, action=action, role=role, times=times)
                        )
    return unique_texts(out)


def candidates_company_misc():
    systems = ["사내 포털", "전자결재", "게시판", "공지 시스템", "VPN", "메일"]
    actions = ["접속이 안돼", "사용법 알려줘", "오류 해결 방법 알려줘", "권한 신청 방법 알려줘", "공지 확인 위치 알려줘"]
    depts = ["인사팀", "총무팀", "전산팀", "영업팀", "생산팀"]
    channels = ["PC", "모바일", "원격", "사내망", "외부망"]
    times = ["오늘", "방금", "이번 주", "이번 달", "지금"]
    issues = ["로그인 실패", "권한 없음", "페이지 오류", "알림 미수신", "첨부 업로드 실패"]
    targets = ["공지", "문서 양식", "부서 연락처", "업무 메뉴", "사용 매뉴얼"]
    templates = [
        "{system} {action}",
        "{dept} 공지 어디서 확인해?",
        "{system}에서 부서 공지 올리는 방법 알려줘",
        "{dept} 담당자 연락처를 포털에서 찾을 수 있어?",
        "회사 문서 양식은 어디서 내려받아?",
        "{times} {channels}에서 {system} {issues} 해결 방법 알려줘",
        "{dept} 관련 {targets}는 {system}에서 어디에 있어?",
        "{system} {channels} 버전에서 {targets} 찾는 경로 알려줘",
        "{times} 기준 {system} 점검 공지 확인 방법 알려줘",
    ]
    out = []
    for tpl in templates:
        for system in systems:
            for action in actions:
                for dept in depts:
                    for channel in channels:
                        for time in times:
                            for issue in issues:
                                for target in targets:
                                    out.append(
                                        tpl.format(
                                            system=system,
                                            action=action,
                                            dept=dept,
                                            channels=channel,
                                            times=time,
                                            issues=issue,
                                            targets=target,
                                        )
                                    )
    return unique_texts(out)


def candidates_general():
    moods = ["좋은 아침", "안녕", "고마워", "수고했어", "오늘도 화이팅", "힘내자", "반가워", "잘 지내?"]
    requests = [
        "재미있는 이야기 해줘",
        "짧은 농담 해줘",
        "격려 한마디 해줘",
        "동기부여 문장 줘",
        "기분 좋아지는 말 해줘",
        "짧은 명언 알려줘",
        "스트레스 풀 문장 하나 줘",
        "웃긴 멘트 하나 해줘",
    ]
    topics = ["날씨", "점심 메뉴", "커피 추천", "운동 루틴", "주말 계획", "퇴근 후 취미", "아침 루틴", "집중 방법"]
    tones = ["친근하게", "짧게", "한 줄로", "재밌게", "가볍게", "밝게"]
    personas = ["친구처럼", "코치처럼", "선배처럼", "동료처럼"]
    times = ["오늘", "지금", "이번 주", "요즘"]
    templates = [
        "{mood}",
        "{topic} 추천해줘",
        "{requests}",
        "오늘 {topic} 어떨까?",
        "{mood}, {requests}",
        "{times} 기준으로 {topic} {tones} 알려줘",
        "{personas} {requests}",
        "{topic} 관련해서 {tones} 조언해줘",
        "{times} 컨디션 올리는 팁 {tones} 말해줘",
    ]
    out = []
    for tpl in templates:
        for mood in moods:
            for req in requests:
                for topic in topics:
                    for tone in tones:
                        for persona in personas:
                            for time in times:
                                out.append(
                                    tpl.format(
                                        mood=mood,
                                        requests=req,
                                        topic=topic,
                                        tones=tone,
                                        personas=persona,
                                        times=time,
                                    )
                                )
    return unique_texts(out)


def build_pool():
    return {
        "재고": candidates_inventory(),
        "생산": candidates_production(),
        "재무": candidates_finance(),
        "규율": candidates_policy(),
        "기타": candidates_company_misc(),
    }


def main():
    rng = random.Random(SEED)
    pools = build_pool()
    rows = []

    for label, count in TARGET_COUNTS.items():
        candidates = pools.get(label, [])
        if len(candidates) < count:
            raise ValueError(f"{label} 후보가 부족합니다: {len(candidates)} < {count}")
        rng.shuffle(candidates)
        picked = candidates[:count]
        rows.extend({"question": q, "label": label} for q in picked)

    # 전체 중복 방지 확인
    seen = set()
    deduped = []
    for row in rows:
        q = row["question"]
        if q in seen:
            continue
        seen.add(q)
        deduped.append(row)

    if len(deduped) != 1000:
        raise ValueError(f"최종 질문 수가 1000이 아닙니다: {len(deduped)}")

    rng.shuffle(deduped)

    out_dir = Path("data")
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_path = out_dir / "random_questions_1000.csv"
    with csv_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["question", "label"])
        writer.writeheader()
        writer.writerows(deduped)

    jsonl_path = out_dir / "random_questions_1000.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as f:
        for row in deduped:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    try:
        from openpyxl import Workbook

        xlsx_path = out_dir / "random_questions_1000.xlsx"
        wb = Workbook()
        ws = wb.active
        ws.title = "questions"
        ws.append(["question", "label"])
        for row in deduped:
            ws.append([row["question"], row["label"]])
        wb.save(xlsx_path)
    except Exception:
        xlsx_path = None

    print(f"생성 완료: {len(deduped)}건")
    print(f"- {csv_path}")
    print(f"- {jsonl_path}")
    if xlsx_path:
        print(f"- {xlsx_path}")


if __name__ == "__main__":
    main()
