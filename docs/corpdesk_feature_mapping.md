# corpdesk 핵심 기능 매핑 (Flask 단계 통합)

## 통합 범위

- 인증/권한: 로그인, 역할 기반 접근 제어(RBAC)
- 조직/직원: 부서, 직급, 직원 정보
- 전자결재: 기안, 결재선, 승인/반려, 상태 추적
- 게시판: 공지/부서 게시글 CRUD
- 챗봇 연동: 질문 결과를 결재 초안으로 변환
- 운영 기능 2차: (제외) 휴가, 일정, 알림

## corpdesk 모듈 -> 현재 프로젝트 매핑

- `employee`, `organization`, `position` -> `app/routes/org.py`, `app/db.py`
- `approval` -> `app/routes/approval.py`, `app/services/approval_service.py`
- `board` -> `app/routes/board.py`
- `chat`(기존 업무 챗봇) -> `app/routes/chatbot.py`
- `admin`, `stats` -> 초기에는 감사로그/운영 API로 대체

## 우선순위

1. 공통 기반 (인증/RBAC, DB, 감사로그)
2. 결재/게시판/조직
3. 챗봇 -> 결재 초안 연계
4. (보류) 휴가/일정/알림
5. 배포/운영 자동화

