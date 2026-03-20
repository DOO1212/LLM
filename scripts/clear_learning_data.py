"""
학습 데이터·학습한 예시 전부 초기화:
- clarified_training_dataset.jsonl → 비움 (시드·학습 예시 모두 제거)
- router_logs.jsonl → 사용자 교정/명확화 항목만 제거 (나머지 조회 로그는 유지)
"""

import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)

DATASET_PATH = "clarified_training_dataset.jsonl"
LOG_PATH = "router_logs.jsonl"

LEARNING_NOTES = ("사용자 수동 교정", "사용자 선택 명확화")


def clear_training_dataset():
    """clarified_training_dataset.jsonl을 비운다. 학습한 예시·시드 모두 제거."""
    with open(DATASET_PATH, "w", encoding="utf-8") as f:
        pass
    print(f"✅ {DATASET_PATH} → 비움 (학습 예시 0건)")


def remove_learning_entries_from_router_logs():
    """router_logs.jsonl에서 학습에 쓰이던 교정/명확화 항목만 제거한다."""
    if not os.path.exists(LOG_PATH):
        print(f"⚠ {LOG_PATH} 없음, 건너뜀")
        return 0
    kept = []
    removed = 0
    with open(LOG_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if row.get("note") in LEARNING_NOTES:
                removed += 1
                continue
            kept.append(line)
    with open(LOG_PATH, "w", encoding="utf-8") as f:
        for line in kept:
            f.write(line + "\n")
    print(f"✅ {LOG_PATH} 에서 학습용 교정/명확화 {removed}건 제거, {len(kept)}건 유지")
    return removed


def main():
    print("학습 데이터·학습한 예시 전부 초기화 중...")
    clear_training_dataset()
    remove_learning_entries_from_router_logs()
    print("완료. 서버 재시작 시 반영됩니다.")


if __name__ == "__main__":
    main()
