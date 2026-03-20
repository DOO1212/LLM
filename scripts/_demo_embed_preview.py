"""채팅 미리보기용: 모델 없이 CSV만 읽어 임베딩 입력 텍스트 + 해시 기반 데모 벡터."""
import csv
import hashlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def read_csv_rows(path: Path) -> tuple[list[str], list[dict]]:
    for enc in ("utf-8-sig", "utf-8", "cp949", "euc-kr"):
        try:
            with open(path, encoding=enc, newline="") as f:
                r = csv.DictReader(f)
                rows = list(r)
                if rows:
                    return list(rows[0].keys()), rows
        except UnicodeDecodeError:
            continue
    return [], []


def embedding_text_columns(columns: list[str], rows: list[dict]) -> list[str]:
    return [
        c
        for c in columns
        if any(
            not str(row.get(c, "")).replace(".", "").replace("-", "").replace("%", "").isdigit()
            and len(str(row.get(c, ""))) > 1
            for row in rows[:5]
        )
    ]


def demo_vec12(text: str) -> list[float]:
    h = hashlib.sha256(text.encode("utf-8")).digest()
    out = []
    for i in range(12):
        chunk = bytes([h[(i + j) % len(h)] for j in range(8)])
        u = int.from_bytes(chunk, "little") / (2**64)
        out.append(round(u * 2 - 1, 6))
    return out


def main() -> None:
    p = ROOT / "data" / "inventory.csv"
    columns, rows = read_csv_rows(p)
    tc = embedding_text_columns(columns, rows)
    t0 = " ".join(str(rows[0].get(c, "")) for c in tc)
    print("TEXT_COLS:", tc)
    print("ROW0_EMBED_INPUT:", t0[:220] + ("..." if len(t0) > 220 else ""))
    print("DEMO_12D_NOT_REAL_MODEL:", demo_vec12(t0))


if __name__ == "__main__":
    main()
