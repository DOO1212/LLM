"""
data/ 폴더의 CSV 파일을 UTF-8 BOM으로 재저장합니다.
Excel에서 한글이 깨지지 않게 됩니다.

실행: python3 fix_csv_encoding.py
"""
import os
import glob

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")

def fix(path: str):
    # 현재 인코딩 자동 감지 후 UTF-8 BOM으로 재저장
    for enc in ("utf-8-sig", "utf-8", "cp949", "euc-kr"):
        try:
            content = open(path, "r", encoding=enc).read()
            # BOM이 이미 있으면 건너뜀
            if enc == "utf-8-sig" and content.startswith("\ufeff"):
                print(f"[이미 BOM 적용됨] {os.path.basename(path)}")
                return
            break
        except UnicodeDecodeError:
            continue
    else:
        print(f"[실패] 인코딩 감지 불가: {path}")
        return

    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        f.write(content)
    print(f"[완료] {os.path.basename(path)} → UTF-8 BOM 저장")

if __name__ == "__main__":
    csv_files = glob.glob(os.path.join(DATA_DIR, "*.csv"))
    if not csv_files:
        print("data/ 폴더에 CSV 파일이 없습니다.")
    for path in csv_files:
        fix(path)
    print("\n모든 파일 변환 완료! Excel에서 다시 열어보세요.")
