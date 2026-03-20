import os
import shutil
from datetime import datetime


BACKUP_ROOT = os.environ.get("BACKUP_DIR", "backups")
FILES_TO_BACKUP = [
    "corpdesk.db",
    "router_logs.jsonl",
    "clarified_training_dataset.jsonl",
]


def main():
    os.makedirs(BACKUP_ROOT, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    target_dir = os.path.join(BACKUP_ROOT, ts)
    os.makedirs(target_dir, exist_ok=True)

    for name in FILES_TO_BACKUP:
        src_candidates = [name, os.path.join("data", name)]
        copied = False
        for src in src_candidates:
            if os.path.exists(src):
                shutil.copy2(src, os.path.join(target_dir, os.path.basename(src)))
                copied = True
                break
        if not copied:
            print(f"[SKIP] not found: {name}")
        else:
            print(f"[OK] backed up: {name}")

    print(f"Backup completed: {target_dir}")


if __name__ == "__main__":
    main()

