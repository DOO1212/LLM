# 운영 가이드

## 1) 로컬 실행

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python server.py
```

### 임베딩 모델 (선택)

기본은 `nlpai-lab/KURE-v1`입니다. GPU가 있고 VRAM이 충분하면 **Qwen3-Embedding-8B** 등으로 바꿀 수 있습니다.

```bash
export EMBEDDING_MODEL=Qwen/Qwen3-Embedding-8B
```

- FP16 기준 VRAM **16GB 이상** 권장. 부족하면 KURE-v1 유지.
- (선택) Flash Attention 2: `export USE_FLASH_ATTENTION=1` — `flash-attn` 설치 필요.
- 기본값으로 되돌리기: `unset EMBEDDING_MODEL` 또는 `export EMBEDDING_MODEL=nlpai-lab/KURE-v1`

## 2) Docker 실행

```bash
docker compose up -d --build
```

기본 접속 주소: `http://localhost:8000`

### 접속이 안 될 때

1. **서버가 켜져 있는지 확인**  
   터미널에서 `bash ~/chatbot/status_server.sh` 실행 → `running (pid: ...)` 이면 실행 중.
2. **서버가 꺼져 있으면**  
   Windows: `start_server.cmd` 더블클릭  
   WSL/터미널: `bash ~/chatbot/start_server.sh` 실행 후 1~2분 기다리기(첫 기동 시 데이터 로드에 시간 걸림).
3. **접속 주소**  
   브라우저에서 **http://localhost:8000** 또는 **http://127.0.0.1:8000** 입력.
4. **WSL에서 서버 켠 경우**  
   Windows 브라우저에서도 `localhost:8000`으로 접속 가능(자동 포워딩). 안 되면 WSL IP 확인: `hostname -I | awk '{print $1}'` 로 나온 IP로 `http://<WSL_IP>:8000` 시도.

## 3) 백업

```bash
python scripts/backup_data.py
```

백업 대상:
- `corpdesk.db` 또는 `data/corpdesk.db`
- `router_logs.jsonl`
- `clarified_training_dataset.jsonl`

## 4) 헬스체크

```bash
python scripts/healthcheck.py
```

정상 시 `healthcheck ok`가 출력됩니다.

## 5) CSV 행 임베딩 사전 계산 (선택)

`data_reader`는 **입출고·수치 필터 적용 후** 기본적으로 **행 임베딩 유사도**로 검색합니다(키워드 행 매칭은 끔).  
예전처럼 키워드로 행을 먼저 거르려면 `USE_KEYWORD_ROW_FILTER=1`.

### 행 임베딩: 전부 가져오기 / 임계값

- 기본: **`ROW_EMBED_SIM_THRESHOLD=all`** (미설정과 동일) → 남은 행 **전부**를 **유사도 높은 순**으로 정렬만 함.
- 예전처럼 잘라내려면: `export ROW_EMBED_SIM_THRESHOLD=0.5` (코사인 0.5 이상만; 없으면 수치·입출고만 적용된 전체 행)
- 최대 개수만: `export ROW_EMBED_TOP_K=30` (`all` 모드에서 상위 30행만)

행 임베딩을 매번 계산하지 않으려면 디스크 캐시를 쓸 수 있습니다.

```bash
# 프로젝트 루트에서 (서버와 동일한 EMBEDDING_MODEL 권장)
python scripts/build_embedding_cache.py --file data/inventory.csv

# data 폴더의 모든 CSV
python scripts/build_embedding_cache.py --all-csv
```

- 캐시 위치: `data/.embedding_cache/` (CSV마다 `.npz` + `.meta.json`)
- CSV를 수정하면 **mtime이 바뀌므로** 캐시가 자동으로 무효화됩니다 → 스크립트를 다시 실행하세요.
- 모델을 바꾼 경우에도 `EMBEDDING_MODEL`이 메타와 다르면 캐시를 쓰지 않습니다.
- 끄기: 환경변수 `USE_ROW_EMBEDDING_CACHE=0`

## 6) 모니터링 권장 지표

- API 5xx 비율
- 평균 응답시간
- `/approval` 문서 생성량/승인 지연시간
- 로그인 실패 횟수
- 디스크 사용량(`data`, `backups`)

