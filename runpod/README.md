# RunPod에서 성능·정확도 벤치마크

로컬 VRAM/속도 한계를 피하고 **GPU 팟**에서 같은 스크립트로 재현할 때 사용합니다.

## 1. 팟 준비

- **GPU 템플릿**: CUDA 있는 이미지(예: PyTorch 공식, 또는 Ubuntu + NVIDIA 드라이버).
- **디스크**: HF 모델 캐시 + 임베딩 캐시 고려해 **30GB 이상** 권장.
- **네트워크**: Hugging Face에서 `KURE-v1`, 답변용 `Qwen2.5-7B-Instruct`(기본) 등을 받아야 함. 더 작은 GPU는 `ANSWER_LLM_MODEL=Qwen/Qwen2.5-3B-Instruct`.

## 2. 코드 올리기

```bash
git clone <이-저장소-URL> chatbot
cd chatbot
```

(또는 RunPod에 압축 업로드 후 풀기.)

## 3. (선택) Ollama — `run_llm_eval` 50문항용

`run_llm_eval.py`는 **Ollama HTTP API**만 사용합니다. 팟에 Ollama를 깔면 로컬과 동일 벤치를 맞출 수 있습니다.

```bash
curl -fsSL https://ollama.com/install.sh | sh
ollama serve   # 별 터미널 또는 nohup
ollama pull qwen2.5:7b
export OLLAMA_HOST=http://127.0.0.1:11434
```

Ollama가 없으면 스크립트는 **Ollama 구간만 건너뜁니다.**

## 4. 벤치 실행

```bash
chmod +x runpod/run_benchmarks.sh
./runpod/run_benchmarks.sh
```

결과 로그: **`runpod/benchmark_output.txt`**

### 환경 변수 (선택)

| 변수 | 설명 |
|------|------|
| `EMBEDDING_MODEL` | 기본 `nlpai-lab/KURE-v1`, 예: `Qwen/Qwen3-Embedding-8B` |
| `OLLAMA_MODEL` | 기본 `qwen2.5:7b` |
| `OLLAMA_HOST` | 기본 `http://127.0.0.1:11434` |
| `HF_TOKEN` | Hub rate limit 완화 |

## 5. 무엇이 도는지

1. `nvidia-smi` (GPU 이름·메모리)
2. `pytest tests/test_table_operations.py` — 백엔드 표 연산
3. `run_rag_pipeline_eval.py` — **3단계 RAG 50문항**, `--llm local` (`answer_llm`, 기본 7B, GPU)
4. Ollama 있으면: `run_llm_eval.py` — **전체 CSV + LLM 50문항**

로컬과 **숫자를 직접 비교**하려면 같은 브랜치·같은 `llm_eval_questions_50.json`을 쓰면 됩니다.

## 6. (선택) Docker로 웹 서버만 올리기

저장소 루트 `Dockerfile`로 빌드하면 Flask 앱이 `0.0.0.0`에 바인딩됩니다.

- **포트**: RunPod가 넣어 주는 `PORT`를 읽습니다(없으면 `8000`).
- **디버그**: 이미지 기본값은 `FLASK_DEBUG=0`입니다.

```bash
docker build -t corpdesk-chatbot .
docker run --rm -p 8000:8000 corpdesk-chatbot
```
