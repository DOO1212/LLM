"""
EXAONE-3.5-2.4B-Instruct 기반 답변 생성 모듈.

라우팅 결과 + CSV 조회 데이터를 컨텍스트로 주고
사용자 질문에 대한 자연어 답변을 생성합니다.
"""

import torch
import re
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"

# 4-bit 양자화 설정 (~2GB VRAM)
_bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

print(f"[LLM] 답변 모델 로딩 중: {MODEL_ID}")
_tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
_model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    quantization_config=_bnb_config,
    device_map="auto",
)
_model.eval()
print("[LLM] 로딩 완료")


def _format_data_as_text(data: dict) -> str:
    """CSV 조회 결과를 LLM이 읽기 쉬운 텍스트로 변환합니다."""
    if not data or not data.get("rows"):
        return ""

    columns = data["columns"]
    rows    = data["rows"][:40]
    lines   = []
    for row in rows:
        pairs = []
        for col, val in zip(columns, row):
            if val in ("", None):
                continue
            text = str(val).strip()
            if len(text) > 60:
                text = text[:57] + "..."
            pairs.append(f"{col}: {text}")
        lines.append("- " + ", ".join(pairs))

    header = f"[{data.get('label', '')} 데이터 ({data.get('filename', '')})]"
    return header + "\n" + "\n".join(lines)



def _build_prompt_text(query: str, label: str, data: dict | None) -> str:
    """
    apply_chat_template 없이 Qwen2.5 채팅 포맷을 직접 구성합니다.
    <|im_start|>/<|im_end|> 특수 토큰을 사용합니다.
    """
    system_msg = (
        "당신은 사내 업무 도우미입니다. "
        "주어진 데이터를 바탕으로 질문에 간결하고 정확하게 한국어로 답변하세요. "
        "데이터에 없는 내용은 추측하지 말고 '데이터에 없습니다'라고 답하세요."
    )

    data_text = _format_data_as_text(data) if data else ""
    user_msg  = f"{data_text}\n\n질문: {query}" if data_text else f"질문: {query}"

    return (
        f"<|im_start|>system\n{system_msg}<|im_end|>\n"
        f"<|im_start|>user\n{user_msg}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )


def _is_garbled_text(text: str) -> bool:
    """
    모델 출력이 비정상 반복(예: filefilefile...)일 때 감지합니다.
    """
    if not text:
        return True
    low = text.lower()
    if re.search(r"(file){6,}", low):
        return True

    words = re.findall(r"[a-zA-Z가-힣0-9_]+", low)
    if len(words) >= 12:
        unique_ratio = len(set(words)) / len(words)
        if unique_ratio < 0.25:
            return True

    # 긴 답변에서 동일 토막 반복 감지
    chunks = [low[i:i+25] for i in range(0, min(len(low), 500), 25)]
    if chunks:
        repeated = len(chunks) - len(set(chunks))
        if repeated >= 6:
            return True
    return False


def _fallback_answer(label: str, data: dict | None) -> str:
    if not data or not data.get("rows"):
        return f"{label} 관련 데이터 조회 결과가 없습니다. 질문 조건을 조금 더 구체적으로 입력해 주세요."

    columns = data.get("columns", [])
    rows = data.get("rows", [])
    summary = data.get("summary", f"{label} 데이터 조회 결과")

    lines = [f"{summary}", "핵심 항목:"]
    preview = rows[:5]
    for row in preview:
        parts = []
        for c, v in zip(columns, row):
            if v in ("", None):
                continue
            parts.append(f"{c}={v}")
        if parts:
            lines.append("- " + ", ".join(parts[:4]))

    if len(rows) > len(preview):
        lines.append(f"외 {len(rows) - len(preview)}개 항목이 더 있습니다.")
    return "\n".join(lines)


def _evaluate_plus_minus_expression(expr: str) -> int | None:
    """
    숫자 +/ - 연산만 포함한 간단한 식을 계산합니다.
    예: "50 + 42 - 13" -> 79
    """
    cleaned = expr.replace(",", "").replace(" ", "")
    if not cleaned:
        return None
    if not re.fullmatch(r"[0-9+\-]+", cleaned):
        return None
    if cleaned[0] in "+-":
        cleaned = "0" + cleaned

    nums = re.findall(r"\d+", cleaned)
    ops = re.findall(r"[+\-]", cleaned)
    if not nums:
        return None

    total = int(nums[0])
    for i, op in enumerate(ops, start=1):
        if i >= len(nums):
            return None
        val = int(nums[i])
        total = total + val if op == "+" else total - val
    return total


def _apply_math_guard(answer: str) -> str:
    """
    '식 = 값개' 패턴에서 식을 재계산해 값이 다르면 자동 교정합니다.
    """
    fixed_lines = []
    for line in answer.splitlines():
        if "=" not in line:
            fixed_lines.append(line)
            continue

        left, right = line.split("=", 1)
        right_num_match = re.search(r"(-?\d[\d,]*)", right)
        if not right_num_match:
            fixed_lines.append(line)
            continue

        # 좌변의 마지막 산술식 구간을 추출 (예: "A-1 총수량: 50+40-3")
        expr_candidates = re.findall(r"(\d[\d,\s+\-]*\d)", left)
        if not expr_candidates:
            fixed_lines.append(line)
            continue

        expr = expr_candidates[-1]
        calc = _evaluate_plus_minus_expression(expr)
        if calc is None:
            fixed_lines.append(line)
            continue

        shown = int(right_num_match.group(1).replace(",", ""))
        if shown == calc:
            fixed_lines.append(line)
            continue

        corrected_right = (
            right[: right_num_match.start()]
            + f"{calc:,}"
            + right[right_num_match.end() :]
        )
        fixed_lines.append(f"{left}={corrected_right}  [자동검산 보정]")

    return "\n".join(fixed_lines)


def generate_answer(query: str, label: str, data: dict | None = None) -> str:
    """
    사용자 질문과 조회된 데이터를 받아 자연어 답변을 반환합니다.
    """
    prompt = _build_prompt_text(query, label, data)

    inputs = _tokenizer(prompt, return_tensors="pt").to(_model.device)

    # <|im_end|> 토큰 ID 확인 (없으면 eos_token 사용)
    im_end_id = _tokenizer.convert_tokens_to_ids("<|im_end|>")
    if im_end_id == _tokenizer.unk_token_id:
        im_end_id = _tokenizer.eos_token_id

    with torch.no_grad():
        output_ids = _model.generate(
            **inputs,
            max_new_tokens=220,
            do_sample=False,
            repetition_penalty=1.2,
            no_repeat_ngram_size=4,
            eos_token_id=[im_end_id, _tokenizer.eos_token_id],
            pad_token_id=_tokenizer.eos_token_id,
            temperature=None,
            top_p=None,
            top_k=None,
        )

    generated = output_ids[0][inputs.input_ids.shape[1]:]
    answer    = _tokenizer.decode(generated, skip_special_tokens=True).strip()

    # <|im_end|> 이후 내용 제거 (혹시 포함된 경우)
    if "<|im_end|>" in answer:
        answer = answer.split("<|im_end|>")[0].strip()

    answer = _apply_math_guard(answer)

    if _is_garbled_text(answer):
        return _apply_math_guard(_fallback_answer(label, data))
    return answer
