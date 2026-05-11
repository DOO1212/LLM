# prompts/entity_prompt.py

KNOWN_ENTITIES = [

    "패딩",
    "청바지",
    "이어폰",
    "충전기",
    "키보드",
    "마우스",
    "모니터"
]


def build_entity_injection_prompt(entities):

    if not entities:

        return ""


    prompt = (

        "--------------------------------------------------\n"
        "[추출된 품목명]\n"
        "--------------------------------------------------\n\n"
    )


    for entity in entities:

        prompt += f"- {entity}\n"


    prompt += (

        "\n"
        "반드시 위 품목명을 "
        "filters에 포함해야 한다.\n"

        "품목명 검색은 반드시 "
        "LIKE를 사용해야 한다.\n"
    )


    return prompt