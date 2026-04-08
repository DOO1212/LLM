def format_output(result):
    if "total" in result:
        return f"""
[결과]
총 합계: {result['total']}원
"""
    elif "error" in result:
        return "처리할 수 없는 요청입니다."
