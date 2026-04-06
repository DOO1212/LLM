from request_parser import parse_query
from router import route
from output_template import format_output

def main():
    query = input("질문: ")

    # 1. LLM → 해석
    parsed = parse_query(query)

    # 2. 라우팅
    result = route(parsed)

    # 3. 출력
    output = format_output(result)

    print(output)


if __name__ == "__main__":
    main()
