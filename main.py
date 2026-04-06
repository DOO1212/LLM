from request_parser import parse_query
from router import route
from output_template import format_output

def main():
    query = input("질문: ")
    
    parsed = parse_query(query)
    result = route(parsed)

    if isinstance(result, dict) and "error" in result:
        print("❌", result["error"])
    else:
        print("✅ 답변:", result)


if __name__ == "__main__":
    main()
