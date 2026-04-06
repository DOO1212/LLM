from langchain_community.llms import Ollama

llm = Ollama(model="llama3", temperature=0)

question = "안녕하세요. 앞으로 모든 답변은 한국어로 해주세요."
response = llm.invoke(question)

print(response)