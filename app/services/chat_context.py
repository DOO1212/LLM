from dataclasses import dataclass
from threading import Lock


@dataclass
class ChatContext:
    last_query: str
    last_label: str
    last_answer: str


_contexts: dict[str, ChatContext] = {}
_lock = Lock()


def get_context(employee_id: str) -> ChatContext | None:
    with _lock:
        return _contexts.get(employee_id)


def set_context(employee_id: str, query: str, label: str, answer: str = ""):
    if not employee_id or not label:
        return
    with _lock:
        _contexts[employee_id] = ChatContext(
            last_query=query or "",
            last_label=label or "",
            last_answer=answer or "",
        )

