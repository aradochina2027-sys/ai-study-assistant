# API-контракти — ПР14

## GET /api/health

Повертає стан системи:
- status
- version
- provider
- model
- mode
- storage
- extra
- uptime_s

HTTP 200 — сервер працює.

## POST /api/ask

Вхід:

{
  "question": "Як створити Git-репозиторій?",
  "documents": ["data/notes.md"]
}

HTTP 202 — завдання прийнято:

{
  "request_id": "UUID",
  "status": "pending",
  "status_url": "/api/requests/UUID"
}

## GET /api/requests/{request_id}

Повертає стан завдання:
pending, running, completed або cancelled.

Після завершення повертає result:

{
  "status": "ok",
  "answer": "Текст відповіді",
  "sources": ["data/notes.md"]
}

Якщо інформації недостатньо:
status = no_evidence.

## POST /api/requests/{request_id}/cancel

Скасовує збереження результату
завдання, яке ще не завершилося.

## Єдиний формат помилки

{
  "error": {
    "kind": "provider_unavailable",
    "message": "Провайдер недоступний",
    "retryable": true,
    "request_id": "UUID"
  }
}

HTTP-статуси:
400 — неправильний документ;
404 — ресурс не знайдено;
409 — конфлікт стану;
422 — неправильний формат;
429 — ліміт запитів;
503 — провайдер недоступний;
504 — тайм-аут.

## Контракт провайдера

ProviderRequest:
- request_id: str
- model: str
- messages: list
- documents: list
- timeout_s: float

ProviderResult:
- text: str
- model: str
- input_tokens: int
- output_tokens: int

Потік відповіді:
stream(request) -> Iterator[str]

## Структурований вихід AI

ModelOutput:
- status: ok або no_evidence
- answer: str
- sources: list[str]

Відповідь перевіряється через Pydantic.

У ПР15 додається справжня AI-модель.
У ПР16 додається RAG.