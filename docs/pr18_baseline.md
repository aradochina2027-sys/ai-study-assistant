# ПР18 — Базові вимірювання

## Загальна статистика

- Запитів: 141
- Подій: 1683
- Завершено: 10
- Помилок: 1
- Скасовано: 0
- Некоректних рядків журналу: 0

## Тривалість завершених запитів

- Кількість вимірювань: 11
- p50: 13407.38 мс
- p95: 25031.74 мс

## Успішні виклики Gateway

- Кількість вимірювань: 125
- p50: 3566.71 мс
- p95: 14738.10 мс

## Найчастіші події

- `controller.get_result`: 252
- `storage.read`: 252
- `gateway.provider_call`: 143
- `service.started`: 141
- `storage.documents_loaded`: 141
- `orchestration.started`: 141
- `gateway.provider_returned`: 125
- `orchestration.validated`: 123
- `service.finished`: 123
- `provider.mock_called`: 39
- `orchestration.context_v=study-v1;sources=data/notes.md;chars=793`: 39
- `orchestration.context_v=study-v2;sources=data/notes.md;chars=793`: 37
- `orchestration.context_v=study-v3;sources=data/notes.md;chars=793`: 36
- `orchestration.context_v=study-v4;sources=data/notes.md;chars=793`: 22
- `controller.accepted`: 11

## Обмеження вимірювань

- Показники обчислено лише з наявних записів журналу.
- Загальна тривалість визначена між accepted і завершальною подією.
- Тривалість Gateway визначена між provider_call і provider_returned.
- Це не окремий час генерації токенів і не час першого токена.
- Невдалі виклики Gateway можуть не мати події provider_returned.
- Для надійного вибору тайм-аутів потрібні додаткові вимірювання.
