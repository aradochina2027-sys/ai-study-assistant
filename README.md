# AI Study Assistant

Інтелектуальний помічник для пошуку інформації
у власних навчальних матеріалах.

## Статус проєкту

ПР13 — постановка задачі.
ПР14 — архітектура та технічний прототип.

У ПР14 використовується mock-провайдер.

## Технології

- Python
- FastAPI
- Pydantic
- SQLite
- HTML / JavaScript
- Pytest
- Mock Provider

## Запуск на Windows

Створити віртуальне середовище:

```powershell
python -m venv .venv
```

Активувати:

```powershell
.\.venv\Scripts\Activate.ps1
```

Встановити залежності:

```powershell
python -m pip install -r requirements.txt
```

Створити конфігурацію:

```powershell
Copy-Item .env.example .env
```

Запустити сервер:

```powershell
python -m uvicorn app.main:app --reload
```

## Адреси

Головна сторінка:
http://127.0.0.1:8000/

Health:
http://127.0.0.1:8000/api/health

Swagger:
http://127.0.0.1:8000/docs

## Перевірка

```powershell
python -m pytest -v
```

## Структура

- app/ — серверні модулі;
- app/providers/ — провайдери;
- static/ — вебінтерфейс;
- tests/ — автоматичні тести;
- docs/ — документація;
- eval/ — приклади оцінювання;
- data/ — навчальні матеріали;
- logs/ — локальні журнали.

## Документація

- docs/passport.md
- docs/architecture.md
- docs/api.md
- docs/adr/
- docs/backlog.md
- AI_USAGE.md

## Подальша розробка

ПР15 — справжня AI-модель.
ПР16 — семантичний пошук RAG.
ПР17 — розвиток інтерфейсу.
ПР18 — надійність.
ПР19 — оцінювання та безпека.
ПР20 — розгортання.

У ПР14 немає платних AI-викликів.