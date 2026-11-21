
"""
Практична робота №18.
Аналіз структурованого JSONL-журналу StudyAI.

Результати:
- кількість запитів;
- p50 / p95 тривалості;
- помилки за видами;
- повтори AI-запитів;
- статистика кешу;
- токени й вартість;
- Markdown-звіт.

Підтримуються попередні журнали та
файли ротації app.jsonl.1, app.jsonl.2.

Не виводяться особисті дані, тексти
запитань, сирі request_id чи секрети.
"""

from __future__ import annotations

import argparse
import json
import math
import re

from collections import Counter, defaultdict, deque
from datetime import datetime, timezone
from pathlib import Path


SAFE_LABEL = re.compile(
    r"^[a-z][a-z0-9_]{0,63}$"
)

KNOWN_LAYERS = {
    "controller",
    "service",
    "rag",
    "gateway",
    "storage",
    "orchestration",
    "provider",
    "budget",
    "embedder",
}

KNOWN_ERRORS = {
    "timeout",
    "rate_limit",
    "provider_unavailable",
    "provider_error",
    "provider_invalid_response",
    "deadline_exceeded",
    "invalid_output",
    "auth",
    "bad_request",
    "rag_unavailable",
    "budget",
    "circuit_open",
    "concurrency_limit",
    "client_rate_limit",
    "ai_disabled",
    "internal_error",
    "validation_error",
    "idempotency_conflict",
    "invalid_idempotency_key",
    "not_found",
    "conflict",
    "document_missing",
    "invalid_document",
    "document_too_large",
    "empty_documents",
    "rag_document_unsupported",
}

TERMINAL_SERVICE = {
    "finished",
    "no_evidence",
    "internal_error",
    "ai_disabled",
}


def safe_label(value):
    """Не допускає довільний текст до звіту."""

    if (
        not isinstance(value, str)
        or not SAFE_LABEL.fullmatch(value)
    ):
        return "other"

    return value


def parse_time(value):
    """Читає дату ISO 8601."""

    if not isinstance(value, str):
        return None

    try:
        result = datetime.fromisoformat(
            value.replace("Z", "+00:00")
        )

        if result.tzinfo is None:
            result = result.replace(
                tzinfo=timezone.utc
            )

        return result.astimezone(
            timezone.utc
        )

    except ValueError:
        return None


def valid_number(value):
    """Приймає лише невід'ємне скінченне число."""

    if type(value) not in (
        int,
        float,
    ):
        return None

    result = float(value)

    if (
        not math.isfinite(result)
        or result < 0
    ):
        return None

    return result


def percentile(values, fraction):
    """
    Обчислює процентиль за лінійною
    інтерполяцією відсортованих значень.
    """

    if not values:
        return None

    numbers = sorted(values)

    position = fraction * (
        len(numbers) - 1
    )

    left = math.floor(position)
    right = math.ceil(position)

    weight = position - left

    return (
        numbers[left] * (1 - weight)
        + numbers[right] * weight
    )


def format_value(value, suffix=""):
    if value is None:
        return "немає даних"

    return f"{value:.2f}{suffix}"


# ==========================================
# 1. Пошук файлів журналу
# ==========================================

def find_log_files(
    path: Path,
    include_rotated=True,
):
    """
    Повертає основний журнал і копії ротації.

    Старіші файли читаються перед новішими.
    """

    if not include_rotated:
        return [path] if path.is_file() else []

    rotated = []

    for candidate in path.parent.glob(
        path.name + ".*"
    ):
        suffix = candidate.name.removeprefix(
            path.name + "."
        )

        if (
            suffix.isdigit()
            and candidate.is_file()
        ):
            rotated.append(
                (
                    int(suffix),
                    candidate,
                )
            )

    rotated.sort(
        key=lambda item: item[0],
        reverse=True,
    )

    result = [
        file
        for _, file in rotated
    ]

    if path.is_file():
        result.append(path)

    return result


# ==========================================
# 2. Завантаження журналу
# ==========================================

def load_events(paths):
    """
    Зчитує лише службові поля.

    Текст запитань, документи, повідомлення
    моделі та інші дані ігноруються.
    """

    events = []
    invalid_lines = 0

    for path in paths:
        with path.open(
            "r",
            encoding="utf-8",
            errors="replace",
        ) as stream:

            for line in stream:
                if not line.strip():
                    continue

                try:
                    raw = json.loads(line)

                except json.JSONDecodeError:
                    invalid_lines += 1
                    continue

                if not isinstance(raw, dict):
                    invalid_lines += 1
                    continue

                timestamp = parse_time(
                    raw.get("ts")
                    or raw.get("timestamp")
                )

                request_id = raw.get(
                    "request_id"
                )

                if (
                    timestamp is None
                    or not isinstance(
                        request_id,
                        str,
                    )
                    or not request_id
                ):
                    invalid_lines += 1
                    continue

                layer = safe_label(
                    raw.get("layer")
                    or raw.get("stage")
                )

                if layer not in KNOWN_LAYERS:
                    layer = "other"

                error_kind = raw.get(
                    "error_kind"
                )

                if error_kind not in KNOWN_ERRORS:
                    error_kind = None

                events.append(
                    {
                        "ts": timestamp,
                        "request_id": request_id,
                        "layer": layer,
                        "event": safe_label(
                            raw.get("event")
                        ),
                        "duration_ms": valid_number(
                            raw.get("duration_ms")
                        ),
                        "input_tokens": valid_number(
                            raw.get("input_tokens")
                        ),
                        "output_tokens": valid_number(
                            raw.get("output_tokens")
                        ),
                        "cost": valid_number(
                            raw.get("cost")
                        ),
                        "cached": (
                            raw.get("cached") is True
                        ),
                        "fallback": (
                            raw.get("fallback") is True
                        ),
                        "error_kind": error_kind,
                    }
                )

    events.sort(
        key=lambda item: item["ts"]
    )

    return events, invalid_lines


# ==========================================
# 3. Аналіз журналу
# ==========================================

def analyze(events):
    errors = Counter()
    durations_by_stage = defaultdict(list)
    by_request = defaultdict(list)

    pending_provider_calls = defaultdict(
        deque
    )

    provider_calls = 0
    provider_returns = 0

    retries_planned = 0
    retries_succeeded = 0

    cache_hits = 0
    cache_misses = 0
    cached_events = 0
    fallback_events = 0

    input_tokens = 0
    output_tokens = 0
    token_records = 0

    total_cost = 0.0
    cost_records = 0

    for item in events:
        request_id = item["request_id"]
        layer = item["layer"]
        event = item["event"]

        by_request[
            request_id
        ].append(item)

        # ----------------------------------
        # Тривалість етапів
        # ----------------------------------

        if item["duration_ms"] is not None:
            durations_by_stage[
                layer
            ].append(
                item["duration_ms"]
            )

        # ----------------------------------
        # Помилки
        # ----------------------------------

        if item["error_kind"] is not None:
            errors[
                item["error_kind"]
            ] += 1

        elif event.startswith("error_"):
            kind = event.removeprefix(
                "error_"
            )

            if kind in KNOWN_ERRORS:
                errors[kind] += 1

        # ----------------------------------
        # Повторні спроби
        # ----------------------------------

        if event == "retry_scheduled":
            retries_planned += 1

        elif event == "retry_succeeded":
            retries_succeeded += 1

        # ----------------------------------
        # Кеш
        # ----------------------------------

        if event == "cache_hit":
            cache_hits += 1

        elif event == "cache_miss":
            cache_misses += 1

        if item["cached"]:
            cached_events += 1

        if item["fallback"]:
            fallback_events += 1

        # ----------------------------------
        # Фактичні виклики AI
        # ----------------------------------

        if (
            layer == "gateway"
            and event == "provider_call"
        ):
            provider_calls += 1

            pending_provider_calls[
                request_id
            ].append(item["ts"])

        elif (
            layer == "gateway"
            and (
                event.startswith("error_")
                or event == "provider_exception"
            )
        ):
            # Закриваємо невдалу спробу,
            # щоб не сплутати її з наступною.
            if pending_provider_calls[request_id]:
                pending_provider_calls[
                    request_id
                ].popleft()

        elif (
            layer == "gateway"
            and event == "provider_returned"
        ):
            provider_returns += 1

            if pending_provider_calls[request_id]:
                started = pending_provider_calls[
                    request_id
                ].popleft()

                # Для старих журналів без
                # duration_ms можна оцінити
                # тривалість за двома подіями.
                if item["duration_ms"] is None:
                    elapsed = (
                        item["ts"] - started
                    ).total_seconds() * 1000

                    if elapsed >= 0:
                        durations_by_stage[
                            "gateway"
                        ].append(elapsed)

        # ----------------------------------
        # Токени та вартість
        # ----------------------------------

        if (
            layer == "gateway"
            and event == "provider_returned"
        ):
            input_value = item[
                "input_tokens"
            ]

            output_value = item[
                "output_tokens"
            ]

            if (
                input_value is not None
                and output_value is not None
            ):
                input_tokens += int(
                    input_value
                )

                output_tokens += int(
                    output_value
                )

                token_records += 1

            if item["cost"] is not None:
                total_cost += item["cost"]
                cost_records += 1

    # ======================================
    # 4. Тривалість повного запиту
    # ======================================

    request_durations = []

    completed = 0
    failed = 0

    for request_events in by_request.values():

        start = next(
            (
                item["ts"]
                for item in request_events
                if (
                    item["layer"],
                    item["event"],
                ) in {
                    ("controller", "accepted"),
                    ("service", "started"),
                }
            ),
            None,
        )

        end = next(
            (
                item
                for item in request_events
                if item["layer"] == "service"
                and (
                    item["event"] in TERMINAL_SERVICE
                    or item["event"].startswith(
                        "error_"
                    )
                )
            ),
            None,
        )

        if end is None:
            continue

        if end["event"] in {
            "finished",
            "no_evidence",
        }:
            completed += 1
        else:
            failed += 1

        if (
            start is not None
            and end["ts"] >= start
        ):
            elapsed = (
                end["ts"] - start
            ).total_seconds() * 1000

            request_durations.append(
                elapsed
            )

    return {
        "requests": len(by_request),
        "completed": completed,
        "failed": failed,
        "request_durations": request_durations,
        "stage_durations": durations_by_stage,
        "errors": errors,
        "provider_calls": provider_calls,
        "provider_returns": provider_returns,
        "retries_planned": retries_planned,
        "retries_succeeded": retries_succeeded,
        "cache_hits": cache_hits,
        "cache_misses": cache_misses,
        "cached_events": cached_events,
        "fallback_events": fallback_events,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "token_records": token_records,
        "total_cost": total_cost,
        "cost_records": cost_records,
    }


# ==========================================
# 5. Генерація Markdown-звіту
# ==========================================

def create_report(
    stats,
    events,
    invalid_lines,
    files_count,
):
    durations = stats[
        "request_durations"
    ]

    lines = [
        "# ПР18 — Аналіз журналу StudyAI",
        "",
        "## 1. Джерело даних",
        "",
        (
            "- Прочитано JSONL-файлів: "
            f"**{files_count}**"
        ),
        (
            "- Коректних подій: "
            f"**{len(events)}**"
        ),
        (
            "- Некоректних рядків: "
            f"**{invalid_lines}**"
        ),
        (
            "- Унікальних ідентифікаторів: "
            f"**{stats['requests']}**"
        ),
        "",
        "## 2. Завершення запитів",
        "",
        (
            "- Успішно завершених сценаріїв: "
            f"**{stats['completed']}**"
        ),
        (
            "- Сценаріїв із помилкою: "
            f"**{stats['failed']}**"
        ),
        (
            "- Вимірювань тривалості: "
            f"**{len(durations)}**"
        ),
        (
            "- p50: **"
            + format_value(
                percentile(durations, 0.50),
                " мс",
            )
            + "**"
        ),
        (
            "- p95: **"
            + format_value(
                percentile(durations, 0.95),
                " мс",
            )
            + "**"
        ),
        "",
        "## 3. Тривалість за етапами",
        "",
        "| Етап | Вимірювань | p50, мс | p95, мс |",
        "|---|---:|---:|---:|",
    ]

    stages = stats[
        "stage_durations"
    ]

    if stages:
        for stage, values in sorted(
            stages.items()
        ):
            p50 = format_value(
                percentile(values, 0.50)
            )

            p95 = format_value(
                percentile(values, 0.95)
            )

            lines.append(
                f"| `{stage}` | {len(values)} "
                f"| {p50} | {p95} |"
            )
    else:
        lines.append(
            "| Немає даних | 0 | — | — |"
        )

    # ======================================
    # Повтори
    # ======================================

    lines.extend(
        [
            "",
            "## 4. AI-провайдер і повтори",
            "",
            (
                "- Фактичних спроб: "
                f"**{stats['provider_calls']}**"
            ),
            (
                "- Успішних повернень: "
                f"**{stats['provider_returns']}**"
            ),
            (
                "- Запланованих повторів: "
                f"**{stats['retries_planned']}**"
            ),
            (
                "- Успішних повторів: "
                f"**{stats['retries_succeeded']}**"
            ),
            "",
            "## 5. Помилки за видом",
            "",
        ]
    )

    if stats["errors"]:
        for kind, count in stats[
            "errors"
        ].most_common():
            lines.append(
                f"- `{kind}`: {count} подій"
            )
    else:
        lines.append(
            "- Немає зафіксованих "
            "подій із видом помилки."
        )

    # ======================================
    # Кеш
    # ======================================

    lines.extend(
        [
            "",
            "## 6. Кешування",
            "",
        ]
    )

    hits = stats["cache_hits"]
    misses = stats["cache_misses"]

    if hits + misses > 0:
        hit_rate = (
            100 * hits / (hits + misses)
        )

        lines.extend(
            [
                f"- Cache hits: **{hits}**",
                f"- Cache misses: **{misses}**",
                (
                    "- Частка cache hit: "
                    f"**{hit_rate:.1f}%**"
                ),
            ]
        )
    else:
        lines.append(
            "- Частка cache hit: **немає даних**. "
            "У журналі немає окремих подій "
            "`cache_hit` та `cache_miss`."
        )

    lines.append(
        "- Подій із `cached=true`: "
        f"**{stats['cached_events']}**."
    )

    lines.append(
        "- Подій із `fallback=true`: "
        f"**{stats['fallback_events']}**."
    )

    # ======================================
    # Токени й вартість
    # ======================================

    lines.extend(
        [
            "",
            "## 7. Токени та вартість",
            "",
        ]
    )

    if stats["token_records"]:
        lines.extend(
            [
                (
                    "- Викликів із даними про токени: "
                    f"**{stats['token_records']}**"
                ),
                (
                    "- Вхідних токенів: "
                    f"**{stats['input_tokens']}**"
                ),
                (
                    "- Вихідних токенів: "
                    f"**{stats['output_tokens']}**"
                ),
            ]
        )
    else:
        lines.append(
            "- Кількість токенів: "
            "**немає даних**."
        )

    if stats["cost_records"]:
        lines.append(
            "- Сума записаних витрат: "
            f"**{stats['total_cost']:.6f}** "
            "(одиницю вартості потрібно "
            "перевірити в конфігурації)."
        )
    else:
        lines.append(
            "- Вартість: **немає даних**. "
            "Відсутність записів не означає "
            "нульових витрат."
        )

    # ======================================
    # Обмеження аналізу
    # ======================================

    lines.extend(
        [
            "",
            "## 8. Обмеження аналізу",
            "",
            (
                "- Показники отримано лише "
                "з наявних записів JSONL."
            ),
            (
                "- p50/p95 за етапами обчислено "
                "із duration_ms або за парою "
                "provider_call/provider_returned."
            ),
            (
                "- Загальна тривалість є оцінкою "
                "часу між прийманням запиту "
                "та завершальною подією сервісу."
            ),
            (
                "- Незавершені задачі не входять "
                "до вимірювань завершення."
            ),
            (
                "- Ротація журналу може "
                "видалити старі події."
            ),
            (
                "- Поточний кеш embeddings "
                "не записує cache_hit/cache_miss "
                "до JSONL автоматично."
            ),
            (
                "- Для вибору тайм-аутів "
                "потрібні реальні вимірювання "
                "Ollama, а не лише mock."
            ),
            (
                "- p95 за одним або двома "
                "вимірюваннями не слід вважати "
                "надійною оцінкою."
            ),
            (
                "- Сирі request_id, запитання, "
                "промпти, відповіді та секрети "
                "у звіт не включено."
            ),
            "",
        ]
    )

    return "\n".join(lines)


# ==========================================
# 6. Запуск аналізатора
# ==========================================

def main():
    parser = argparse.ArgumentParser(
        description=(
            "ПР18 — аналіз структурованого "
            "журналу AI Study Assistant"
        )
    )

    parser.add_argument(
        "--log",
        default="logs/app.jsonl",
        help="Основний JSONL-журнал",
    )

    parser.add_argument(
        "--output",
        default="docs/pr18_analysis.md",
        help="Markdown-звіт",
    )

    parser.add_argument(
        "--no-rotated",
        action="store_true",
        help="Не читати файли ротації",
    )

    args = parser.parse_args()

    log_path = Path(
        args.log
    )

    paths = find_log_files(
        log_path,
        include_rotated=not args.no_rotated,
    )

    if not paths:
        print(
            f"Журнал не знайдено: {log_path}"
        )

        print(
            "Спочатку запусти StudyAI "
            "та виконай кілька запитів."
        )

        return 1

    events, invalid_lines = load_events(
        paths
    )

    if not events:
        print(
            "У журналі немає коректних подій. "
            "Звіт не створено."
        )

        return 1

    stats = analyze(
        events
    )

    report = create_report(
        stats,
        events,
        invalid_lines,
        len(paths),
    )

    output_path = Path(
        args.output
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path.write_text(
        report,
        encoding="utf-8",
    )

    print(report)

    print(
        "\nЗвіт збережено: "
        f"{output_path}"
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
