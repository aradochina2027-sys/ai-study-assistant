
"""
ПР18 — Базовий аналіз журналу AI Study Assistant.

Читає JSONL-журнал і формує звіт:
- кількість подій;
- кількість унікальних запитів;
- приблизна тривалість завершених запитів;
- p50 та p95;
- розподіл подій;
- тривалість успішних викликів Gateway.

Не виводить request_id або тексти користувачів.
"""

import argparse
import json

from collections import Counter, defaultdict, deque
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

FINISHED = {
    "completed",
    "failed",
    "cancelled",
}


def parse_time(value):
    """Перетворює ISO-час у datetime."""

    if not isinstance(value, str):
        return None

    try:
        parsed = datetime.fromisoformat(
            value.replace("Z", "+00:00")
        )

        if parsed.tzinfo is None:
            parsed = parsed.replace(
                tzinfo=timezone.utc
            )

        return parsed

    except ValueError:
        return None


def percentile(values, fraction):
    """Обчислює процентиль з інтерполяцією."""

    if not values:
        return None

    ordered = sorted(values)

    if len(ordered) == 1:
        return ordered[0]

    position = (len(ordered) - 1) * fraction

    lower = int(position)
    upper = min(
        lower + 1,
        len(ordered) - 1,
    )

    weight = position - lower

    return (
        ordered[lower] * (1 - weight)
        + ordered[upper] * weight
    )


def format_ms(value):
    if value is None:
        return "недостатньо даних"

    return f"{value:.2f} мс"


def load_events(path):
    """
    Читає тільки дозволені поля журналу.

    Не передає до результату сторонні
    поля, які можуть містити чутливі дані.
    """

    events = []
    invalid = 0

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        for line in file:
            if not line.strip():
                continue

            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                invalid += 1
                continue

            if not isinstance(item, dict):
                invalid += 1
                continue

            timestamp = parse_time(
                item.get("ts")
                or item.get("timestamp")
            )

            request_id = item.get("request_id")
            layer = item.get(
                "layer",
                item.get("stage"),
            )
            event = item.get("event")

            if not (
                timestamp
                and isinstance(request_id, str)
                and isinstance(layer, str)
                and isinstance(event, str)
            ):
                invalid += 1
                continue

            events.append(
                {
                    "timestamp": timestamp,
                    "request_id": request_id,
                    "layer": layer,
                    "event": event,
                }
            )

    events.sort(
        key=lambda item: item["timestamp"]
    )

    return events, invalid


def analyze(events):
    """Формує агреговану статистику."""

    by_request = defaultdict(list)
    event_counts = Counter()

    for item in events:
        by_request[
            item["request_id"]
        ].append(item)

        event_counts[
            f"{item['layer']}.{item['event']}"
        ] += 1

    request_durations = []
    provider_durations = []
    completed = 0
    failed = 0
    cancelled = 0

    for request_events in by_request.values():
        starts = [
            item["timestamp"]
            for item in request_events
            if item["event"] == "accepted"
        ]

        ends = [
            item
            for item in request_events
            if item["event"] in FINISHED
        ]

        if starts and ends:
            start = min(starts)

            valid_ends = [
                item
                for item in ends
                if item["timestamp"] >= start
            ]

            if valid_ends:
                end = min(
                    valid_ends,
                    key=lambda item: item["timestamp"],
                )

                elapsed = (
                    end["timestamp"] - start
                ).total_seconds() * 1000

                request_durations.append(
                    elapsed
                )

        statuses = {
            item["event"]
            for item in request_events
            if item["event"] in FINISHED
        }

        if "completed" in statuses:
            completed += 1
        elif "failed" in statuses:
            failed += 1
        elif "cancelled" in statuses:
            cancelled += 1

        # Пара викликів Gateway:
        # provider_call -> provider_returned.
        # Невдалі або незавершені виклики
        # не включаємо у тривалість успішних.

        calls = deque()

        for item in request_events:
            if item["layer"] != "gateway":
                continue

            if item["event"] == "provider_call":
                calls.append(
                    item["timestamp"]
                )

            elif (
                item["event"] == "provider_returned"
                and calls
            ):
                started = calls.popleft()

                elapsed = (
                    item["timestamp"] - started
                ).total_seconds() * 1000

                if elapsed >= 0:
                    provider_durations.append(
                        elapsed
                    )

    return {
        "requests": len(by_request),
        "events": len(events),
        "completed": completed,
        "failed": failed,
        "cancelled": cancelled,
        "request_durations": request_durations,
        "provider_durations": provider_durations,
        "event_counts": event_counts,
    }


def build_report(stats, invalid):
    """Формує Markdown-звіт без персональних даних."""

    request_times = stats["request_durations"]
    provider_times = stats["provider_durations"]

    lines = [
        "# ПР18 — Базові вимірювання",
        "",
        "## Загальна статистика",
        "",
        f"- Запитів: {stats['requests']}",
        f"- Подій: {stats['events']}",
        f"- Завершено: {stats['completed']}",
        f"- Помилок: {stats['failed']}",
        f"- Скасовано: {stats['cancelled']}",
        f"- Некоректних рядків журналу: {invalid}",
        "",
        "## Тривалість завершених запитів",
        "",
        (
            "- Кількість вимірювань: "
            f"{len(request_times)}"
        ),
        (
            "- p50: "
            f"{format_ms(percentile(request_times, 0.50))}"
        ),
        (
            "- p95: "
            f"{format_ms(percentile(request_times, 0.95))}"
        ),
        "",
        "## Успішні виклики Gateway",
        "",
        (
            "- Кількість вимірювань: "
            f"{len(provider_times)}"
        ),
        (
            "- p50: "
            f"{format_ms(percentile(provider_times, 0.50))}"
        ),
        (
            "- p95: "
            f"{format_ms(percentile(provider_times, 0.95))}"
        ),
        "",
        "## Найчастіші події",
        "",
    ]

    for name, count in stats[
        "event_counts"
    ].most_common(15):
        lines.append(
            f"- `{name}`: {count}"
        )

    lines.extend(
        [
            "",
            "## Обмеження вимірювань",
            "",
            (
                "- Показники обчислено лише з "
                "наявних записів журналу."
            ),
            (
                "- Загальна тривалість визначена "
                "між accepted і завершальною подією."
            ),
            (
                "- Тривалість Gateway визначена між "
                "provider_call і provider_returned."
            ),
            (
                "- Це не окремий час генерації токенів "
                "і не час першого токена."
            ),
            (
                "- Невдалі виклики Gateway можуть "
                "не мати події provider_returned."
            ),
            (
                "- Для надійного вибору тайм-аутів "
                "потрібні додаткові вимірювання."
            ),
            "",
        ]
    )

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description=(
            "ПР18: аналіз журналу AI Study Assistant"
        )
    )

    parser.add_argument(
        "--log",
        default=str(
            ROOT / "logs" / "app.jsonl"
        ),
        help="Шлях до JSONL-журналу",
    )

    parser.add_argument(
        "--output",
        default=str(
            ROOT / "docs" / "pr18_baseline.md"
        ),
        help="Шлях до Markdown-звіту",
    )

    args = parser.parse_args()

    log_path = Path(args.log)
    output_path = Path(args.output)

    if not log_path.is_file():
        print(
            "Журнал не знайдено: "
            f"{log_path}"
        )
        print(
            "Спочатку запусти застосунок "
            "і виконай кілька запитів."
        )
        return 1

    events, invalid = load_events(
        log_path
    )

    stats = analyze(events)

    report = build_report(
        stats,
        invalid,
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
