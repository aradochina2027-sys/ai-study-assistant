
import argparse
import json
import uuid

from collections import Counter, defaultdict
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

from app.main import create_app
from app.prompting import read_template
from app.models import AskInput, AppError
from app.settings import load_settings


def check_case(case: dict, result) -> list[str]:
    """Перевіряє відповідь за умовами тестового прикладу."""

    problems = []

    for expected in case["expect"]:
        check = expected["check"]
        value = expected["value"]

        if check == "contains":
            if value.casefold() not in result.answer.casefold():
                problems.append(
                    f"Відповідь не містить: {value}"
                )

        elif check == "status":
            if result.status != value:
                problems.append(
                    f"Очікувався статус {value}, "
                    f"отримано {result.status}"
                )

        elif check == "source_in":
            if value not in result.sources:
                problems.append(
                    f"Відсутнє джерело: {value}"
                )

        else:
            problems.append(
                f"Невідома перевірка: {check}"
            )

    return problems


def main():
    parser = argparse.ArgumentParser(
        description="Оцінювання AI Study Assistant — ПР15"
    )

    parser.add_argument(
        "--cases",
        default="eval/cases.json",
        help="Шлях до тестових прикладів",
    )

    parser.add_argument(
        "--k",
        type=int,
        default=2,
        help="Кількість повторень: 2 або 3",
    )

    parser.add_argument(
        "--provider",
        choices=[
            "mock",
            "ollama",
            "recorded",
            "openai_compat",
        ],
        default="mock",
    )

    parser.add_argument(
        "--live",
        action="store_true",
        help="Дозволити виклики справжньої моделі",
    )

    parser.add_argument(
        "--record",
        action="store_true",
        help="Записати відповіді справжньої моделі",
    )

    args = parser.parse_args()

    # Перевірка параметрів запуску
    if args.k not in (2, 3):
        parser.error(
            "Кількість повторень повинна бути 2 або 3"
        )

    if (
        args.provider in {"ollama", "openai_compat"}
        and not args.live
    ):
        parser.error(
            "Для справжньої моделі додай --live"
        )

    if args.record and not args.live:
        parser.error(
            "Для запису відповідей потрібен --live"
        )

    # Завантажуємо налаштування
    settings = load_settings()

    provider_mode = args.provider

    if args.record:
        provider_mode = "recorded"

        settings = replace(
            settings,
            llm_provider="recorded",
            record_mode="record",
        )
    else:
        settings = replace(
            settings,
            llm_provider=provider_mode,
        )

    # Mock використовує власну назву моделі
    if provider_mode == "mock":
        settings = replace(
            settings,
            llm_model="mock-study-v1",
        )

    settings.validate()
    prompt_metadata, _ = read_template(settings)
    prompt_version = prompt_metadata["prompt_version"]    

    # Створюємо застосунок
    app = create_app(settings)

    # Читаємо тестові приклади
    cases_path = Path(args.cases)

    if not cases_path.is_file():
        raise FileNotFoundError(
            f"Не знайдено файл: {cases_path}"
        )

    cases = json.loads(
        cases_path.read_text(
            encoding="utf-8"
        )
    )

    if not isinstance(cases, list):
        raise ValueError(
            "eval/cases.json повинен містити список прикладів"
        )

    # Формуємо дату запуску
    stamp = datetime.now(
        timezone.utc
    ).strftime("%Y%m%dT%H%M%SZ")

    model_label = settings.llm_model.replace(
        ":", "-"
    ).replace("/", "-")

    label = (
        f"{stamp}-"
        f"{provider_mode}-"
        f"{model_label}-{prompt_version}"
    )

    # Зберігаємо результати
    runs = []

    summary = defaultdict(
        lambda: Counter(
            total=0,
            passed=0,
            failed=0,
        )
    )

    total_expected = len(cases) * args.k

    print("=" * 55)
    print("AI STUDY ASSISTANT — ПР15")
    print("=" * 55)
    print(f"Провайдер: {provider_mode}")
    print(f"Модель: {settings.llm_model}")
    print(f"Тестових прикладів: {len(cases)}")
    print(f"Повторень: {args.k}")
    print(f"Усього перевірок: {total_expected}")
    print("=" * 55)

    # Основний цикл оцінювання
    for case in cases:

        for repetition in range(1, args.k + 1):

            request_id = str(uuid.uuid4())
            issues = []
            measurements = {}
            result_dict = None

            try:
                # Запит без HTTP, через усі рівні програми
                response = (
                    app.state.service.handle_question(
                        AskInput(**case["input"]),
                        request_id,
                    )
                )

                # Перевірка відповіді
                issues = check_case(
                    case,
                    response,
                )

                result_dict = response.model_dump()

            except (AppError, ValueError) as error:
                kind = getattr(
                    error,
                    "kind",
                    type(error).__name__,
                )

                issues = [
                    f"{kind}: {str(error)}"
                ]

            # Дані про час, токени та модель
            measurements = (
                app.state.orchestrator.measurements.get(
                    request_id,
                    {},
                )
            )

            passed = len(issues) == 0

            kind = case["kind"]

            summary[kind]["total"] += 1

            if passed:
                summary[kind]["passed"] += 1
            else:
                summary[kind]["failed"] += 1

            # Запис одного проходження
            runs.append({
                "case_id": case["id"],
                "kind": kind,
                "repetition": repetition,
                "passed": passed,
                "issues": issues,
                "result": result_dict,
                "metrics": measurements,
            })

            status = (
                "PASS"
                if passed
                else "FAIL"
            )

            print(
                f"{case['id']} | "
                f"повтор {repetition} | "
                f"{status}",
                flush=True,
            )

            if issues:
                for issue in issues:
                    print(
                        f"  Причина: {issue}",
                        flush=True,
                    )

    # Загальні результати
    total = len(runs)

    passed_count = sum(
        item["passed"]
        for item in runs
    )

    failed_count = total - passed_count

    overall = {
        "total": total,
        "passed": passed_count,
        "failed": failed_count,
    }

    # JSON-звіт
    report = {
        "date_utc": stamp,
        "provider": provider_mode,
        "model": settings.llm_model,
        "prompt_version": prompt_version,
        "repetitions": args.k,
        "overall": overall,
        "by_kind": {
            key: dict(value)
            for key, value in summary.items()
        },
        "runs": runs,
    }

    output_dir = Path("eval/runs")

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_file = output_dir / f"{label}.json"

    output_file.write_text(
        json.dumps(
            report,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    # Для справжньої моделі формуємо вимірювання
    if args.live:

        docs_dir = Path("docs")

        docs_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        measurement_lines = [
            "# ПР15 — Вимірювання AI-моделі",
            "",
            f"Дата UTC: {stamp}",
            f"Провайдер: {provider_mode}",
            f"Модель: {settings.llm_model}",
            "",
            "Модель працює в локальному середовищі.",
            "Для Ollama плата за API становить 0 USD.",
            "Витрати на обладнання не враховані.",
            "Час до першого токена не вимірювався.",
            "",
            "| Кейс | Повтор | Час (мс) | Вхідні токени | Вихідні токени |",
            "|---|---:|---:|---:|---:|",
        ]

        for item in runs[:10]:
            metrics = item["metrics"]

            measurement_lines.append(
                f"| {item['case_id']} "
                f"| {item['repetition']} "
                f"| {metrics.get('elapsed_ms', '—')} "
                f"| {metrics.get('input_tokens', '—')} "
                f"| {metrics.get('output_tokens', '—')} |"
            )

        (
            docs_dir / "measurements.md"
        ).write_text(
            "\n".join(measurement_lines) + "\n",
            encoding="utf-8",
        )

        # Аналіз результатів
        analysis_lines = [
            "# ПР15 — Аналіз оцінювання",
            "",
            f"Модель: {settings.llm_model}",
            f"Дата UTC: {stamp}",
            "",
            f"Усього перевірок: {total}",
            f"Пройдено: {passed_count}",
            f"Не пройдено: {failed_count}",
            "",
            "## Результати за категоріями",
            "",
        ]

        for kind, counts in summary.items():
            analysis_lines.append(
                f"- {kind}: "
                f"{counts['passed']} з "
                f"{counts['total']} пройдено"
            )

        analysis_lines.extend([
            "",
            "## Невдалі приклади",
            "",
        ])

        failures = [
            item
            for item in runs
            if not item["passed"]
        ]

        if failures:
            for item in failures:
                analysis_lines.append(
                    f"- {item['case_id']} "
                    f"(повтор {item['repetition']}): "
                    f"{'; '.join(item['issues'])}. "
                    "Причину потрібно проаналізувати."
                )
        else:
            analysis_lines.append(
                "Усі перевірки пройдено успішно."
            )

        (
            docs_dir / "eval_analysis.md"
        ).write_text(
            "\n".join(analysis_lines) + "\n",
            encoding="utf-8",
        )

    # Підсумок у терміналі
    print()
    print("=" * 55)
    print("ПІДСУМОК ОЦІНЮВАННЯ")
    print("=" * 55)

    print(
        f"Усього: {total}"
    )

    print(
        f"Пройдено: {passed_count}"
    )

    print(
        f"Не пройдено: {failed_count}"
    )

    print(
        f"RESULT: {passed_count}/{total} passed"
    )

    print(
        f"SAVED: {output_file}"
    )

    print("=" * 55)


if __name__ == "__main__":
    main()
