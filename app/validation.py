import re

from pydantic import ValidationError
from app.models import ModelOutput


_NUMBER = re.compile(
    r"(?<!\w)\d+(?:[.,]\d+)?(?!\w)"
)

_COMMAND = re.compile(
    r"\b(?:git|python|pip)\s+[a-z][a-z0-9-]*\b",
    re.I,
)


def check_output(
    raw: str,
    allowed_sources: list[str],
    context: str,
) -> ModelOutput:

    # 1. Перевірка JSON-схеми
    try:
        output = ModelOutput.model_validate_json(raw)

    except (ValidationError, ValueError) as error:
        raise ValueError(
            "response does not match ModelOutput JSON schema"
        ) from error

    # 2. Перевірка правил сценарію
    if not 1 <= len(output.answer.strip()) <= 500:
        raise ValueError(
            "answer must contain 1..500 characters"
        )

    if len(set(output.sources)) != len(output.sources):
        raise ValueError("duplicate source")

    if any(
        source not in allowed_sources
        for source in output.sources
    ):
        raise ValueError("unknown source")

    if output.status == "ok" and not output.sources:
        raise ValueError("missing evidence source")

    if (
        output.status == "no_evidence"
        and output.sources
    ):
        raise ValueError(
            "no_evidence must not cite sources"
        )

    # 3. Перевірка деяких тверджень
    if output.status == "ok":
        facts = context.lower()

        numbers = _NUMBER.findall(output.answer)

        if any(
            number.lower() not in facts
            for number in numbers
        ):
            raise ValueError(
                "unsupported numeric claim"
            )

        commands = _COMMAND.findall(output.answer)

        if any(
            command.lower() not in facts
            for command in commands
        ):
            raise ValueError("unsupported command")

    return output