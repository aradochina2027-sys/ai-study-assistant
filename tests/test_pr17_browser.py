
"""Браузерні тести інтерфейсу StudyAI для ПР17."""

from functools import partial
from http.server import (
    SimpleHTTPRequestHandler,
    ThreadingHTTPServer,
)
from pathlib import Path
from threading import Thread

import pytest

pytest.importorskip("playwright.sync_api")

from playwright.sync_api import (
    expect,
    sync_playwright,
)


ROOT = Path(__file__).resolve().parents[1]


class QuietHandler(SimpleHTTPRequestHandler):
    """HTTP-сервер без зайвих записів у консолі."""

    def log_message(self, format, *args):
        pass


@pytest.fixture(scope="module")
def site_url():
    """
    Тимчасово відкриває локальний HTML.

    FastAPI та Ollama не потрібні:
    API-відповіді імітує Playwright.
    """

    handler = partial(
        QuietHandler,
        directory=str(ROOT),
    )

    server = ThreadingHTTPServer(
        ("127.0.0.1", 0),
        handler,
    )

    thread = Thread(
        target=server.serve_forever,
        daemon=True,
    )

    thread.start()

    try:
        yield (
            f"http://127.0.0.1:{server.server_port}"
            "/static/index.html"
        )
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=3)


@pytest.fixture
def browser_page():
    """Створює окремий браузер для кожного тесту."""

    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(
            headless=True,
        )

        context = browser.new_context(
            accept_downloads=True,
        )

        page = context.new_page()

        try:
            yield page
        finally:
            context.close()
            browser.close()


def mock_base_api(page):
    """Імітація стану сервера та схеми OpenAPI."""

    page.route(
        "**/api/health",
        lambda route: route.fulfill(
            status=200,
            json={
                "status": "ok",
                "provider": "mock",
                "model": "mock-study-v1",
                "mode": "mock",
                "extra": {
                    "rag": "disabled",
                },
            },
        ),
    )

    page.route(
        "**/openapi.json",
        lambda route: route.fulfill(
            status=200,
            json={
                "components": {
                    "schemas": {
                        "AskInput": {
                            "properties": {
                                "question": {
                                    "minLength": 1,
                                    "maxLength": 500,
                                }
                            }
                        }
                    }
                }
            },
        ),
    )


def completed_response(
    answer,
    sources=None,
    status="ok",
):
    """Готова відповідь у форматі FastAPI."""

    return {
        "request_id": "pr17-browser-001",
        "status": "completed",
        "result": {
            "status": status,
            "answer": answer,
            "sources": (
                sources
                if sources is not None
                else ["data/notes.md"]
            ),
        },
    }


def test_new_design_validation_and_theme(
    browser_page,
    site_url,
):
    """
    Перевіряє:
    - новий дизайн;
    - валідацію;
    - перемикання теми;
    - збереження теми після F5.
    """

    page = browser_page

    mock_base_api(page)
    page.goto(site_url)

    expect(
        page.locator("#heroTitle")
    ).to_contain_text(
        "Навчайся швидше"
    )

    expect(
        page.locator("#sendBtn")
    ).to_be_disabled()

    page.locator("#question").fill("   ")

    expect(
        page.locator("#sendBtn")
    ).to_be_disabled()

    page.locator("#askForm").evaluate(
        "form => form.requestSubmit()"
    )

    expect(
        page.locator("#fieldError")
    ).to_be_visible()

    page.locator("#themeToggle").click()

    assert (
        page.locator("body").get_attribute(
            "data-theme"
        )
        == "light"
    )

    page.reload()

    assert (
        page.locator("body").get_attribute(
            "data-theme"
        )
        == "light"
    )


def test_answer_sources_reload_and_key(
    browser_page,
    site_url,
):
    """
    Перевіряє:
    - запитання;
    - Idempotency-Key;
    - відповідь;
    - джерела;
    - відновлення після F5.
    """

    page = browser_page

    mock_base_api(page)

    keys = []

    def accept(route):
        keys.append(
            route.request.headers.get(
                "idempotency-key"
            )
        )

        route.fulfill(
            status=202,
            json={
                "request_id": "pr17-browser-001",
                "status": "pending",
                "status_url": (
                    "/api/requests/pr17-browser-001"
                ),
            },
        )

    page.route(
        "**/api/ask",
        accept,
    )

    page.route(
        "**/api/requests/**",
        lambda route: route.fulfill(
            status=200,
            json=completed_response(
                "git push надсилає коміти."
            ),
        ),
    )

    page.goto(site_url)

    page.locator("#question").fill(
        "Що робить git push?"
    )

    page.locator("#sendBtn").click()

    expect(
        page.locator("#resultSection")
    ).to_be_visible(timeout=8000)

    expect(
        page.locator("#answerText")
    ).to_contain_text("git push")

    expect(
        page.locator("#sources")
    ).to_contain_text("data/notes.md")

    assert len(keys) == 1
    assert keys[0]
    assert len(keys[0]) >= 8

    # Оновлення сторінки не створює
    # нового запиту до моделі.

    page.reload()

    expect(
        page.locator("#resultSection")
    ).to_be_visible(timeout=8000)

    assert len(keys) == 1


def test_network_retry_reuses_same_key(
    browser_page,
    site_url,
):
    """
    Після втрати з'єднання повторне
    надсилання має той самий ключ.
    """

    page = browser_page

    mock_base_api(page)

    keys = []

    def accept(route):
        keys.append(
            route.request.headers.get(
                "idempotency-key"
            )
        )

        if len(keys) == 1:
            route.abort("failed")
        else:
            route.fulfill(
                status=202,
                json={
                    "request_id": "pr17-browser-001",
                    "status": "pending",
                    "status_url": (
                        "/api/requests/pr17-browser-001"
                    ),
                },
            )

    page.route(
        "**/api/ask",
        accept,
    )

    page.route(
        "**/api/requests/**",
        lambda route: route.fulfill(
            status=200,
            json=completed_response(
                "Готова відповідь."
            ),
        ),
    )

    page.goto(site_url)

    page.locator("#question").fill(
        "Поясни Git"
    )

    page.locator("#sendBtn").click()

    expect(
        page.locator("#errorPanel")
    ).to_be_visible(timeout=8000)

    page.locator("#retryBtn").click()

    expect(
        page.locator("#resultSection")
    ).to_be_visible(timeout=8000)

    assert len(keys) == 2
    assert keys[0] == keys[1]


def test_no_evidence_and_xss_safe(
    browser_page,
    site_url,
):
    """
    Перевіряє:
    - відсутність підтверджених даних;
    - захист від виконання HTML/JavaScript
      у тексті відповіді та джерел.
    """

    page = browser_page

    mock_base_api(page)

    page.route(
        "**/api/ask",
        lambda route: route.fulfill(
            status=202,
            json={
                "request_id": "pr17-browser-001",
                "status": "pending",
                "status_url": (
                    "/api/requests/pr17-browser-001"
                ),
            },
        ),
    )

    unsafe_answer = (
        '<img src=x onerror="window.injected=1">'
        " Даних немає"
    )

    unsafe_source = (
        '<svg onload="window.injected=1"></svg>'
    )

    page.route(
        "**/api/requests/**",
        lambda route: route.fulfill(
            status=200,
            json=completed_response(
                unsafe_answer,
                [unsafe_source],
                status="no_evidence",
            ),
        ),
    )

    page.goto(site_url)

    page.locator("#question").fill(
        "Невідоме питання"
    )

    page.locator("#sendBtn").click()

    expect(
        page.locator("#resultSection")
    ).to_be_visible(timeout=8000)

    expect(
        page.locator("#answerStatus")
    ).to_contain_text(
        "Недостатньо підтверджених даних"
    )

    expect(
        page.locator("#answerText")
    ).to_contain_text("<img src=x")

    # HTML залишається текстом,
    # не перетворюється на елементи.

    assert (
        page.locator("#answerText img").count()
        == 0
    )

    assert (
        page.locator("#sources svg").count()
        == 0
    )

    assert page.evaluate(
        "window.injected === undefined"
    )


def test_cancel_pending_request(
    browser_page,
    site_url,
):
    """
    Перевіряє скасування запиту,
    який очікує обробки.
    """

    page = browser_page

    mock_base_api(page)

    cancelled = {
        "value": False,
    }

    page.route(
        "**/api/ask",
        lambda route: route.fulfill(
            status=202,
            json={
                "request_id": "pr17-browser-001",
                "status": "pending",
                "status_url": (
                    "/api/requests/pr17-browser-001"
                ),
            },
        ),
    )

    def request_handler(route):
        if route.request.method == "POST":
            cancelled["value"] = True

            route.fulfill(
                status=200,
                json={
                    "request_id": "pr17-browser-001",
                    "status": "cancelled",
                },
            )
        else:
            route.fulfill(
                status=200,
                json={
                    "request_id": "pr17-browser-001",
                    "status": (
                        "cancelled"
                        if cancelled["value"]
                        else "pending"
                    ),
                    "result": None,
                },
            )

    page.route(
        "**/api/requests/**",
        request_handler,
    )

    page.goto(site_url)

    page.locator("#question").fill(
        "Поясни Python"
    )

    page.locator("#sendBtn").click()

    expect(
        page.locator("#cancelBtn")
    ).to_be_visible(timeout=8000)

    page.locator("#cancelBtn").click()

    expect(
        page.locator("#statusIndicator")
    ).to_contain_text("Скасовано")

    assert cancelled["value"]
