from collections.abc import Iterator

from app.models import AppError, ProviderRequest
from app.providers.base import BaseProvider


class NotImplementedProvider(BaseProvider):
    def stream(
        self,
        request: ProviderRequest,
    ) -> Iterator[str]:

        raise AppError(
            "not_implemented",
            "Справжній провайдер "
            "буде додано в ПР15",
            503,
        )

        yield ""