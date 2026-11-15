from abc import ABC, abstractmethod
from collections.abc import Iterator

from app.models import ProviderRequest, ProviderResult


class BaseProvider(ABC):
    @abstractmethod
    def stream(
        self,
        request: ProviderRequest,
    ) -> Iterator[str]:
        """Потік фрагментів відповіді."""

    def complete(
        self,
        request: ProviderRequest,
    ) -> ProviderResult:
        return ProviderResult(
            text="".join(self.stream(request)),
            model=request.model,
        )