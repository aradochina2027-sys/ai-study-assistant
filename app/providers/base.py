from abc import ABC, abstractmethod
from collections.abc import Iterator

from app.models import ProviderRequest


class BaseProvider(ABC):
    @abstractmethod
    def stream(
        self,
        request: ProviderRequest,
    ) -> Iterator[str]:
        """Повертає фрагменти відповіді AI."""