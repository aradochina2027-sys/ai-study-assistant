from app.models import ProviderRequest, ProviderResult
from app.providers.base import BaseProvider
from app.telemetry import Telemetry


class Gateway:
    def __init__(
        self,
        provider: BaseProvider,
        telemetry: Telemetry,
    ):
        self.provider = provider
        self.telemetry = telemetry

    def complete(
        self,
        request: ProviderRequest,
    ) -> ProviderResult:

        self.telemetry.emit(
            request.request_id,
            "gateway",
            "provider_call",
        )

        text = "".join(
            self.provider.stream(request)
        )

        self.telemetry.emit(
            request.request_id,
            "gateway",
            "provider_returned",
        )

        return ProviderResult(
            text=text,
            model=request.model,
        )