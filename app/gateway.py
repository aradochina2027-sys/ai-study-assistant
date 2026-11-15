from app.models import ProviderRequest, ProviderResult
from app.providers.base import BaseProvider
from app.telemetry import Telemetry
from app.budget import DailyBudget


class Gateway:
    def __init__(
        self,
        provider: BaseProvider,
        telemetry: Telemetry,
        budget: DailyBudget | None = None,
    ):
        self.provider = provider
        self.telemetry = telemetry
        self.budget = budget

    def complete(
        self,
        request: ProviderRequest,
    ) -> ProviderResult:

        self.telemetry.emit(
            request.request_id,
            "gateway",
            "provider_call",
        )

        if self.budget is not None:
            self.budget.charge()

        result = self.provider.complete(request)

        self.telemetry.emit(
            request.request_id,
            "gateway",
            "provider_returned",
        )

        return result