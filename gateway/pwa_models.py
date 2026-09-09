"""Native-owned allowlisted model catalog policy for the managed PWA."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from gateway.pwa_config import PwaModelCatalogConfig, PwaModelRoute


class ModelCatalogUnavailable(RuntimeError):
    """The operator omitted the managed model catalog."""


class ModelRouteUnavailable(RuntimeError):
    """A configured route cannot currently be resolved."""


@dataclass(frozen=True)
class ResolvedModelRoute:
    model_id: str
    model: str
    runtime: dict[str, Any]


class NativeModelCatalog:
    """Resolve only opaque IDs from the explicit managed catalog.

    Provider credentials and upstream model names never enter the wire DTO.
    Route resolution is deliberately lazy so a temporary credential or provider
    outage is reported without converting a valid static catalog into bad config.
    """

    def __init__(
        self,
        config: PwaModelCatalogConfig | None,
        resolver: Callable[[str], dict[str, Any]],
    ):
        self.config = config
        self._resolver = resolver

    def _configured(self) -> PwaModelCatalogConfig:
        if self.config is None:
            raise ModelCatalogUnavailable("managed model catalog is not configured")
        return self.config

    def _resolve(self, route: PwaModelRoute) -> ResolvedModelRoute:
        try:
            runtime = self._resolver(route.provider)
        except Exception as exc:
            raise ModelRouteUnavailable("configured route is temporarily unavailable") from exc
        if not isinstance(runtime, dict):
            raise ModelRouteUnavailable("configured route is temporarily unavailable")
        # Copy before returning: a caller may attach per-execution values, while
        # the provider resolver can hand out cached dictionaries.
        return ResolvedModelRoute(route.model_id, route.model, dict(runtime))

    def resolve(self, model_id: str) -> ResolvedModelRoute:
        route = self._configured().route(model_id)
        if route is None:
            raise LookupError("model is outside the managed catalog")
        return self._resolve(route)

    def resolve_default(self) -> ResolvedModelRoute:
        config = self._configured()
        return self.resolve(config.default_model_id)

    def catalog(self) -> dict[str, Any]:
        config = self._configured()
        models = []
        for route in config.entries:
            try:
                self._resolve(route)
                availability = {"availability": "available"}
            except ModelRouteUnavailable:
                availability = {
                    "availability": "unavailable",
                    "reason": "Configured route is temporarily unavailable.",
                }
            models.append({
                "model_id": route.model_id,
                "display_name": route.display_name,
                **availability,
                "capabilities": route.capabilities.to_wire(),
            })
        return {
            "schema_version": "1.0",
            "default_model_id": config.default_model_id,
            "models": models,
        }
