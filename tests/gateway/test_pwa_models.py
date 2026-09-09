"""Managed PWA model catalog configuration and route policy."""

from __future__ import annotations

import copy

import pytest

from gateway.pwa_config import PwaHttpConfig
from gateway.pwa_models import (
    ModelCatalogUnavailable,
    ModelRouteUnavailable,
    NativeModelCatalog,
)


def catalog_config():
    return {
        "default_model_id": "daily",
        "entries": [
            {
                "model_id": "daily",
                "display_name": "Daily",
                "provider": "synthetic-primary",
                "model": "upstream/daily",
                "capabilities": {
                    "text_input": "supported",
                    "image_input": "unknown",
                    "tools": "supported",
                    "reasoning_controls": "unsupported",
                },
            },
            {
                "model_id": "careful",
                "display_name": "Careful",
                "provider": "synthetic-secondary",
                "model": "upstream/careful",
                "capabilities": {
                    "text_input": "supported",
                    "image_input": "unsupported",
                    "tools": "supported",
                    "reasoning_controls": "supported",
                },
            },
        ],
    }


def pwa_config():
    raw = {
        "enabled": True,
        "concierge_id": "synthetic",
        "allowed_hosts": ["native.test"],
        "bindings": [
            {
                "issuer": "https://issuer.invalid",
                "subject": "owner",
                "default_source_id": "telegram-owner",
                "sources": [
                    {
                        "source_id": "telegram-owner",
                        "platform": "telegram",
                        "chat_id": "chat",
                        "chat_type": "dm",
                        "user_id": "user",
                    }
                ],
            }
        ],
    }
    raw["models"] = catalog_config()
    return raw


def test_omitted_catalog_is_distinct_from_invalid_present_catalog():
    raw = pwa_config()
    del raw["models"]
    assert PwaHttpConfig.from_dict(raw).models is None
    with pytest.raises(ModelCatalogUnavailable):
        NativeModelCatalog(None, lambda _: {}).catalog()

    raw["models"] = None
    with pytest.raises(ValueError):
        PwaHttpConfig.from_dict(raw)


@pytest.mark.parametrize(
    "mutate",
    [
        lambda value: value.update({"unknown": True}),
        lambda value: value.pop("default_model_id"),
        lambda value: value.update({"default_model_id": "outside"}),
        lambda value: value["entries"].append(copy.deepcopy(value["entries"][0])),
        lambda value: value["entries"][0]["capabilities"].update({"tools": "maybe"}),
        lambda value: value["entries"][0].update({"provider": "auto route"}),
        lambda value: value["entries"][0].update({"provider": "auto"}),
    ],
)
def test_present_malformed_catalog_fails_config_parse(mutate):
    raw = pwa_config()
    mutate(raw["models"])
    with pytest.raises(ValueError):
        PwaHttpConfig.from_dict(raw)


def test_catalog_uses_opaque_ids_and_reports_each_route_truthfully():
    config = PwaHttpConfig.from_dict(pwa_config()).models

    def resolver(provider):
        if provider == "synthetic-primary":
            raise RuntimeError("synthetic credential details")
        return {"provider": provider, "api_key": "private"}

    service = NativeModelCatalog(config, resolver)
    result = service.catalog()
    assert result == {
        "schema_version": "1.0",
        "default_model_id": "daily",
        "models": [
            {
                "model_id": "daily",
                "display_name": "Daily",
                "availability": "unavailable",
                "reason": "Configured route is temporarily unavailable.",
                "capabilities": config.entries[0].capabilities.to_wire(),
            },
            {
                "model_id": "careful",
                "display_name": "Careful",
                "availability": "available",
                "capabilities": config.entries[1].capabilities.to_wire(),
            },
        ],
    }
    assert "upstream" not in str(result)
    assert "private" not in str(result)
    assert service.resolve("careful").model == "upstream/careful"
    with pytest.raises(ModelRouteUnavailable):
        service.resolve_default()


def test_default_outage_does_not_block_an_existing_healthy_selection():
    config = PwaHttpConfig.from_dict(pwa_config()).models
    service = NativeModelCatalog(
        config,
        lambda provider: (
            (_ for _ in ()).throw(RuntimeError("down"))
            if provider == "synthetic-primary"
            else {"provider": provider}
        ),
    )
    assert service.resolve("careful").model_id == "careful"
    with pytest.raises(ModelRouteUnavailable):
        service.resolve_default()
    with pytest.raises(LookupError):
        service.resolve("outside")
