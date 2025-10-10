from ai_ped_red_team.config import Settings
from ai_ped_red_team.models import gateway


def _settings_with(**values):
    return Settings.model_construct(**values)


def test_llm_complete_callable():
    assert hasattr(gateway, "llm_complete")


def _fake_completion_factory(captured):
    def _impl(**kwargs):
        captured["kwargs"] = kwargs
        return {
            "choices": [{"message": {"content": "mock"}}],
            "usage": {},
            "id": "fake-id",
            "provider": "fake-provider",
        }

    return _impl


def test_llm_complete_forces_gpt5_temperature(monkeypatch):
    captured = {}
    monkeypatch.setattr(gateway, "completion", _fake_completion_factory(captured))

    cfg = _settings_with(openai_api_key="test")
    result = gateway.llm_complete(
        "hi",
        model="openai/gpt-5-nano",
        temperature=0.0,
        settings=cfg,
    )

    assert captured["kwargs"]["temperature"] == 1.0
    assert result.metadata["overrides"]["forced"]["temperature"] == 1.0


def test_llm_complete_respects_temperature_for_other_models(monkeypatch):
    captured = {}
    monkeypatch.setattr(gateway, "completion", _fake_completion_factory(captured))

    cfg = _settings_with(openai_api_key="test")
    result = gateway.llm_complete(
        "hi",
        model="openai/gpt-4o-mini",
        temperature=0.3,
        settings=cfg,
    )

    assert captured["kwargs"]["temperature"] == 0.3
    assert "overrides" not in result.metadata


def test_llm_complete_disables_cohere_force_single_step(monkeypatch):
    captured = {}
    monkeypatch.setattr(gateway, "completion", _fake_completion_factory(captured))

    cfg = _settings_with(openai_api_key="test", cohere_api_key="test")

    gateway.llm_complete(
        "hi",
        model="cohere/command-r",
        temperature=0.2,
        settings=cfg,
    )

    assert captured["kwargs"]["force_single_step"] is False


def test_llm_complete_inserts_cohere_assistant_bridge(monkeypatch):
    captured = {}
    monkeypatch.setattr(gateway, "completion", _fake_completion_factory(captured))
    messages = [
        {"role": "user", "content": "one"},
        {"role": "user", "content": "two"},
        {"role": "user", "content": "final"},
    ]

    cfg = _settings_with(openai_api_key="test", cohere_api_key="test")

    gateway.llm_complete("hi", model="cohere/command-r", messages=messages, settings=cfg)

    sent = captured["kwargs"]["messages"]
    assert sent[-3:][0]["role"] == "user"
    assert sent[-2]["role"] == "assistant" and sent[-2]["content"] == " "
    assert sent[-1]["role"] == "user"


def test_llm_complete_normalizes_mistral_model(monkeypatch):
    captured = {}
    monkeypatch.setattr(gateway, "completion", _fake_completion_factory(captured))

    cfg = _settings_with(mistral_api_key="test")

    gateway.llm_complete("hi", model="mistral-medium-2505", settings=cfg)

    assert captured["kwargs"]["model"] == "mistral/mistral-medium-2505"
    assert captured["kwargs"]["custom_llm_provider"] == "mistral"
    assert captured["kwargs"]["litellm_params"]["custom_llm_provider"] == "mistral"
    assert captured["kwargs"]["litellm_params"]["api_base"] == "https://api.mistral.ai/v1"


def test_llm_complete_normalizes_openai_model(monkeypatch):
    captured = {}
    monkeypatch.setattr(gateway, "completion", _fake_completion_factory(captured))

    cfg = _settings_with(openai_api_key="test")

    gateway.llm_complete("hi", model="gpt-4o-mini", settings=cfg)

    assert captured["kwargs"]["model"] == "openai/gpt-4o-mini"
    assert captured["kwargs"]["custom_llm_provider"] == "openai"
