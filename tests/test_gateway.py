from ai_ped_red_team.models import gateway


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

    result = gateway.llm_complete("hi", model="openai/gpt-5-nano", temperature=0.0)

    assert captured["kwargs"]["temperature"] == 1.0
    assert result.metadata["overrides"]["forced"]["temperature"] == 1.0


def test_llm_complete_respects_temperature_for_other_models(monkeypatch):
    captured = {}
    monkeypatch.setattr(gateway, "completion", _fake_completion_factory(captured))

    result = gateway.llm_complete("hi", model="openai/gpt-4o-mini", temperature=0.3)

    assert captured["kwargs"]["temperature"] == 0.3
    assert "overrides" not in result.metadata
