from typer.testing import CliRunner

from ai_ped_red_team import cli


class DummySettings:
    openai_api_key = "sk-test"
    anthropic_api_key = None
    openrouter_api_key = None
    mistral_api_key = None
    cohere_api_key = None
    google_api_key = None


def test_vendors_lists_keys(monkeypatch):
    runner = CliRunner()
    monkeypatch.setattr(cli, "load_settings", lambda: DummySettings())

    result = runner.invoke(cli.app, ["vendors"])
    assert result.exit_code == 0
    assert "openai" in result.stdout
    assert "google" in result.stdout


def test_models_wildcard(monkeypatch):
    runner = CliRunner()
    monkeypatch.setattr(cli, "load_settings", lambda: DummySettings())

    result = runner.invoke(cli.app, ["models", "*"])
    assert result.exit_code == 0
    assert "openai" in result.stdout


def test_models_specific(monkeypatch):
    runner = CliRunner()
    monkeypatch.setattr(cli, "load_settings", lambda: DummySettings())

    def _stub_models(vendor, settings, limit):
        return ["model-a", "model-b"]

    monkeypatch.setattr(cli, "_list_vendor_models", _stub_models)

    result = runner.invoke(cli.app, ["models", "openai"])
    assert result.exit_code == 0
    assert "model-a" in result.stdout
    assert "model-b" in result.stdout
