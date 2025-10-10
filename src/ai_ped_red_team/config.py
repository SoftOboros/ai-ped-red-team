"""Configuration utilities for the project."""

from __future__ import annotations

from functools import lru_cache
from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings sourced from environment variables or .env."""

    generator_model: str = Field(
        "openai/gpt-5-nano",
        alias="APRT_GENERATOR_MODEL",
        description="Model for variant generation.",
    )
    tester_model: str = Field(
        "openai/gpt-5-nano",
        alias="APRT_TESTER_MODEL",
        description="Model used during runs.",
    )
    analyst_model: str = Field(
        "openai/gpt-5-nano",
        alias="APRT_ANALYST_MODEL",
        description="Model for analysis summaries.",
    )
    request_timeout: float = Field(
        60.0, alias="APRT_REQUEST_TIMEOUT", description="Timeout in seconds for LLM calls."
    )
    max_retries: int = Field(
        3, alias="APRT_MAX_RETRIES", description="Maximum retry attempts for LLM calls."
    )
    backoff_seconds: float = Field(
        2.0, alias="APRT_BACKOFF_SECONDS", description="Initial backoff delay between retries."
    )
    reports_dir: str = Field(
        "reports", alias="APRT_REPORTS_DIR", description="Directory for run artefacts."
    )
    default_variant_count: int = Field(
        5,
        alias="APRT_DEFAULT_VARIANT_COUNT",
        description="Default number of prompt variants to request.",
    )
    openai_api_key: Optional[str] = Field(None, alias="OPENAI_API_KEY")
    anthropic_api_key: Optional[str] = Field(None, alias="ANTHROPIC_API_KEY")
    openrouter_api_key: Optional[str] = Field(None, alias="OPENROUTER_API_KEY")
    mistral_api_key: Optional[str] = Field(None, alias="MISTRAL_API_KEY")
    cohere_api_key: Optional[str] = Field(None, alias="COHERE_API_KEY")
    google_api_key: Optional[str] = Field(None, alias="GOOGLE_API_KEY")

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="allow")


@lru_cache(maxsize=1)
def load_settings() -> Settings:
    """Load settings once per process."""

    return Settings()


__all__ = ["Settings", "load_settings"]
