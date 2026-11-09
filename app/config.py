"""Application settings loaded from environment / .env file."""

from typing import Any

from pydantic import Field, field_validator
from pydantic.fields import FieldInfo
from pydantic_settings import (
    BaseSettings,
    DotEnvSettingsSource,
    EnvSettingsSource,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
)

_CSV_FIELDS: frozenset[str] = frozenset({"cors_allow_origins"})


def _split_csv_if_csv_field(
    field_name: str, value: Any
) -> tuple[Any, bool]:
    if field_name in _CSV_FIELDS and isinstance(value, str):
        return [item.strip() for item in value.split(",") if item.strip()], True
    return value, False


class _CsvEnvSettingsSource(EnvSettingsSource):
    def prepare_field_value(
        self,
        field_name: str,
        field: FieldInfo,
        value: Any,
        value_is_complex: bool,
    ) -> Any:
        parsed, was_csv = _split_csv_if_csv_field(field_name, value)
        if was_csv:
            return parsed
        return super().prepare_field_value(field_name, field, value, value_is_complex)


class _CsvDotEnvSettingsSource(DotEnvSettingsSource):
    def prepare_field_value(
        self,
        field_name: str,
        field: FieldInfo,
        value: Any,
        value_is_complex: bool,
    ) -> Any:
        parsed, was_csv = _split_csv_if_csv_field(field_name, value)
        if was_csv:
            return parsed
        return super().prepare_field_value(field_name, field, value, value_is_complex)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    fastapi_port: int = 8081
    log_level: str = "info"
    log_output: str = "stdout"

    gemini_api_key: str = ""
    extraction_model: str = "gemini-3.1-flash-lite"
    retry_wait_seconds: int = 2
    max_retries: int = 2
    gemini_timeout_ms: int = 20000
    file_poll_timeout_seconds: int = 20

    hf_token: str = ""
    embedding_model_id: str = "google/embeddinggemma-300M"

    api_key: str = ""

    verdict_tampered_threshold: float = 0.95
    verdict_suspicious_threshold: float = 0.75
    verdict_low_similarity_threshold: float = 0.55

    locales_dir: str = "./locales"
    max_files_per_request: int = 100

    cors_allow_origins: list[str] = Field(default_factory=lambda: ["*"])

    @field_validator("cors_allow_origins", mode="before")
    @classmethod
    def split_csv(cls, v: str | list[str]) -> list[str]:
        if isinstance(v, str):
            return [item.strip() for item in v.split(",") if item.strip()]
        return v

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        return (
            init_settings,
            _CsvEnvSettingsSource(settings_cls),
            _CsvDotEnvSettingsSource(settings_cls),
            file_secret_settings,
        )


settings = Settings()
