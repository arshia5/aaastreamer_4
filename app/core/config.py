from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    # Database
    database_url: str = (
        "postgresql+asyncpg://postgres:postgres@localhost:5432/aaastreamer"
    )

    # JWT auth
    jwt_secret: str = "change_me_to_a_long_random_string"
    jwt_algorithm: str = "HS256"
    access_token_expire_minutes: int = 30
    refresh_token_expire_days: int = 30

    # Embeddings
    embedding_dim: int = 390

    # TMDB (used to lazily enrich people biographies & movie collections)
    tmdb_api_key: str = ""

    # App
    project_name: str = "aaastreamer API"
    cors_origins: str = "*"

    @property
    def cors_origin_list(self) -> list[str]:
        if self.cors_origins.strip() == "*":
            return ["*"]
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
