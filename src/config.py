from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

_REPO_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(_REPO_ROOT / '.env', override=False)

# Fixed extraction model (not read from AGENT_MODEL in .env).
EXTRACT_MODEL = 'gpt-4o-mini'


class PipelineSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(_REPO_ROOT / '.env'),
        env_file_encoding='utf-8',
        case_sensitive=False,
        extra='ignore',
    )

    openai_api_key: str | None = Field(default=None, validation_alias='OPENAI_API_KEY')
    openai_base_url: str | None = Field(default=None, validation_alias='BASE_URL')

    neo4j_uri: str | None = Field(default=None, validation_alias='NEO4J_URI')
    neo4j_username: str = Field(default='neo4j', validation_alias='NEO4J_USERNAME')
    neo4j_password: str | None = Field(default=None, validation_alias='NEO4J_PASSWORD')
    neo4j_database: str = Field(default='neo4j', validation_alias='NEO4J_DATABASE')


@lru_cache(maxsize=1)
def get_pipeline_settings() -> PipelineSettings:
    return PipelineSettings()


def require_openai_api_key() -> str:
    key = (get_pipeline_settings().openai_api_key or os.getenv('OPENAI_API_KEY') or '').strip()
    if not key:
        raise RuntimeError('OPENAI_API_KEY is required for template extraction')
    return key
