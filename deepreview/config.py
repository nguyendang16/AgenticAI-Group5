from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file='.env',
        env_file_encoding='utf-8',
        case_sensitive=False,
        extra='ignore',
    )

    app_name: str = 'Review Agent OSS Backend'

    data_dir: Path = Field(default=Path('./data'))

    # OpenAI Agent SDK runtime
    openai_api_key: str | None = Field(
        default=None,
        validation_alias=AliasChoices('OPENAI_API_KEY', 'API_KEY', 'LLM_API_KEY'),
    )
    openai_base_url: str | None = Field(
        default=None,
        validation_alias=AliasChoices('BASE_URL', 'OPENAI_BASE_URL', 'LLM_BASE_URL'),
    )
    openai_use_responses_api: bool = Field(
        default=False,
        validation_alias=AliasChoices(
            'OPENAI_USE_RESPONSES_API',
            'USE_RESPONSES_API',
            'LLM_USE_RESPONSES_API',
        ),
    )
    agent_model: str = 'gpt-5.2'
    agent_temperature: float = 0.2
    agent_max_tokens: int = 4096
    agent_max_turns: int = 1000
    agent_resume_attempts: int = 2
    agent_reasoning_effort: str = 'low'  # low, medium, high, xhigh
    max_markdown_chars_to_model: int = 120000

    # Fast review: short prompt, fewer tool/LLM rounds, no paper search.
    review_fast_mode: bool = False
    review_fast_max_turns: int = 40
    review_fast_max_markdown_chars: int = 100000
    review_fast_min_annotations: int = 8

    # Submit behavior
    submit_default_wait_seconds: int = 8
    submit_poll_interval_seconds: float = 1.0
    max_pdf_bytes: int = 50 * 1024 * 1024

    # MinerU v4 upload + parse
    mineru_base_url: str = 'https://mineru.net/api/v4'
    mineru_api_token: str | None = None
    mineru_model_version: str = 'vlm'
    mineru_upload_endpoint: str = '/file-urls/batch'
    # Comma-separated endpoint templates. Must include {batch_id}
    mineru_poll_endpoint_templates: str = (
        '/extract-results/batch/{batch_id},'
        '/extract-results/{batch_id},'
        '/extract/task/{batch_id}'
    )
    mineru_poll_interval_seconds: float = 3.0
    mineru_poll_timeout_seconds: int = 900
    # Default strict mode: keep MinerU parity and fail loudly if unavailable.
    mineru_allow_local_fallback: bool = False

    # Optional external paper search/read service
    paper_search_enabled: bool = True
    paper_search_provider: str = 'deepxiv'
    paper_search_base_url: str | None = None
    paper_search_api_key: str | None = None
    paper_search_endpoint: str = '/pasa/search'
    paper_search_timeout_seconds: int = 120
    paper_search_health_endpoint: str = '/health'
    paper_search_health_timeout_seconds: int = 5

    # Recommended direct DeepXiv search provider
    deepxiv_api_base_url: str = 'https://data.rag.ac.cn'
    deepxiv_api_token: str | None = None
    deepxiv_request_timeout_seconds: int = 60
    deepxiv_retrieve_top_k: int = 8
    deepxiv_default_source: str = 'arxiv'

    paper_read_base_url: str | None = None
    paper_read_api_key: str | None = None
    paper_read_endpoint: str = '/read'
    paper_read_timeout_seconds: int = 180

    # Gates aligned with DeepReviewer finalization logic
    enable_final_gates: bool = False
    min_paper_search_calls_for_pdf_annotate: int = 3
    min_paper_search_calls_for_final: int = 3
    min_distinct_paper_queries_for_final: int = 3
    min_annotations_for_final: int = 10
    min_english_words_for_final: int = 0
    min_chinese_chars_for_final: int = 0
    force_english_output: bool = True
    ui_language: str = 'en'

    # Tier-1 evaluation: run automatically when a job reaches a terminal state.
    auto_evaluate_on_job_finish: bool = Field(
        default=True,
        validation_alias=AliasChoices('AUTO_EVALUATE_ON_JOB_FINISH', 'AUTO_EVALUATE'),
    )

    # Neo4j review-criteria knowledge graph
    review_criteria_enabled: bool = True
    review_venue: str | None = Field(default=None, validation_alias='REVIEW_VENUE')
    review_journal: str | None = Field(default=None, validation_alias='REVIEW_JOURNAL')
    review_domain: str | None = Field(default=None, validation_alias='REVIEW_DOMAIN')
    review_article_type: str | None = Field(default=None, validation_alias='REVIEW_ARTICLE_TYPE')
    review_infer_venue_from_paper: bool = Field(
        default=True,
        validation_alias='REVIEW_INFER_VENUE_FROM_PAPER',
    )
    review_criteria_json_dir: Path = Field(
        default=Path('outputs/extracted_json'),
        validation_alias='REVIEW_CRITERIA_JSON_DIR',
    )
    neo4j_uri: str | None = Field(default=None, validation_alias='NEO4J_URI')
    neo4j_username: str = Field(default='neo4j', validation_alias='NEO4J_USERNAME')
    neo4j_password: str | None = Field(default=None, validation_alias='NEO4J_PASSWORD')
    neo4j_database: str = Field(default='neo4j', validation_alias='NEO4J_DATABASE')

    # PDF export
    pdf_font_name: str = 'Helvetica'
    pdf_title_font_size: int = 15
    pdf_body_font_size: int = 10
    pdf_page_margin: int = 48
    pdf_brand_name: str = 'DILab'
    pdf_producer_name: str = 'DILab'
    pdf_logo_path: str = 'assets/branding/logo.png'

    def mineru_poll_templates(self) -> list[str]:
        templates: list[str] = []
        for item in self.mineru_poll_endpoint_templates.split(','):
            normalized = item.strip()
            if not normalized:
                continue
            templates.append(normalized)
        return templates

def apply_review_fast_profile(settings: Settings) -> Settings:
    if not settings.review_fast_mode:
        return settings
    return settings.model_copy(
        update={
            'agent_max_turns': min(settings.agent_max_turns, settings.review_fast_max_turns),
            'agent_resume_attempts': min(settings.agent_resume_attempts, 1),
            'max_markdown_chars_to_model': min(
                settings.max_markdown_chars_to_model,
                settings.review_fast_max_markdown_chars,
            ),
            'min_annotations_for_final': min(
                settings.min_annotations_for_final,
                settings.review_fast_min_annotations,
            ),
            'min_paper_search_calls_for_pdf_annotate': 0,
            'min_paper_search_calls_for_final': 0,
            'min_distinct_paper_queries_for_final': 0,
            'paper_search_enabled': False,
            'mineru_allow_local_fallback': True,
        }
    )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    settings = apply_review_fast_profile(Settings())
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    (settings.data_dir / 'jobs').mkdir(parents=True, exist_ok=True)
    return settings
