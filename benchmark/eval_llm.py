from __future__ import annotations

import os
import time
from typing import Any

DEFAULT_EVAL_PROVIDER = 'google'
DEFAULT_EVAL_MODEL = 'gemma-4-31b-it'
DEFAULT_EVAL_PAUSE_SECONDS = 5

DEFAULT_JUDGE_PROVIDER = 'openai'
DEFAULT_FAITHFULNESS_PROVIDER = 'google'
DEFAULT_JUDGE_MODEL = 'gpt-5-mini'
DEFAULT_FAITHFULNESS_MODEL = 'gemma-4-31b-it'


def _legacy_provider() -> str | None:
    raw = os.environ.get('BENCHMARK_EVAL_PROVIDER', '').strip().lower()
    return raw or None


def judge_provider() -> str:
    explicit = os.environ.get('BENCHMARK_JUDGE_PROVIDER', '').strip().lower()
    if explicit:
        return explicit
    return _legacy_provider() or DEFAULT_JUDGE_PROVIDER


def faithfulness_provider() -> str:
    explicit = os.environ.get('BENCHMARK_FAITHFULNESS_PROVIDER', '').strip().lower()
    if explicit:
        return explicit
    return _legacy_provider() or DEFAULT_FAITHFULNESS_PROVIDER


def judge_model_name() -> str:
    return os.environ.get('BENCHMARK_JUDGE_MODEL', '').strip() or DEFAULT_JUDGE_MODEL


def faithfulness_model_name() -> str:
    return (
        os.environ.get('BENCHMARK_FAITHFULNESS_MODEL', '').strip() or DEFAULT_FAITHFULNESS_MODEL
    )


def eval_provider() -> str:
    return os.environ.get('BENCHMARK_EVAL_PROVIDER', DEFAULT_EVAL_PROVIDER).strip().lower()


def eval_model_name() -> str:
    return (
        os.environ.get('BENCHMARK_EVAL_MODEL')
        or os.environ.get('BENCHMARK_JUDGE_MODEL')
        or DEFAULT_EVAL_MODEL
    )


def eval_pause_seconds() -> float:
    raw = os.environ.get('BENCHMARK_EVAL_PAUSE_SECONDS')
    if raw is None:
        return float(DEFAULT_EVAL_PAUSE_SECONDS)
    try:
        return max(0.0, float(raw))
    except ValueError:
        return float(DEFAULT_EVAL_PAUSE_SECONDS)


def pause_between_eval_calls() -> None:
    seconds = eval_pause_seconds()
    if seconds > 0:
        time.sleep(seconds)


def build_judge_model() -> Any:
    provider = judge_provider()
    if provider == 'google':
        from deepeval.models import GeminiModel

        api_key = os.environ.get('GOOGLE_API_KEY', '').strip()
        if not api_key:
            raise RuntimeError('GOOGLE_API_KEY is required when BENCHMARK_JUDGE_PROVIDER=google')
        return GeminiModel(model=judge_model_name(), api_key=api_key, temperature=0)
    if provider == 'openai':
        from deepeval.models import GPTModel

        base_url = (
            os.environ.get('OPENAI_BASE_URL')
            or os.environ.get('BASE_URL')
            or os.environ.get('OPENAI_API_BASE')
            or None
        )
        api_key = os.environ.get('OPENAI_API_KEY') or os.environ.get('API_KEY')
        return GPTModel(
            model=judge_model_name(),
            api_key=api_key,
            base_url=base_url,
            temperature=0,
        )
    raise ValueError(f'Unsupported BENCHMARK_JUDGE_PROVIDER: {provider}')


def build_deepeval_model() -> Any:
    return build_judge_model()


def build_faithfulness_llm() -> Any:
    __import__('ragas')
    provider = faithfulness_provider()
    if provider == 'google':
        from langchain_google_genai import ChatGoogleGenerativeAI
        from ragas.llms import LangchainLLMWrapper

        api_key = os.environ.get('GOOGLE_API_KEY', '').strip()
        if not api_key:
            raise RuntimeError(
                'GOOGLE_API_KEY is required when BENCHMARK_FAITHFULNESS_PROVIDER=google'
            )
        return LangchainLLMWrapper(
            ChatGoogleGenerativeAI(
                model=faithfulness_model_name(), google_api_key=api_key, temperature=0
            )
        )
    if provider == 'openai':
        from langchain_openai import ChatOpenAI
        from ragas.llms import LangchainLLMWrapper

        return LangchainLLMWrapper(
            ChatOpenAI(model=faithfulness_model_name(), temperature=0)
        )
    raise ValueError(f'Unsupported BENCHMARK_FAITHFULNESS_PROVIDER: {provider}')


def build_ragas_llm() -> Any:
    return build_faithfulness_llm()
