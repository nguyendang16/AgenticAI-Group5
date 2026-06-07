import pytest


def test_eval_provider_defaults_google(monkeypatch):
    monkeypatch.delenv('BENCHMARK_EVAL_PROVIDER', raising=False)
    from benchmark.eval_llm import eval_provider, eval_model_name

    assert eval_provider() == 'google'
    assert eval_model_name() == 'gemma-4-31b-it'


def test_eval_pause_default(monkeypatch):
    monkeypatch.delenv('BENCHMARK_EVAL_PAUSE_SECONDS', raising=False)
    from benchmark.eval_llm import eval_pause_seconds

    assert eval_pause_seconds() == 5


def test_build_deepeval_model_openai(monkeypatch):
    monkeypatch.setenv('BENCHMARK_EVAL_PROVIDER', 'openai')
    monkeypatch.setenv('OPENAI_API_KEY', 'sk-test')
    monkeypatch.setenv('BENCHMARK_JUDGE_MODEL', 'gpt-5-mini')
    from benchmark.eval_llm import build_deepeval_model
    from deepeval.models import GPTModel

    assert isinstance(build_deepeval_model(), GPTModel)


def test_build_deepeval_model_google(monkeypatch):
    pytest.importorskip('deepeval')
    monkeypatch.setenv('BENCHMARK_EVAL_PROVIDER', 'google')
    monkeypatch.setenv('GOOGLE_API_KEY', 'test-key')
    monkeypatch.setenv('BENCHMARK_EVAL_MODEL', 'gemma-4-31b-it')
    from benchmark.eval_llm import build_deepeval_model
    from deepeval.models import GeminiModel

    model = build_deepeval_model()
    assert isinstance(model, GeminiModel)
