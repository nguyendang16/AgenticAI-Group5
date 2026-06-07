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


def test_judge_provider_defaults_openai(monkeypatch):
    monkeypatch.delenv('BENCHMARK_EVAL_PROVIDER', raising=False)
    monkeypatch.delenv('BENCHMARK_JUDGE_PROVIDER', raising=False)
    from benchmark.eval_llm import judge_provider
    assert judge_provider() == 'openai'


def test_faithfulness_provider_defaults_google(monkeypatch):
    monkeypatch.delenv('BENCHMARK_EVAL_PROVIDER', raising=False)
    monkeypatch.delenv('BENCHMARK_FAITHFULNESS_PROVIDER', raising=False)
    from benchmark.eval_llm import faithfulness_provider
    assert faithfulness_provider() == 'google'


def test_legacy_eval_provider_fallback(monkeypatch):
    monkeypatch.setenv('BENCHMARK_EVAL_PROVIDER', 'openai')
    monkeypatch.delenv('BENCHMARK_JUDGE_PROVIDER', raising=False)
    monkeypatch.delenv('BENCHMARK_FAITHFULNESS_PROVIDER', raising=False)
    from benchmark.eval_llm import judge_provider, faithfulness_provider
    assert judge_provider() == 'openai'
    assert faithfulness_provider() == 'openai'


def test_build_judge_model_openai(monkeypatch):
    monkeypatch.setenv('BENCHMARK_JUDGE_PROVIDER', 'openai')
    monkeypatch.setenv('OPENAI_API_KEY', 'sk-test')
    monkeypatch.setenv('BENCHMARK_JUDGE_MODEL', 'gpt-5-mini')
    from benchmark.eval_llm import build_judge_model
    from deepeval.models import GPTModel
    assert isinstance(build_judge_model(), GPTModel)


def test_faithfulness_model_name(monkeypatch):
    monkeypatch.setenv('BENCHMARK_FAITHFULNESS_MODEL', 'gemma-4-31b-it')
    from benchmark.eval_llm import faithfulness_model_name
    assert faithfulness_model_name() == 'gemma-4-31b-it'
