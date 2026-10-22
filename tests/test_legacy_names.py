"""Former public names stay importable from their original modules and are the
same objects as the current names."""
import pytest


def test_model_legacy_names():
    from kotodama.model.llama import KotodamaModel, KotodamaModelConfig, LuxiaBaseModel, LuxiaModelConfig

    assert LuxiaBaseModel is KotodamaModel
    assert LuxiaModelConfig is KotodamaModelConfig


def test_eval_adapter_legacy_names():
    pytest.importorskip("lm_eval")
    from kotodama.eval.generate import generate
    from kotodama.eval.lm_eval_adapter import KotodamaEvalLM, LuxiaEvalLM, kotodama_generate, luxia_generate

    assert LuxiaEvalLM is KotodamaEvalLM
    assert luxia_generate is kotodama_generate
    assert luxia_generate is generate
