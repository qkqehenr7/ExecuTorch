import pytest
from run_bench import benchmark
from utils import gen_model_pte
import torch
import os

# test data
input_tensor: torch.Tensor = torch.randn(1, 3, 224, 224)
sample_inputs = (input_tensor,)

SUPPORTED_MODELS = ["MobileNetV2", "ResNet18"]
UNSUPPORTED_MODELS = ["SuperMoblieV7"]

REPEAT = 5

@pytest.mark.parametrize("model_name", SUPPORTED_MODELS)
def test_benchmark_supported(model_name):
    model_path = gen_model_pte(model_name=model_name, sample_inputs=sample_inputs)
    assert os.path.exists(model_path), f" pte file not generated for {model_name}."

    result = benchmark(model_path, repeat=REPEAT)

    assert isinstance(result, dict)
    assert "model_name" in result
    assert "latency_ms_avg" in result
    assert "repeat" in result

    assert result["repeat"] == REPEAT

@pytest.mark.parametrize("model_name", UNSUPPORTED_MODELS)
def test_benchmark_unsupported(model_name):
    with pytest.raises(ValueError):
        gen_model_pte(model_name=model_name, sample_inputs=sample_inputs)
