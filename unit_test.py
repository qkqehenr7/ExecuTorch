from run_bench import benchmark
from utils import gen_model_pte
import torch
import os

# test data
input_tensor: torch.Tensor = torch.randn(1, 3, 224, 224)
sample_inputs = (input_tensor,)
MODEL_NAME = "MobileNetV2"
REPEAT = 5

def test_benchmark():
    model_path = gen_model_pte(model_name=MODEL_NAME, sample_inputs=sample_inputs)
    assert os.path.exists(model_path), "Not generate pte file."

    result = benchmark(model_path, repeat=REPEAT)

    assert isinstance(result, dict)
    assert "model_name" in result
    assert "latency_ms_avg" in result
    assert "repeat" in result

    assert result["repeat"] == REPEAT
