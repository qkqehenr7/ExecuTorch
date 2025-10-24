import time
import torch
import numpy as np
from executorch.runtime import Runtime
from utils import gen_model_pte, get_model

# sample data
input_tensor: torch.Tensor = torch.randn(1, 3, 224, 224)
sample_inputs = (input_tensor,)
MODEL_NAME = "MobileNetV2"
TRIAL = 100
def test_model_verification():

    model_path = gen_model_pte(model_name=MODEL_NAME, sample_inputs=sample_inputs)

    model = get_model(MODEL_NAME)

    # ExecuTorch Model load
    runtime = Runtime.get()
    program = runtime.load_program(model_path)
    method = program.load_method("forward")

    # Warmup
    model(input_tensor)
    method.execute([input_tensor])

    # Calculating MAD
    eager_reference_output = model(input_tensor)
    executorch_output = method.execute([input_tensor])[0]
    mad_value = torch.mean(torch.abs(eager_reference_output - executorch_output)).item()

    pytorch_latencies, executorch_latencies  = [], []

    ## Calculating Latency
    for _ in range(TRIAL):
        # --- PyTorch ---
        start = time.perf_counter()
        model(input_tensor)
        pytorch_latencies.append((time.perf_counter() - start) * 1000)

        # --- ExecuTorch ---
        start = time.perf_counter()
        method.execute([input_tensor])
        executorch_latencies.append((time.perf_counter() - start) * 1000)

    pytorch_avg, pytorch_max = np.mean(pytorch_latencies), np.max(pytorch_latencies)
    executorch_avg, executorch_max = np.mean(executorch_latencies), np.max(executorch_latencies)

    print()
    print(f"model_name: {MODEL_NAME}")
    print(f"mean_absolute_difference: {mad_value: .8f}")
    print(f"pytorch_latency: avg {pytorch_avg:.2f} ms, max {pytorch_max:.2f} ms")
    print(f"executorch_latency: avg {executorch_avg:.2f} ms, max {executorch_max:.2f} ms")