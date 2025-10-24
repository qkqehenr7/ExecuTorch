import argparse
import os
import time
import json
import logging
import torch
from executorch.runtime import Runtime
from utils import gen_model_pte, setup_logging

input_tensor: torch.Tensor = torch.randn(1, 3, 224, 224)
sample_inputs = (input_tensor,)

def benchmark(model_path: str, repeat: int):
    logging.info(f"Loading model from {model_path} ...")
    runtime = Runtime.get()
    program = runtime.load_program(model_path)
    method = program.load_method("forward")
    logging.info("Model loaded successfully. Starting warm-up...")

    method.execute([input_tensor])
    logging.info("warm-up complete. Running benchmark..")

    latencies = []
    for i in range(repeat):
        start = time.perf_counter()
        method.execute([input_tensor])
        latency = (time.perf_counter() - start) * 1000
        latencies.append(latency)
        logging.info(f"iteration {i + 1} / {repeat}: {latency: .3f} ms")

    avg_latency = sum(latencies) / len(latencies)
    logging.info("Complete benchmark...")
    result = {
        "model_name": os.path.splitext(os.path.basename(model_path))[0],
        "latency_ms_avg": round(avg_latency, 3),
        "repeat": repeat
    }
    return result

def main():
    setup_logging()
    parser = argparse.ArgumentParser(
        prog="run_bench",
        description="ExecuTorch Model Benchmarker")
    parser.add_argument("--model", required=True, help="Model name")
    parser.add_argument("--repeat", type=int, required=True, help="Repeat number")
    args = parser.parse_args()

    model_path = gen_model_pte(model_name=args.model, sample_inputs=sample_inputs)

    result = benchmark(model_path, args.repeat)
    print(json.dumps(result, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()