import os
import logging
from torch.export import export
import torchvision.models as models
from torchvision.models.mobilenetv2 import MobileNet_V2_Weights
from torchvision.models.resnet import ResNet18_Weights
from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner
from executorch.exir import to_edge_transform_and_lower

def gen_model_pte(model_name: str, sample_inputs: tuple) -> str:
    model_path = model_name + ".pte"
    if not os.path.exists(model_path):
        model = get_model(model_name)

        # Transform to ExecuTorch program(xnnpack)
        et_program = to_edge_transform_and_lower(
            export(model, sample_inputs),
            partitioner=[XnnpackPartitioner()]
        ).to_executorch()

        # Save pte
        with open(model_path, "wb") as f:
            f.write(et_program.buffer)

    return model_path



MODEL_MAP = {
    "mobilenetv2": (models.mobilenetv2.mobilenet_v2, MobileNet_V2_Weights.DEFAULT),
    "resnet18": (models.resnet18, ResNet18_Weights.DEFAULT)
}
def get_model(model_name: str):
    key = model_name.lower()
    if key not in MODEL_MAP:
        raise ValueError(f"지원하지 않는 모델: {model_name}")
    model_class, weights = MODEL_MAP[key]
    return model_class(weights=weights).eval()

def setup_logging():
    logger = logging.getLogger()

    if logger.hasHandlers():
        logger.handlers.clear()

    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s] [%(levelname)s] %(message)s'
    )