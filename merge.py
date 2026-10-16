from dataclasses import dataclass
import json
from pathlib import Path
import re
import shutil
from tempfile import TemporaryDirectory
from typing import Literal

from huggingface_hub import snapshot_download
from safetensors.torch import load_file, save_file
from simple_parsing import parse
import torch

from primitives.hub import download_model, make_gptq_path, make_mxfp4_path, save_merged


@dataclass
class Config:
    model_name: str = "Qwen3-0.6B"
    quantization: Literal["gptq", "mxfp4"] = "gptq"
    num_bits: int = 4
    group_size: int = 128
    num_samples: int = 128
    mxfp4_group_size: int = 32


def merge(
    bf16_path: str,
    quantized_path: str,
    directory: str,
) -> None:
    # vllm은 모델 하나만 올릴 수 있으므로 bf16과 양자화 모델을 한 모델로 합침.
    # 앞쪽 layer가 bf16, 뒤쪽이 양자화 모델이고, embedding과 마지막 norm만 공유.
    model_config = json.loads((Path(quantized_path) / "config.json").read_text())
    num_layers = model_config["num_hidden_layers"]
    model_config["architectures"] = ["RoutedQwen3ForCausalLM"]
    model_config["num_hidden_layers"] = 2 * num_layers
    model_config["max_window_layers"] = 2 * num_layers
    model_config["layer_types"] = model_config["layer_types"] * 2

    weights = {}
    for file in sorted(Path(bf16_path).glob("model*.safetensors")):
        weights.update(load_file(file))
    quantized_weights = {}
    for file in sorted(Path(quantized_path).glob("model*.safetensors")):
        quantized_weights.update(load_file(file))
    for name, tensor in quantized_weights.items():
        match = re.fullmatch(r"model\.layers\.(\d+)\.(.+)", name)
        if match:
            weights[f"model.layers.{int(match[1]) + num_layers}.{match[2]}"] = tensor
        else:
            assert torch.equal(tensor, weights[name]), name

    (Path(directory) / "config.json").write_text(json.dumps(model_config, indent=2))
    shutil.copy(Path(bf16_path) / "generation_config.json", Path(directory) / "generation_config.json")
    save_file(weights, Path(directory) / "model.safetensors")
    print(f"merged {len(weights)} tensors num_hidden_layers {model_config['num_hidden_layers']}")


def main() -> None:
    config = parse(Config)
    if config.quantization == "gptq":
        quantized_path = make_gptq_path(config.model_name, config.num_bits, config.group_size, config.num_samples)
    elif config.quantization == "mxfp4":
        quantized_path = make_mxfp4_path(config.model_name, config.mxfp4_group_size)
    with TemporaryDirectory() as directory:
        merge(snapshot_download(f"Qwen/{config.model_name}"), download_model(quantized_path), directory)
        save_merged(f"merge/{quantized_path}", directory)


if __name__ == "__main__":
    main()
