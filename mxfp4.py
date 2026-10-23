from dataclasses import dataclass
import time

from simple_parsing import parse
import torch
from torch.nn import Linear

from primitives.model import load_model, save_quantized
from primitives.paths import make_mxfp4_path
from primitives.quantization import find_mxfp4_scale, round_to_mxfp4_grid
from primitives.utils import format_duration, format_memory


@dataclass
class Config:
    model_name: str = "Qwen3-0.6B"
    group_size: int = 32


@torch.no_grad()
def main() -> None:
    config = parse(Config)
    model, _ = load_model(f"Qwen/{config.model_name}", config.model_name)

    grid = {}
    log_time = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    for layer_index, layer in enumerate(model.model.layers):
        for name, linear in layer.named_modules():
            if not isinstance(linear, Linear):
                continue
            groups = linear.weight.float().reshape(linear.weight.shape[0], -1, config.group_size)
            scale = find_mxfp4_scale(groups)
            linear.weight.copy_(round_to_mxfp4_grid(groups, scale[..., None]).reshape(linear.weight.shape))
            grid[f"model.layers.{layer_index}.{name}.scale"] = scale

        print(f"layer {layer_index + 1}/{len(model.model.layers)} time {format_duration(time.perf_counter() - log_time)} memory {format_memory(torch.cuda.max_memory_allocated())}")
        log_time = time.perf_counter()
        torch.cuda.reset_peak_memory_stats()

    save_quantized(make_mxfp4_path(config.model_name, config.group_size), model, grid)


if __name__ == "__main__":
    main()
