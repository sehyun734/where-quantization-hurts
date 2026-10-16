from dataclasses import dataclass
import time

from simple_parsing import parse
import torch
from torch import Tensor
from torch.nn import Linear, Module
from transformers import PreTrainedModel

from primitives.dataset import load_generations
from primitives.hub import make_calibration_path, make_gptq_path, save_quantized
from primitives.model import load_model
from primitives.quantization import find_grid, round_to_grid
from primitives.utils import format_duration, format_memory


@dataclass
class Config:
    model_name: str = "Qwen3-0.6B"
    num_bits: int = 4
    group_size: int = 128
    block_size: int = 128
    damp_ratio: float = 0.01
    num_samples: int = 128
    max_sequence_len: int = 2048


class StopForward(Exception):
    pass


def capture_layer_inputs(
    model: PreTrainedModel,
    layer: Module,
    samples: list[Tensor],
) -> list[tuple[tuple, dict]]:
    inputs = []

    def hook(
        module: Module,
        args: tuple,
        kwargs: dict,
    ) -> None:
        inputs.append((args, kwargs))
        raise StopForward

    handle = layer.register_forward_pre_hook(hook, with_kwargs=True)
    for input_ids in samples:
        try:
            model(input_ids.unsqueeze(0).to(model.device), use_cache=False)
        except StopForward:
            pass
    handle.remove()
    return inputs


def compute_hessian(
    layer: Module,
    linear: Linear,
    inputs: list[tuple[tuple, dict]],
) -> Tensor:
    hessian = torch.zeros(linear.in_features, linear.in_features, device=linear.weight.device)

    def hook(
        module: Module,
        args: tuple,
    ) -> None:
        x = args[0].reshape(-1, args[0].shape[-1]).float()
        hessian.add_(x.T @ x)

    handle = linear.register_forward_pre_hook(hook)
    for args, kwargs in inputs:
        layer(*args, **kwargs)
    handle.remove()
    return hessian


def quantize_weight(
    weight: Tensor,
    hessian: Tensor,
    num_bits: int,
    group_size: int,
    block_size: int,
    damp_ratio: float,
) -> tuple[Tensor, Tensor, Tensor]:
    weight = weight.float().clone()
    is_dead = hessian.diagonal() == 0
    weight[:, is_dead] = 0
    hessian.diagonal()[is_dead] = 1
    hessian.diagonal().add_(damp_ratio * hessian.diagonal().mean())
    hessian_inv = torch.linalg.cholesky(torch.cholesky_inverse(torch.linalg.cholesky(hessian)), upper=True)

    quantized = torch.zeros_like(weight)
    scale = torch.zeros(weight.shape[0], weight.shape[1] // group_size, device=weight.device)
    zero = torch.zeros(weight.shape[0], weight.shape[1] // group_size, device=weight.device)
    error = torch.zeros(weight.shape[0], block_size, device=weight.device)
    for start in range(0, weight.shape[1], block_size):
        end = min(start + block_size, weight.shape[1])
        for column in range(start, end):
            group = column // group_size
            if column % group_size == 0:
                # grid를 미리 정하면 보상으로 바뀐 weight가 반영되지 않음.
                # 따라서 group 첫 column 차례에 정함.
                scale[:, group], zero[:, group] = find_grid(weight[:, column : column + group_size], num_bits)
            quantized[:, column] = round_to_grid(weight[:, column], scale[:, group], zero[:, group], num_bits)
            error[:, column - start] = (weight[:, column] - quantized[:, column]) / hessian_inv[column, column]
            weight[:, column:end] -= torch.outer(error[:, column - start], hessian_inv[column, column:end])
        weight[:, end:] -= error[:, : end - start] @ hessian_inv[start:end, end:]
    return quantized, scale, zero


@torch.no_grad()
def main() -> None:
    config = parse(Config)
    model, tokenizer = load_model(f"Qwen/{config.model_name}", config.model_name)
    samples = load_generations(make_calibration_path(config.model_name, config.num_samples), tokenizer, config.max_sequence_len)

    grid = {}
    log_time = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    for layer_index, layer in enumerate(model.model.layers):
        inputs = capture_layer_inputs(model, layer, samples)
        for name, linear in layer.named_modules():
            if not isinstance(linear, Linear):
                continue
            hessian = compute_hessian(layer, linear, inputs)
            quantized, scale, zero = quantize_weight(linear.weight, hessian, config.num_bits, config.group_size, config.block_size, config.damp_ratio)
            linear.weight.copy_(quantized)
            grid[f"model.layers.{layer_index}.{name}.scale"] = scale
            grid[f"model.layers.{layer_index}.{name}.zero"] = zero

        print(f"layer {layer_index + 1}/{len(model.model.layers)} time {format_duration(time.perf_counter() - log_time)} memory {format_memory(torch.cuda.max_memory_allocated())}")
        log_time = time.perf_counter()
        torch.cuda.reset_peak_memory_stats()

    save_quantized(make_gptq_path(config.model_name, config.num_bits, config.group_size, config.num_samples), model, grid)


if __name__ == "__main__":
    main()
