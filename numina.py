from dataclasses import dataclass
from statistics import mean
import time
from typing import Literal

from simple_parsing import parse
import torch

from primitives.dataset import load_numina_prompts, save_generated
from primitives.model import download_model, generate_vllm, load_vllm
from primitives.paths import make_gptq_path, make_numina_path
from primitives.utils import format_duration


@dataclass
class Config:
    model_name: str = "Qwen3-0.6B"
    quantization: Literal["bf16", "gptq"] = "bf16"
    num_bits: int = 4
    group_size: int = 128
    gptq_num_samples: int = 128
    num_samples: int = 128
    max_generation_len: int = 2048
    temperature: float = 0.6
    seed: int = 0


def main() -> None:
    config = parse(Config)
    prompts = load_numina_prompts(config.num_samples, config.seed)
    if config.quantization == "bf16":
        quantized_path = None
        model_path = f"Qwen/{config.model_name}"
    elif config.quantization == "gptq":
        quantized_path = make_gptq_path(config.model_name, config.num_bits, config.group_size, config.gptq_num_samples)
        model_path = download_model(quantized_path)
    model, _ = load_vllm(model_path, config.model_name)

    log_time = time.perf_counter()
    outputs = generate_vllm(model, prompts, config.max_generation_len, config.temperature, config.seed)
    output_lens = [len(output.token_ids) for output in outputs]
    print(f"generated {len(outputs)} output_len {mean(output_lens):.0f} time {format_duration(time.perf_counter() - log_time)}")

    rows = [{"prompt": prompt, "text": output.text} for prompt, output in zip(prompts, outputs)]
    tensors = {"output_ids": torch.cat([torch.tensor(output.token_ids) for output in outputs]), "output_lens": torch.tensor(output_lens)}
    save_generated(make_numina_path(config.model_name, config.num_samples, quantized_path), rows, tensors)


if __name__ == "__main__":
    main()
