from dataclasses import dataclass
from statistics import mean
import time

from simple_parsing import parse
import torch

from primitives.dataset import load_numina_prompts
from primitives.hub import make_calibration_path, save_generations
from primitives.model import generate_vllm, load_vllm
from primitives.utils import format_duration


@dataclass
class Config:
    model_name: str = "Qwen3-0.6B"
    num_samples: int = 128
    max_generation_len: int = 2048
    temperature: float = 0.6
    seed: int = 0


def main() -> None:
    config = parse(Config)
    prompts = load_numina_prompts(config.num_samples, config.seed)
    model, _ = load_vllm(f"Qwen/{config.model_name}", config.model_name)

    log_time = time.perf_counter()
    outputs = generate_vllm(model, prompts, config.max_generation_len, config.temperature, config.seed)
    output_lens = [len(output.token_ids) for output in outputs]
    print(f"generated {len(outputs)} output_len {mean(output_lens):.0f} time {format_duration(time.perf_counter() - log_time)}")

    rows = [{"prompt": prompt, "text": output.text} for prompt, output in zip(prompts, outputs)]
    tensors = {"output_ids": torch.cat([torch.tensor(output.token_ids) for output in outputs]), "output_lens": torch.tensor(output_lens)}
    save_generations(make_calibration_path(config.model_name, config.num_samples), rows, tensors)


if __name__ == "__main__":
    main()
