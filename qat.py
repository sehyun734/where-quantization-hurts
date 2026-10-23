from dataclasses import dataclass
import math
import random
import time

from simple_parsing import parse
import torch
from torch.nn import Linear
from torch.nn.functional import log_softmax
from torch.optim import AdamW
from transformers import set_seed

from primitives.dataset import load_generations
from primitives.hub import download_model, load_grid, make_gptq_path, make_numina_path, make_qat_path, save_quantized
from primitives.model import load_model
from primitives.quantization import STELinear
from primitives.utils import format_duration, format_memory


@dataclass
class Config:
    model_name: str = "Qwen3-0.6B"
    num_bits: int = 4
    group_size: int = 128
    num_samples: int = 128
    qat_num_samples: int = 2048
    high_entropy_ratio: float = 0.2
    high_entropy_weight: float = 1.0
    learning_rate: float = 1e-4
    num_epochs: int = 1
    threshold_num_samples: int = 256
    on_policy: bool = False
    max_sequence_len: int = 8192
    batch_size: int = 8
    seed: int = 0


def main() -> None:
    config = parse(Config)
    set_seed(config.seed)
    gptq_path = make_gptq_path(config.model_name, config.num_bits, config.group_size, config.num_samples)
    bf16_model, tokenizer = load_model(f"Qwen/{config.model_name}", config.model_name)
    model, _ = load_model(download_model(gptq_path), config.model_name, dtype=torch.float32)
    samples = load_generations(make_numina_path(config.model_name, config.qat_num_samples, gptq_path if config.on_policy else None), tokenizer, config.max_sequence_len)

    # 가중을 줄 엔트로피 경계는 학습 데이터 일부의 bf16 엔트로피로 정함.
    entropies = []
    with torch.no_grad():
        for input_ids in random.sample(samples, config.threshold_num_samples):
            bf16_logprobs = log_softmax(bf16_model(input_ids=input_ids.unsqueeze(0).to(bf16_model.device)).logits[0, :-1].float(), dim=-1)
            entropies.append(-(bf16_logprobs.exp() * bf16_logprobs).sum(dim=-1).cpu())
            del bf16_logprobs
    entropy_threshold = torch.cat(entropies).quantile(1 - config.high_entropy_ratio).item()
    print(f"entropy threshold {entropy_threshold:.4f}")

    grid = load_grid(gptq_path)
    linears = [(name, linear) for name, linear in model.model.layers.named_modules() if isinstance(linear, Linear)]
    ste_linears = []
    parameters = []
    for name, linear in linears:
        scale = grid[f"model.layers.{name}.scale"].to(model.device)
        zero = grid[f"model.layers.{name}.zero"].to(model.device)
        ste_linear = STELinear(linear, scale, zero, config.num_bits, config.group_size)
        parent_name, child_name = name.rsplit(".", 1)
        setattr(model.model.layers.get_submodule(parent_name), child_name, ste_linear)
        ste_linears.append(ste_linear)
        parameters.append(ste_linear.weight.requires_grad_(True))
    optimizer = AdamW(parameters, lr=config.learning_rate, weight_decay=0)
    mean_weight = 1 - config.high_entropy_ratio + config.high_entropy_ratio * config.high_entropy_weight

    num_batches = math.ceil(len(samples) / config.batch_size)
    kls = []
    log_time = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    for epoch in range(config.num_epochs):
        random.shuffle(samples)
        for step, start in enumerate(range(0, len(samples), config.batch_size)):
            batch = samples[start : start + config.batch_size]
            num_positions = sum(len(input_ids) - 1 for input_ids in batch)
            batch_kl = 0.0

            for input_ids in batch:
                input_ids = input_ids.unsqueeze(0).to(model.device)
                with torch.no_grad():
                    bf16_logprobs = log_softmax(bf16_model(input_ids=input_ids).logits[0, :-1].float(), dim=-1)
                    bf16_probs = bf16_logprobs.exp()
                    entropy = -(bf16_probs * bf16_logprobs).sum(dim=-1)
                    del bf16_logprobs
                    weight = torch.where(entropy > entropy_threshold, config.high_entropy_weight, 1.0)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    logprobs = log_softmax(model(input_ids=input_ids).logits[0, :-1].float(), dim=-1)
                kl = -entropy - (bf16_probs * logprobs).sum(dim=-1)
                # 가중의 평균으로 나눠, 가중을 바꿔도 loss 크기가 같게 함.
                loss = (weight * kl).sum() / (num_positions * mean_weight)
                loss.backward()
                batch_kl += kl.sum().item() / num_positions

            optimizer.step()
            optimizer.zero_grad()
            kls.append(batch_kl)
            if (step + 1) % 16 == 0:
                print(f"epoch {epoch + 1}/{config.num_epochs} batch {step + 1}/{num_batches} kl {sum(kls) / len(kls):.4f} time {format_duration(time.perf_counter() - log_time)} memory {format_memory(torch.cuda.max_memory_allocated())}")
                kls = []
                log_time = time.perf_counter()
                torch.cuda.reset_peak_memory_stats()

    with torch.no_grad():
        for (name, linear), ste_linear in zip(linears, ste_linears):
            linear.weight.copy_(ste_linear.quantize())
            parent_name, child_name = name.rsplit(".", 1)
            setattr(model.model.layers.get_submodule(parent_name), child_name, linear)
    model.to(torch.bfloat16)
    save_quantized(make_qat_path(config.model_name, config.num_bits, config.group_size, config.num_samples, config.qat_num_samples, config.high_entropy_ratio, config.high_entropy_weight, config.learning_rate, config.num_epochs, config.threshold_num_samples, config.on_policy), model, grid)


if __name__ == "__main__":
    main()
