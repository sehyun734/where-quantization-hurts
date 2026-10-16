from collections import defaultdict
from dataclasses import dataclass
import time

from simple_parsing import parse
import torch
from torch import Tensor
from torch.nn.functional import log_softmax
from transformers import PreTrainedModel

from primitives.dataset import load_math500
from primitives.hub import download_model, get_pending_targets, load_result, save_tensors
from primitives.model import load_model
from primitives.utils import format_duration, format_memory


@dataclass
class Config:
    model_name: str = "Qwen3-0.6B"
    seed: int = 0
    chunk_len: int = 4096


def get_think_hiddens(
    model: PreTrainedModel,
    input_ids: Tensor,
    prompt_len: int,
) -> Tensor:
    # <think> 예측하는 위치부터 think 마지막 토큰 예측하는 위치까지의 hidden.
    hiddens = model.model(input_ids=input_ids.unsqueeze(0).to(model.device)).last_hidden_state[0]
    return hiddens[prompt_len - 1 : -1]


@torch.no_grad()
def main() -> None:
    config = parse(Config)
    file_name = f"seed={config.seed}"
    targets = [target for target in get_pending_targets("teacher_forcing", config.model_name, file_name, "safetensors") if target.startswith(("gptq/", "mxfp4/", "qat/"))]
    bf16_result = load_result(f"math500/bf16/{config.model_name}", file_name)
    prompts = [prompt for prompt, _ in load_math500()]
    bf16_model, tokenizer = load_model(f"Qwen/{config.model_name}", config.model_name)

    # bf16이 맞히고 잘리지 않은 reasoning만 입력으로 씀.
    samples = []
    for prompt, text, correct, truncated in zip(prompts, bf16_result["texts"], bf16_result["corrects"], bf16_result["truncateds"]):
        if correct and not truncated:
            prompt_ids = tokenizer.apply_chat_template([{"role": "user", "content": prompt}], add_generation_prompt=True, enable_thinking=True, return_tensors="pt")["input_ids"][0]
            think_ids = tokenizer(text.split("</think>")[0], add_special_tokens=False, return_tensors="pt")["input_ids"][0]
            samples.append((prompt_ids, think_ids))
    print(f"samples {len(samples)} positions {sum(len(think_ids) for _, think_ids in samples)}")

    for target in targets:
        model, _ = load_model(download_model(target), config.model_name)
        tensors = defaultdict(list)
        log_time = time.perf_counter()
        torch.cuda.reset_peak_memory_stats()
        for step, (prompt_ids, think_ids) in enumerate(samples):
            input_ids = torch.cat([prompt_ids, think_ids])
            bf16_hiddens = get_think_hiddens(bf16_model, input_ids, len(prompt_ids))
            hiddens = get_think_hiddens(model, input_ids, len(prompt_ids))
            for start in range(0, len(think_ids), config.chunk_len):
                bf16_logprobs = log_softmax(bf16_model.lm_head(bf16_hiddens[start : start + config.chunk_len]).float(), dim=-1)
                logprobs = log_softmax(model.lm_head(hiddens[start : start + config.chunk_len]).float(), dim=-1)
                bf16_probs = bf16_logprobs.exp()
                tensors["entropy"].append(-(bf16_probs * bf16_logprobs).sum(dim=-1).cpu())
                tensors["kl"].append((bf16_probs * (bf16_logprobs - logprobs)).sum(dim=-1).cpu())
                # 두 모델이 가장 높은 확률 준 토큰.
                # 실제 다음 토큰과 다를 수 있어 따로 저장.
                bf16_top1 = bf16_logprobs.argmax(dim=-1)
                tensors["bf16_top1"].append(bf16_top1.cpu())
                tensors["top1"].append(logprobs.argmax(dim=-1).cpu())
                # bf16 top-1에 두 모델이 준 확률.
                # 1에서 빼면 bf16 top-1 이외로 샘플링될 확률.
                tensors["bf16_prob"].append(bf16_probs.gather(-1, bf16_top1[:, None])[:, 0].cpu())
                tensors["prob"].append(logprobs.gather(-1, bf16_top1[:, None])[:, 0].exp().cpu())
            tensors["think_ids"].append(think_ids)
            tensors["think_lens"].append(torch.tensor([len(think_ids)]))

            if (step + 1) % 50 == 0:
                print(f"sample {step + 1}/{len(samples)} time {format_duration(time.perf_counter() - log_time)} memory {format_memory(torch.cuda.max_memory_allocated())}")
                log_time = time.perf_counter()
                torch.cuda.reset_peak_memory_stats()

        tensors = {name: torch.cat(values) for name, values in tensors.items()}
        print(f"{target} kl {tensors['kl'].mean():.4f} top1 mismatch {(tensors['bf16_top1'] != tensors['top1']).float().mean():.2%}")
        save_tensors(f"teacher_forcing/{target}", file_name, tensors)


if __name__ == "__main__":
    main()
