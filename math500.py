from dataclasses import dataclass
from statistics import mean
import time

from simple_parsing import parse

from primitives.dataset import load_math500
from primitives.evaluation import evaluate_math500
from primitives.hub import get_pending_targets, save_result
from primitives.model import download_model, generate_vllm, load_vllm
from primitives.utils import format_duration


@dataclass
class Config:
    model_name: str = "Qwen3-0.6B"
    max_generation_len: int = 32768
    temperature: float = 0.6
    seed: int = 0


def main() -> None:
    config = parse(Config)
    file_name = f"seed={config.seed}"
    targets = get_pending_targets("math500", config.model_name, file_name)
    samples = load_math500()

    for target in targets:
        method = target.split("/")[0]
        if method == "bf16":
            model_path = f"Qwen/{config.model_name}"
        else:
            model_path = download_model(target)
        model, tokenizer = load_vllm(model_path, config.model_name)
        think_end_id = tokenizer.convert_tokens_to_ids("</think>")

        log_time = time.perf_counter()
        outputs = generate_vllm(model, [prompt for prompt, _ in samples], config.max_generation_len, config.temperature, config.seed)
        result = evaluate_math500(outputs, [answer for _, answer in samples], think_end_id)
        print(f"{target} accuracy {result['accuracy']:.2%} think_len {mean(result['think_lens']):.0f} truncated {mean(result['truncateds']):.2%} time {format_duration(time.perf_counter() - log_time)}")
        save_result(f"math500/{target}", file_name, result)
        model.llm_engine.engine_core.shutdown()


if __name__ == "__main__":
    main()
