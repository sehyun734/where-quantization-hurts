from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
from simple_parsing import parse
from transformers import AutoTokenizer

from primitives.plot import get_labels, get_method_handles, get_words, load_tensor_results, plot_words, save_figure


@dataclass
class Config:
    model_name: str = "Qwen3-0.6B"
    seed: int = 0
    high_entropy_ratio: float = 0.2
    num_words: int = 15


def main() -> None:
    config = parse(Config)
    results = load_tensor_results("teacher_forcing", config.model_name, f"seed={config.seed}", ["gptq"])
    targets = list(results)
    labels = [label.replace("\n", ",") for label in get_labels(targets)]
    tokenizer = AutoTokenizer.from_pretrained(f"Qwen/{config.model_name}")
    token_words = [words[0] if len(words) == 1 else None for words in map(get_words, [tokenizer.decode([token_id]) for token_id in range(len(tokenizer))])]

    fig, axes = plt.subplots(len(targets), 3, figsize=(21, len(targets) * (1 + 0.4 * config.num_words)), layout="constrained", squeeze=False)
    for row_axes, target, label in zip(axes, targets, labels):
        result = results[target]
        method = target.split("/")[0]
        is_high = result["entropy"] > result["entropy"].quantile(1 - config.high_entropy_ratio)

        bf16_counts = Counter(token_words[token_id] for token_id in result["bf16_top1"][is_high].tolist() if token_words[token_id])
        counts = Counter(token_words[token_id] for token_id in result["top1"][is_high].tolist() if token_words[token_id])
        bf16_rates = {word: count / is_high.sum().item() * 1000 for word, count in bf16_counts.items()}
        rates = {word: count / is_high.sum().item() * 1000 for word, count in counts.items()}
        plot_words(row_axes, bf16_rates, rates, method, config.num_words, f"{label}, {is_high.sum().item()} high entropy positions", "top-1 count per 1k high entropy positions")

    handles = get_method_handles(targets)
    fig.legend(handles=handles, loc="outside upper center", ncol=len(handles), frameon=False, title=f"{config.model_name} math500 teacher forcing words, high_entropy_ratio={config.high_entropy_ratio:g}")
    save_figure(fig, Path(f"figures/gptq/{config.model_name}/teacher_forcing_words.png"))


if __name__ == "__main__":
    main()
