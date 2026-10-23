from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from statistics import mean

import matplotlib.pyplot as plt
from simple_parsing import parse

from primitives.plot import get_labels, get_method_handles, get_words, load_results, plot_words, save_figure


@dataclass
class Config:
    model_name: str = "Qwen3-0.6B"
    num_words: int = 15


def count_word_rates(
    text: str,
    think_len: int,
) -> dict[str, float]:
    counts = Counter(get_words(text.split("</think>")[0]))
    return {word: count / think_len * 1000 for word, count in counts.items()}


def main() -> None:
    config = parse(Config)
    results = load_results("math500", config.model_name, "seed=*", ["bf16", "gptq"])
    targets = [target for target in results if target.split("/")[0] != "bf16"]
    bf16_results = results[f"bf16/{config.model_name}"]
    labels = [label.replace("\n", ",") for label in get_labels(targets)]

    rows = []
    for target, label in zip(targets, labels):
        method = target.split("/")[0]
        groups = {"both correct": ([], []), f"{method} wrong": ([], [])}
        for result, bf16_result in zip(results[target], bf16_results):
            for text, think_len, correct, truncated, bf16_text, bf16_think_len, bf16_correct in zip(result["texts"], result["think_lens"], result["corrects"], result["truncateds"], bf16_result["texts"], bf16_result["think_lens"], bf16_result["corrects"]):
                if bf16_correct and not truncated:
                    word_rates, bf16_word_rates = groups["both correct" if correct else f"{method} wrong"]
                    word_rates.append(count_word_rates(text, think_len))
                    bf16_word_rates.append(count_word_rates(bf16_text, bf16_think_len))
        rows.extend((label, method, group, word_rates, bf16_word_rates) for group, (word_rates, bf16_word_rates) in groups.items())

    fig, axes = plt.subplots(len(rows), 3, figsize=(21, len(rows) * (1 + 0.4 * config.num_words)), layout="constrained", squeeze=False)
    for row_axes, (label, method, row_group, word_rates, bf16_word_rates) in zip(axes, rows):
        words = set().union(*word_rates, *bf16_word_rates)
        mean_rates = {word: mean(rates.get(word, 0) for rates in word_rates) for word in words}
        bf16_mean_rates = {word: mean(rates.get(word, 0) for rates in bf16_word_rates) for word in words}
        plot_words(row_axes, bf16_mean_rates, mean_rates, method, config.num_words, f"{label}, {len(word_rates)} {row_group} problems", "mean count per 1k reasoning tokens")

    handles = get_method_handles(targets)
    fig.legend(handles=handles, loc="outside upper center", ncol=len(handles), frameon=False, title=f"{config.model_name} math500 gptq words, truncated excluded")
    save_figure(fig, Path(f"figures/gptq/{config.model_name}/words.png"))


if __name__ == "__main__":
    main()
