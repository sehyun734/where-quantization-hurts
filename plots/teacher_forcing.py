from dataclasses import dataclass
from pathlib import Path

from matplotlib.patches import Patch
import matplotlib.pyplot as plt
from simple_parsing import parse
import torch
from torch import Tensor
from transformers import AutoTokenizer

from primitives.plot import ENTROPY_COLORS, METHOD_COLORS, get_labels, get_method_handles, load_tensor_results, save_figure

TOKEN_CLASSES = ["digit", "symbol", "alpha", "other"]


@dataclass
class Config:
    model_name: str = "Qwen3-0.6B"
    seed: int = 0
    high_entropy_ratio: float = 0.2


def get_token_classes(
    tokens: list[str],
) -> Tensor:
    classes = []
    for token in tokens:
        token = token.strip()
        if token.isdigit():
            classes.append(TOKEN_CLASSES.index("digit"))
        elif token and not any(char.isalnum() for char in token):
            classes.append(TOKEN_CLASSES.index("symbol"))
        elif token.isalpha():
            classes.append(TOKEN_CLASSES.index("alpha"))
        else:
            classes.append(TOKEN_CLASSES.index("other"))
    return torch.tensor(classes)


def main() -> None:
    config = parse(Config)
    results = load_tensor_results("teacher_forcing", config.model_name, f"seed={config.seed}", ["gptq"])
    targets = list(results)
    labels = [label.replace("\n", ",") for label in get_labels(targets)]
    tokenizer = AutoTokenizer.from_pretrained(f"Qwen/{config.model_name}")
    token_classes = get_token_classes([tokenizer.decode([token_id]) for token_id in range(len(tokenizer))])

    fig, axes = plt.subplots(len(targets), 4, figsize=(28, 6 * len(targets)), layout="constrained", squeeze=False)
    for row_axes, target, label in zip(axes, targets, labels):
        result = results[target]
        method = target.split("/")[0]
        kl = result["kl"]
        mismatch = (result["bf16_top1"] != result["top1"]).float()
        is_high = result["entropy"] > result["entropy"].quantile(1 - config.high_entropy_ratio)
        group_masks = {"low entropy": ~is_high, "high entropy": is_high}

        group_values = {
            "share of kl": [(kl[group_mask].sum() / kl.sum()).item() for group_mask in group_masks.values()],
            "top-1 mismatch": [mismatch[group_mask].mean().item() for group_mask in group_masks.values()],
        }
        for ax, (name, values) in zip(row_axes[:2], group_values.items()):
            ax.bar(range(len(group_masks)), values, width=0.6, color=list(ENTROPY_COLORS.values()))
            for x, value in enumerate(values):
                ax.annotate(f"{value:.1%}", (x, value), xytext=(0, 4), textcoords="offset points", ha="center", va="bottom", fontsize=8)
            ax.set_xticks(range(len(group_masks)), list(group_masks))
            ax.set_ylabel(name)
            ax.set_title(f"{label}\n{name}", fontsize=10)

        ax = row_axes[2]
        classes = token_classes[result["think_ids"]]
        for offset, (group, group_mask) in zip([-0.2, 0.2], group_masks.items()):
            group_mismatches = [mismatch[(classes == index) & group_mask].mean().item() for index in range(len(TOKEN_CLASSES))]
            ax.bar([x + offset for x in range(len(TOKEN_CLASSES))], group_mismatches, width=0.4, color=ENTROPY_COLORS[group])
            for x, group_mismatch in enumerate(group_mismatches):
                ax.annotate(f"{group_mismatch:.1%}", (x + offset, group_mismatch), xytext=(0, 4), textcoords="offset points", ha="center", va="bottom", fontsize=8)
        ax.set_xticks(range(len(TOKEN_CLASSES)), TOKEN_CLASSES)
        ax.set_ylabel("top-1 mismatch")
        ax.set_title(f"{label}\ntop-1 mismatch by token class", fontsize=10)

        ax = row_axes[3]
        for offset, color, probs in [(-0.2, METHOD_COLORS["bf16"], result["bf16_prob"]), (0.2, METHOD_COLORS[method], result["prob"])]:
            tail_masses = [(1 - probs[(classes == index) & ~is_high]).mean().item() for index in range(len(TOKEN_CLASSES))]
            ax.bar([x + offset for x in range(len(TOKEN_CLASSES))], tail_masses, width=0.4, color=color)
            for x, tail_mass in enumerate(tail_masses):
                ax.annotate(f"{tail_mass:.1%}", (x + offset, tail_mass), xytext=(0, 4), textcoords="offset points", ha="center", va="bottom", fontsize=8)
        ax.set_xticks(range(len(TOKEN_CLASSES)), TOKEN_CLASSES)
        ax.set_ylabel("tail mass")
        ax.set_title(f"{label}\nlow entropy tail mass by token class", fontsize=10)

    handles = [Patch(color=color, label=group) for group, color in ENTROPY_COLORS.items()]
    handles += get_method_handles(targets)
    fig.legend(handles=handles, loc="outside upper center", ncol=len(handles), frameon=False, title=f"{config.model_name} math500 gptq teacher forcing, high_entropy_ratio={config.high_entropy_ratio:g}")
    save_figure(fig, Path(f"figures/gptq/{config.model_name}/teacher_forcing.png"))


if __name__ == "__main__":
    main()
