from dataclasses import dataclass
from pathlib import Path

from matplotlib.patches import Patch
import matplotlib.pyplot as plt
from simple_parsing import parse

from primitives.plot import ENTROPY_COLORS, load_tensor_results, save_figure, style_axis


@dataclass
class Config:
    model_name: str = "Qwen3-0.6B"
    seed: int = 0
    high_entropy_ratio: float = 0.2


def main() -> None:
    config = parse(Config)
    results = load_tensor_results("teacher_forcing", config.model_name, f"seed={config.seed}", ["gptq", "qat"])
    targets = [target for target in results if "num_bits=4" in target]

    fig, ax = plt.subplots(figsize=(2 + 1.5 * len(targets), 6), layout="constrained")
    for x, target in enumerate(targets):
        result = results[target]
        is_high = result["entropy"] > result["entropy"].quantile(1 - config.high_entropy_ratio)
        group_masks = {"low entropy": ~is_high, "high entropy": is_high}
        for offset, (group, group_mask) in zip([-0.2, 0.2], group_masks.items()):
            kl = result["kl"][group_mask].mean().item()
            ax.bar(x + offset, kl, width=0.4, color=ENTROPY_COLORS[group], zorder=3)
            ax.annotate(f"{kl:.3f}", (x + offset, kl), xytext=(0, 4), textcoords="offset points", ha="center", va="bottom", fontsize=8)
    style_axis(ax, targets, "mean kl")
    ax.set_title("mean kl by bf16 entropy", fontsize=10)

    handles = [Patch(color=color, label=group) for group, color in ENTROPY_COLORS.items()]
    fig.legend(handles=handles, loc="outside upper center", ncol=len(handles), frameon=False, title=f"{config.model_name} math500 qat teacher forcing, high_entropy_ratio={config.high_entropy_ratio:g}")
    save_figure(fig, Path(f"figures/{config.model_name}/qat_teacher_forcing.png"))


if __name__ == "__main__":
    main()
