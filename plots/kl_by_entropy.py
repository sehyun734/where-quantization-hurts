from dataclasses import dataclass
from pathlib import Path

from matplotlib.patches import Patch
import matplotlib.pyplot as plt
from simple_parsing import parse

from primitives.plot import ENTROPY_COLORS, SECTIONS, load_tensor_results, save_figure, select_section_targets, style_axis


@dataclass
class Config:
    model_name: str = "Qwen3-0.6B"
    # 비우면 qat 결과가 있는 섹션을 모두 그림.
    section: str | None = None
    seed: int = 0
    high_entropy_ratio: float = 0.2


def main() -> None:
    config = parse(Config)
    for name in [config.section] if config.section else [name for name, section in SECTIONS.items() if "qat" in section.methods]:
        section = SECTIONS[name]
        results = load_tensor_results("teacher_forcing", config.model_name, f"seed={config.seed}", section.methods)
        results = select_section_targets({target: result for target, result in results.items() if "num_bits=4" in target}, section)
        if not any(target.startswith("qat/") for target in results):
            continue

        targets = list(results)
        fig, ax = plt.subplots(figsize=(2 + 1.5 * len(targets), 5), layout="constrained")
        for x, target in enumerate(targets):
            result = results[target]
            is_high = result["entropy"] > result["entropy"].quantile(1 - config.high_entropy_ratio)
            for offset, (group, group_mask) in zip([-0.2, 0.2], {"low entropy": ~is_high, "high entropy": is_high}.items()):
                kl = result["kl"][group_mask].mean().item()
                ax.bar(x + offset, kl, width=0.4, color=ENTROPY_COLORS[group])
                ax.annotate(f"{kl:.3f}", (x + offset, kl), xytext=(0, 4), textcoords="offset points", ha="center", va="bottom", fontsize=8)
        style_axis(ax, targets, "mean kl")
        handles = [Patch(color=color, label=group) for group, color in ENTROPY_COLORS.items()]
        fig.legend(handles=handles, loc="outside upper center", ncol=len(handles), frameon=False, title=f"{config.model_name} math500 {name.replace('_', ' ')} teacher forcing, high_entropy_ratio={config.high_entropy_ratio:g}")
        save_figure(fig, Path(f"figures/{name}/{config.model_name}/kl_by_entropy.png"))


if __name__ == "__main__":
    main()
