from collections import defaultdict
from dataclasses import dataclass
import json
from pathlib import Path
from statistics import mean, median

from huggingface_hub import snapshot_download
import matplotlib.pyplot as plt
from simple_parsing import parse

from primitives.hub import get_repo_name
from primitives.plot import METHOD_COLORS, SECTIONS, get_labels, get_method_handles, load_results, save_figure, select_section_targets, style_axis


@dataclass
class Config:
    model_name: str = "Qwen3-0.6B"
    # 비우면 결과가 있는 섹션을 모두 그림.
    section: str | None = None


def load_routing_results(
    quantization: str,
    model_name: str,
) -> dict[str, list[dict]]:
    result_dir = Path(snapshot_download(get_repo_name(), allow_patterns=[f"routing/{quantization}/{model_name}/*/seed=*.json"])) / "routing" / quantization
    results = defaultdict(list)
    for file in sorted(result_dir.glob(f"{model_name}/*/seed=*.json")):
        # 맨 앞을 method로, 세 번째를 설정으로 보므로 양자화 이름은 빼고 routing을 붙임.
        results[f"routing/{file.parent.relative_to(result_dir)}"].append(json.loads(file.read_text()))
    return dict(results)


def is_loop(
    text: str,
) -> bool:
    # 출력 끝에서 200~150글자 사이의 50글자가 그 앞에서 5번 이상 나오면 반복으로 봄.
    # 맨 끝은 잘린 문장일 수 있어 조금 앞에서 뗌.
    return len(text) > 200 and text[:-200].count(text[-200:-150]) >= 5


def main() -> None:
    config = parse(Config)
    for name in [config.section] if config.section else SECTIONS:
        section = SECTIONS[name]
        results = load_results("math500", config.model_name, "seed=*", ["bf16", *section.methods])
        if section.routing:
            results |= load_routing_results(section.routing, config.model_name)
        results = select_section_targets(results, section)
        # 섹션에서 비교할 방법의 결과가 없으면 건너뜀.
        method = "routing" if section.routing else section.methods[-1]
        if not any(target.startswith(f"{method}/") for target in results):
            continue

        targets = list(results)
        bf16_results = results[f"bf16/{config.model_name}"]
        accuracies, truncateds, ratios, loops = [], [], [], []
        for target in targets:
            accuracies.append(mean(result["accuracy"] for result in results[target]))
            truncateds.append(mean(mean(result["truncateds"]) for result in results[target]))
            # 둘 다 맞힌 문제에서 bf16 대비 길이 비율의 중앙값.
            ratios.append(mean(median(think_len / bf16_think_len for think_len, bf16_think_len, correct, bf16_correct in zip(result["think_lens"], bf16_result["think_lens"], result["corrects"], bf16_result["corrects"]) if correct and bf16_correct) for result, bf16_result in zip(results[target], bf16_results)))
            loops.append(mean(mean(is_loop(text) for text in result["texts"]) for result in results[target]))
        panels = [
            ("accuracy", accuracies, [f"{accuracy:.1%}\ntruncated {truncated:.1%}" for accuracy, truncated in zip(accuracies, truncateds)]),
            ("reasoning tokens / bf16", ratios, [f"{ratio:.2f}x" for ratio in ratios]),
            ("loop ratio", loops, [f"{loop:.1%}" for loop in loops]),
        ]

        # 막대 폭을 가장 긴 라벨 줄에 맞춤.
        target_width = max(1.1, 0.07 * max(len(line) for label in get_labels(targets) for line in label.split("\n")))
        fig, axes = plt.subplots(1, len(panels), figsize=(len(panels) * (1 + target_width * len(targets)), 5), layout="constrained")
        for ax, (ylabel, values, texts) in zip(axes, panels):
            ax.bar(range(len(targets)), values, color=[METHOD_COLORS[target.split("/")[0]] for target in targets])
            for x, (value, text) in enumerate(zip(values, texts)):
                ax.annotate(text, (x, value), xytext=(0, 4), textcoords="offset points", ha="center", va="bottom", fontsize=8)
            style_axis(ax, targets, ylabel)
        handles = get_method_handles(targets)
        fig.legend(handles=handles, loc="outside upper center", ncol=len(handles), frameon=False, title=f"{config.model_name} math500 {name.replace('_', ' ')}")
        save_figure(fig, Path(f"figures/{name}/{config.model_name}/math500.png"))


if __name__ == "__main__":
    main()
