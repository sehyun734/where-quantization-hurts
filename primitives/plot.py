from collections import defaultdict
import json
from pathlib import Path
import re
from statistics import mean, median

from huggingface_hub import snapshot_download
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.patches import Patch
import matplotlib.pyplot as plt
from safetensors.torch import load_file
from torch import Tensor

from primitives.hub import get_repo_name

METHOD_COLORS = {"bf16": "tab:gray", "gptq": "tab:orange", "mxfp4": "tab:purple", "qat": "tab:green", "routing": "tab:blue"}
GRID_COLOR = "#e5e4df"
ENTROPY_COLORS = {"low entropy": "tab:blue", "high entropy": "tab:red"}
STOPWORDS = {"the", "an", "is", "are", "was", "were", "be", "been", "being", "am", "it", "it's", "its", "this", "that", "that's", "these", "those", "of", "to", "in", "on", "at", "by", "with", "from", "as", "into", "for", "and", "or", "we", "me", "my", "our", "us", "you", "your", "he", "she", "they", "them", "their", "his", "her", "there", "there's", "here", "which", "what", "where", "who", "how", "has", "have", "had", "do", "does", "did", "can", "could", "would", "will"}
MARKERS = {
    "doubt": ["wait", "but", "however", "hmm", "wrong"],
    "branch": ["alternatively", "maybe", "perhaps"],
    "check": ["verify", "confirm", "remember", "sure"],
}


def get_labels(
    targets: list[str],
) -> list[str]:
    settings = {target: target.split("/")[2].split(",") for target in targets if len(target.split("/")) > 2}
    shared = set.intersection(*[set(target_settings) for target_settings in settings.values()]) if len(settings) > 1 else set()
    return ["\n".join([target.split("/")[0], *[setting for setting in settings.get(target, []) if setting not in shared]]) for target in targets]


def is_loop(
    text: str,
) -> bool:
    # 출력 끝에서 200~150글자 사이의 50글자가 그 앞에서 5번 이상 나오면 반복으로 봄.
    # 맨 끝은 잘린 문장일 수 있어 조금 앞에서 뗌.
    return len(text) > 200 and text[:-200].count(text[-200:-150]) >= 5


def get_words(
    text: str,
) -> list[str]:
    # latex 명령, 한 글자 변수, stopword는 제외.
    words = re.findall(r"(?<![\\a-z])[a-z]+(?:'[a-z]+)?(?:-[a-z]{2,})*", text.lower())
    return [word for word in words if len(word) > 1 and word not in STOPWORDS]


def get_method_handles(
    targets: list[str],
) -> list[Patch]:
    methods = {"bf16", *[target.split("/")[0] for target in targets]}
    return [Patch(color=color, label=method) for method, color in METHOD_COLORS.items() if method in methods]


def load_results(
    entry_name: str,
    model_name: str,
    file_name: str,
    methods: list[str],
) -> dict[str, list[dict]]:
    result_dir = Path(snapshot_download(get_repo_name(), allow_patterns=[f"{entry_name}/{method}/{model_name}/*{file_name}.json" for method in methods])) / entry_name
    results = defaultdict(list)
    for method in methods:
        for file in sorted(result_dir.glob(f"{method}/{model_name}/**/{file_name}.json")):
            results[str(file.parent.relative_to(result_dir))].append(json.loads(file.read_text()))
    return dict(results)


def load_routing_results(
    quantization: str,
    model_name: str,
    file_name: str,
) -> dict[str, list[dict]]:
    result_dir = Path(snapshot_download(get_repo_name(), allow_patterns=[f"routing/{quantization}/{model_name}/*/{file_name}.json"])) / "routing" / quantization
    results = defaultdict(list)
    for file in sorted(result_dir.glob(f"{model_name}/*/{file_name}.json")):
        # 맨 앞을 method로, 세 번째를 설정으로 보므로 양자화 이름은 빼고 routing을 붙임.
        results[f"routing/{file.parent.relative_to(result_dir)}"].append(json.loads(file.read_text()))
    return dict(results)


def load_tensor_results(
    entry_name: str,
    model_name: str,
    file_name: str,
    methods: list[str],
) -> dict[str, dict[str, Tensor]]:
    result_dir = Path(snapshot_download(get_repo_name(), allow_patterns=[f"{entry_name}/{method}/{model_name}/*{file_name}.safetensors" for method in methods])) / entry_name
    return {str(file.parent.relative_to(result_dir)): load_file(file) for method in methods for file in sorted(result_dir.glob(f"{method}/{model_name}/**/{file_name}.safetensors"))}


def style_axis(
    ax: Axes,
    targets: list[str],
    ylabel: str,
) -> None:
    ax.set_xticks(range(len(targets)), get_labels(targets), fontsize=9)
    ax.set_ylabel(ylabel)
    ax.set_ylim(bottom=0)
    ax.margins(y=0.15)
    ax.grid(axis="y", color=GRID_COLOR, linewidth=0.8)
    ax.spines[["top", "right"]].set_visible(False)


def save_figure(
    fig: Figure,
    path: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200)
    print(f"saved {path}")


def plot_math500(
    results: dict[str, list[dict]],
    bf16_results: list[dict],
    title: str,
) -> Figure:
    targets = list(results)
    fig, axes = plt.subplots(1, 4, figsize=(4 * (2 + 1.5 * len(targets)), 6), layout="constrained")
    for x, target in enumerate(targets):
        color = METHOD_COLORS[target.split("/")[0]]

        accuracies = [result["accuracy"] for result in results[target]]
        truncated_ratios = [mean(result["truncateds"]) for result in results[target]]
        accuracy = mean(accuracies)
        truncated_ratio = mean(truncated_ratios)
        axes[0].bar(x, accuracy, width=0.6, color=color, zorder=3)
        axes[0].bar(x, truncated_ratio, width=0.6, bottom=1 - truncated_ratio, color="white", edgecolor="gray", hatch="//", zorder=3)
        axes[0].scatter([x] * len(accuracies), accuracies, color="black", s=12, zorder=4)
        axes[0].annotate(f"{accuracy:.1%}\ntruncated {truncated_ratio:.1%}", (x, 1), xytext=(0, 6), textcoords="offset points", ha="center", va="bottom", fontsize=9)

        think_lens = [mean(result["think_lens"]) for result in results[target]]
        think_len = mean(think_lens)
        axes[1].bar(x, think_len, width=0.6, color=color, zorder=3)
        axes[1].scatter([x] * len(think_lens), think_lens, color="black", s=12, zorder=4)
        axes[1].annotate(f"{think_len:.0f}", (x, think_len), xytext=(0, 6), textcoords="offset points", ha="center", va="bottom", fontsize=9)

        ratios = []
        for result, bf16_result in zip(results[target], bf16_results):
            problem_ratios = [think_len / bf16_think_len for think_len, bf16_think_len, correct, bf16_correct in zip(result["think_lens"], bf16_result["think_lens"], result["corrects"], bf16_result["corrects"]) if correct and bf16_correct]
            ratios.append(median(problem_ratios))
        ratio = mean(ratios)
        axes[2].bar(x, ratio, width=0.6, color=color, zorder=3)
        axes[2].scatter([x] * len(ratios), ratios, color="black", s=12, zorder=4)
        axes[2].annotate(f"{ratio:.2f}x", (x, ratio), xytext=(0, 6), textcoords="offset points", ha="center", va="bottom", fontsize=9)

        loop_ratios = [mean(is_loop(text) for text in result["texts"]) for result in results[target]]
        loop_ratio = mean(loop_ratios)
        axes[3].bar(x, loop_ratio, width=0.6, color=color, zorder=3)
        axes[3].scatter([x] * len(loop_ratios), loop_ratios, color="black", s=12, zorder=4)
        axes[3].annotate(f"{loop_ratio:.1%}", (x, loop_ratio), xytext=(0, 6), textcoords="offset points", ha="center", va="bottom", fontsize=9)

    style_axis(axes[0], targets, "accuracy")
    axes[0].set_ylim(0, 1)
    axes[0].set_title("accuracy", fontsize=10, pad=30)
    style_axis(axes[1], targets, "reasoning tokens")
    axes[1].set_title("mean reasoning tokens", fontsize=10, pad=30)
    style_axis(axes[2], targets, "reasoning tokens / bf16")
    axes[2].set_title("reasoning tokens / bf16 per problem\nmedian on both correct", fontsize=10, pad=30)
    style_axis(axes[3], targets, "loop ratio")
    axes[3].set_title("loop ratio", fontsize=10, pad=30)

    handles = get_method_handles(targets)
    handles += [Patch(facecolor="white", edgecolor="gray", hatch="//", label="truncated")]
    fig.legend(handles=handles, loc="outside upper center", ncol=len(handles), frameon=False, title=title)
    return fig


def plot_word_rates(
    ax: Axes,
    words: list[str],
    bf16_rates: dict[str, float],
    rates: dict[str, float],
    method: str,
) -> None:
    ys = range(len(words))
    for offset, color, method_rates in [(-0.2, METHOD_COLORS["bf16"], bf16_rates), (0.2, METHOD_COLORS[method], rates)]:
        values = [method_rates.get(word, 0) for word in words]
        ax.barh([y + offset for y in ys], values, height=0.4, color=color, zorder=3)
        for y, value in zip(ys, values):
            ax.annotate(f"{value:.2f}", (value, y + offset), xytext=(4, 0), textcoords="offset points", va="center", fontsize=8)
    ax.set_yticks(ys, words)
    ax.invert_yaxis()
    ax.margins(x=0.1)
    ax.grid(axis="x", color=GRID_COLOR, linewidth=0.8)
    ax.spines[["top", "right"]].set_visible(False)
