from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass
import json
from pathlib import Path
import re

from huggingface_hub import snapshot_download
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.patches import Patch
from safetensors.torch import load_file
from torch import Tensor

from primitives.hub import get_repo_name

METHOD_COLORS = {"bf16": "tab:gray", "gptq": "tab:orange", "mxfp4": "tab:purple", "qat": "tab:green", "routing": "tab:blue"}
ENTROPY_COLORS = {"low entropy": "tab:blue", "high entropy": "tab:red"}
STOPWORDS = {"the", "an", "is", "are", "was", "were", "be", "been", "being", "am", "it", "it's", "its", "this", "that", "that's", "these", "those", "of", "to", "in", "on", "at", "by", "with", "from", "as", "into", "for", "and", "or", "we", "me", "my", "our", "us", "you", "your", "he", "she", "they", "them", "their", "his", "her", "there", "there's", "here", "which", "what", "where", "who", "how", "has", "have", "had", "do", "does", "did", "can", "could", "would", "will"}
MARKERS = {
    "doubt": ["wait", "but", "however", "hmm", "wrong"],
    "branch": ["alternatively", "maybe", "perhaps"],
    "check": ["verify", "confirm", "remember", "sure"],
}


@dataclass
class Section:
    methods: list[str]
    routing: str | None = None
    # qat 중 이 섹션에서 볼 것.
    is_qat_target: Callable[[str], bool] = lambda target: True


# README 섹션마다 figures/{섹션}에 그림을 모음.
SECTIONS = {
    "gptq": Section(["gptq"]),
    "gptq_routing": Section(["gptq"], routing="gptq"),
    "mxfp4": Section(["mxfp4"], routing="mxfp4"),
    # 엔트로피 경계를 teacher forcing으로 정한 처음 qat kd.
    "qat": Section(["gptq", "qat"], is_qat_target=lambda target: "threshold_num_samples" not in target),
    # 학습 데이터로 엔트로피 경계를 정한 off-policy와 on-policy.
    "qat_on_policy": Section(["gptq", "qat"], is_qat_target=lambda target: "threshold_num_samples" in target and "scale_learning_rate" not in target),
    # scale을 고정한 off-policy 가중 1과 scale을 학습한 lsq.
    "qat_lsq": Section(["gptq", "qat"], is_qat_target=lambda target: "threshold_num_samples" in target and "on_policy=False" in target and "high_entropy_weight=1," in target),
}


def select_section_targets(
    results: dict,
    section: Section,
) -> dict:
    return {target: result for target, result in results.items() if not target.startswith("qat/") or section.is_qat_target(target)}


def get_labels(
    targets: list[str],
) -> list[str]:
    settings = {target: target.split("/")[2].split(",") for target in targets if len(target.split("/")) > 2}
    shared = set.intersection(*[set(target_settings) for target_settings in settings.values()]) if len(settings) > 1 else set()
    # 같은 방법끼리 공통인 설정도 뺌.
    for method in {target.split("/")[0] for target in settings}:
        method_settings = [set(target_settings) for target, target_settings in settings.items() if target.split("/")[0] == method]
        if len(method_settings) > 1:
            settings.update({target: [setting for setting in target_settings if setting not in set.intersection(*method_settings)] for target, target_settings in settings.items() if target.split("/")[0] == method})
    return ["\n".join([target.split("/")[0], *[setting for setting in settings.get(target, []) if setting not in shared]]) for target in targets]


def get_method_handles(
    targets: list[str],
) -> list[Patch]:
    methods = {"bf16", *[target.split("/")[0] for target in targets]}
    return [Patch(color=color, label=method) for method, color in METHOD_COLORS.items() if method in methods]


def get_words(
    text: str,
) -> list[str]:
    # latex 명령, 한 글자 변수, stopword는 제외.
    words = re.findall(r"(?<![\\a-z])[a-z]+(?:'[a-z]+)?(?:-[a-z]{2,})*", text.lower())
    return [word for word in words if len(word) > 1 and word not in STOPWORDS]


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
    ax.set_xticks(range(len(targets)), get_labels(targets), fontsize=8)
    ax.set_ylabel(ylabel)
    # 막대 위 숫자가 들어가게 위쪽 여백을 둠.
    ax.margins(y=0.2)
    ax.autoscale_view()
    ax.set_ylim(bottom=0)


def save_figure(
    fig: Figure,
    path: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200)
    print(f"saved {path}")


def plot_words(
    axes: list[Axes],
    bf16_rates: dict[str, float],
    rates: dict[str, float],
    method: str,
    num_words: int,
    title: str,
    xlabel: str,
) -> None:
    # bf16 대비 가장 많이 늘어난 단어, 줄어든 단어, overthinking 단어를 한 줄에 그림.
    words = sorted(bf16_rates.keys() | rates.keys(), key=lambda word: rates.get(word, 0) - bf16_rates.get(word, 0), reverse=True)
    word_groups = {"increased": words[:num_words], "decreased": words[::-1][:num_words], "markers": [word for marker_words in MARKERS.values() for word in marker_words]}
    for ax, (group, group_words) in zip(axes, word_groups.items()):
        ys = range(len(group_words))
        for offset, color, method_rates in [(-0.2, METHOD_COLORS["bf16"], bf16_rates), (0.2, METHOD_COLORS[method], rates)]:
            values = [method_rates.get(word, 0) for word in group_words]
            ax.barh([y + offset for y in ys], values, height=0.4, color=color)
            for y, value in zip(ys, values):
                ax.annotate(f"{value:.2f}", (value, y + offset), xytext=(4, 0), textcoords="offset points", va="center", fontsize=8)
        ax.set_yticks(ys, group_words)
        ax.invert_yaxis()
        ax.margins(x=0.1)
        ax.set_title(f"{title}\n{group}", fontsize=10)
        ax.set_xlabel(xlabel)
