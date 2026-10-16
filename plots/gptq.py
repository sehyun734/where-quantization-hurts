from dataclasses import dataclass
from pathlib import Path

from simple_parsing import parse

from primitives.plot import load_results, plot_math500, save_figure


@dataclass
class Config:
    model_name: str = "Qwen3-0.6B"


def main() -> None:
    config = parse(Config)
    results = load_results("math500", config.model_name, "seed=*", ["bf16", "gptq"])
    fig = plot_math500(results, results[f"bf16/{config.model_name}"], f"{config.model_name} math500 gptq")
    save_figure(fig, Path(f"figures/{config.model_name}/gptq.png"))


if __name__ == "__main__":
    main()
