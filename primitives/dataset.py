import json
from pathlib import Path

from datasets import load_dataset
from safetensors.torch import load_file, save
import torch
from torch import Tensor
from transformers import PreTrainedTokenizerBase

from primitives.hub import download_file, upload_file

MATH_PROMPT = """\
{problem}
Please reason step by step, and put your final answer within \\boxed{{}}."""


def load_numina_prompts(
    num_samples: int,
    seed: int,
) -> list[str]:
    dataset = load_dataset("AI-MO/NuminaMath-1.5", split="train").shuffle(seed=seed)
    prompts = []
    for row in dataset:
        if row["synthetic"] or (row["question_type"] != "math-word-problem") or (row["problem_is_valid"] != "Yes"):
            continue
        prompts.append(MATH_PROMPT.format(problem=row["problem"]))
        if len(prompts) == num_samples:
            break
    return prompts


def save_generated(
    path: str,
    rows: list[dict],
    tensors: dict[str, Tensor],
) -> None:
    upload_file(f"{path}/rows.json", json.dumps(rows, indent=2).encode())
    upload_file(f"{path}/outputs.safetensors", save(tensors))


def load_generated(
    path: str,
    tokenizer: PreTrainedTokenizerBase,
    max_sequence_len: int,
) -> list[Tensor]:
    rows = json.loads(Path(download_file(f"{path}/rows.json")).read_text())
    tensors = load_file(download_file(f"{path}/outputs.safetensors"))
    output_ids = tensors["output_ids"].split(tensors["output_lens"].tolist())

    messages = [[{"role": "user", "content": row["prompt"]}] for row in rows]
    prompt_ids = [tokenizer.apply_chat_template(message, add_generation_prompt=True, enable_thinking=True, return_tensors="pt")["input_ids"][0] for message in messages]
    return [torch.cat([sample_prompt_ids, sample_output_ids])[:max_sequence_len] for sample_prompt_ids, sample_output_ids in zip(prompt_ids, output_ids)]


def load_math500() -> list[tuple[str, str]]:
    dataset = load_dataset("HuggingFaceH4/MATH-500", split="test")
    return [(MATH_PROMPT.format(problem=row["problem"]), row["answer"]) for row in dataset]
