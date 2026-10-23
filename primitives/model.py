from pathlib import Path
from tempfile import TemporaryDirectory

from huggingface_hub import snapshot_download
from safetensors.torch import load_file, save_file
import torch
from torch import Tensor
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizerBase
from vllm import LLM, SamplingParams  # type: ignore
from vllm.outputs import CompletionOutput  # type: ignore
from vllm.tokenizers import TokenizerLike  # type: ignore

from primitives.hub import download_file, get_repo_name, upload_folder


def download_model(
    path: str,
) -> str:
    # vllm은 폴더 안 safetensors를 모두 weight로 읽음.
    # 따라서 grid는 받지 않음.
    return str(Path(snapshot_download(get_repo_name(), allow_patterns=[f"{path}/*"], ignore_patterns=[f"{path}/grid.safetensors"])) / path)


def save_quantized(
    path: str,
    model: PreTrainedModel,
    grid: dict[str, Tensor],
) -> None:
    with TemporaryDirectory() as directory:
        model.save_pretrained(directory)
        save_file({name: tensor.cpu().contiguous() for name, tensor in grid.items()}, f"{directory}/grid.safetensors")
        upload_folder(path, directory)


def load_grid(
    path: str,
) -> dict[str, Tensor]:
    return load_file(download_file(f"{path}/grid.safetensors"))


def load_model(
    model_path: str,
    model_name: str,
    dtype: torch.dtype = torch.bfloat16,
) -> tuple[PreTrainedModel, PreTrainedTokenizerBase]:
    tokenizer = AutoTokenizer.from_pretrained(f"Qwen/{model_name}")
    model = AutoModelForCausalLM.from_pretrained(model_path, dtype=dtype, device_map="auto")
    model.requires_grad_(False)
    return model, tokenizer


def load_vllm(
    model_path: str,
    model_name: str,
    hf_overrides: dict | None = None,
) -> tuple[LLM, TokenizerLike]:
    model = LLM(model_path, tokenizer=f"Qwen/{model_name}", dtype="bfloat16", max_model_len=40960, hf_overrides=hf_overrides)
    return model, model.get_tokenizer()


def generate_vllm(
    model: LLM,
    prompts: list[str],
    max_generation_len: int,
    temperature: float,
    seed: int,
) -> list[CompletionOutput]:
    messages = [[{"role": "user", "content": prompt}] for prompt in prompts]
    sampling_params = SamplingParams(temperature=temperature, top_p=0.95, top_k=20, max_tokens=max_generation_len, seed=seed)
    outputs = model.chat(messages, sampling_params, use_tqdm=True, chat_template_kwargs={"enable_thinking": True})
    return [output.outputs[0] for output in outputs]
