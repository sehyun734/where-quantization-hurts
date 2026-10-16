import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizerBase
from vllm import LLM, SamplingParams  # type: ignore
from vllm.outputs import CompletionOutput  # type: ignore
from vllm.tokenizers import TokenizerLike  # type: ignore


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
