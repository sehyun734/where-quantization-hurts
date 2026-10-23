from collections.abc import Iterable
from dataclasses import dataclass
import os
from statistics import mean
import time
from typing import Literal

from simple_parsing import parse
import torch
from torch import Tensor
from torch.nn import Module
from torch.nn.functional import log_softmax
from vllm import ModelRegistry  # type: ignore
from vllm.compilation.decorators import support_torch_compile  # type: ignore
from vllm.config import VllmConfig  # type: ignore
from vllm.model_executor.layers.logits_processor import LogitsProcessor  # type: ignore
from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead  # type: ignore
from vllm.model_executor.models.qwen2 import Qwen2Model  # type: ignore
from vllm.model_executor.models.qwen3 import Qwen3DecoderLayer  # type: ignore
from vllm.model_executor.models.utils import AutoWeightsLoader, maybe_prefix  # type: ignore

from primitives.dataset import load_math500
from primitives.evaluation import evaluate_math500
from primitives.hub import load_tensors, save_result
from primitives.model import download_model, generate_vllm, load_vllm
from primitives.paths import make_gptq_path, make_mxfp4_path, make_routing_path
from primitives.utils import format_duration


@dataclass
class Config:
    model_name: str = "Qwen3-0.6B"
    quantization: Literal["gptq", "mxfp4"] = "gptq"
    num_bits: int = 4
    group_size: int = 128
    num_samples: int = 128
    mxfp4_group_size: int = 32
    route_ratio: float = 0.2
    route: Literal["high", "low", "random"] = "high"
    max_generation_len: int = 32768
    temperature: float = 0.6
    seed: int = 0


@support_torch_compile(dynamic_arg_dims={"input_ids": 0, "positions": -1, "intermediate_tensors": 0, "inputs_embeds": 0})
class MergedQwen3Model(Qwen2Model):
    def __init__(
        self,
        *,
        vllm_config: VllmConfig,
        prefix: str = "",
    ) -> None:
        super().__init__(vllm_config=vllm_config, prefix=prefix, decoder_layer_type=Qwen3DecoderLayer)

    def forward(
        self,
        input_ids: Tensor | None,
        positions: Tensor,
        intermediate_tensors: None = None,
        inputs_embeds: Tensor | None = None,
    ) -> Tensor:
        # 같은 입력을 앞쪽 bf16과 뒤쪽 양자화 모델에 각각 통과시켜 hidden 2개로 반환.
        num_layers = len(self.layers) // 2
        hiddens = []
        for layers in [self.layers[:num_layers], self.layers[num_layers:]]:
            hidden_states = self.embed_input_ids(input_ids)
            residual = None
            for layer in layers:
                hidden_states, residual = layer(positions, hidden_states, residual)
            hidden_states, _ = self.norm(hidden_states, residual)
            hiddens.append(hidden_states)
        return torch.cat(hiddens, dim=-1)


class RoutedQwen3ForCausalLM(Module):
    def __init__(
        self,
        *,
        vllm_config: VllmConfig,
        prefix: str = "",
    ) -> None:
        super().__init__()
        config = vllm_config.model_config.hf_config
        self.config = config
        self.model = MergedQwen3Model(vllm_config=vllm_config, prefix=maybe_prefix(prefix, "model"))
        self.lm_head = ParallelLMHead(config.vocab_size, config.hidden_size, prefix=maybe_prefix(prefix, "lm_head"))
        if config.tie_word_embeddings:
            self.lm_head = self.lm_head.tie_weights(self.model.embed_tokens)
        self.logits_processor = LogitsProcessor(config.vocab_size)

    def embed_input_ids(
        self,
        input_ids: Tensor,
    ) -> Tensor:
        return self.model.embed_input_ids(input_ids)

    def forward(
        self,
        input_ids: Tensor | None,
        positions: Tensor,
        intermediate_tensors: None = None,
        inputs_embeds: Tensor | None = None,
    ) -> Tensor:
        return self.model(input_ids, positions, intermediate_tensors, inputs_embeds)

    def compute_logits(
        self,
        hidden_states: Tensor,
    ) -> Tensor:
        # vllm이 샘플링 직전에 호출함.
        # hf_overrides로 받은 route에 따라 위치마다 bf16과 양자화 모델 logit 중 하나 고름.
        bf16_hiddens, quantized_hiddens = hidden_states.chunk(2, dim=-1)
        bf16_logits = self.logits_processor(self.lm_head, bf16_hiddens.contiguous())
        quantized_logits = self.logits_processor(self.lm_head, quantized_hiddens.contiguous())
        bf16_logprobs = log_softmax(bf16_logits.float(), dim=-1)
        entropy = -(bf16_logprobs.exp() * bf16_logprobs).sum(dim=-1)
        if self.config.route == "high":
            is_routed = entropy > self.config.entropy_threshold
        elif self.config.route == "low":
            is_routed = entropy < self.config.entropy_threshold
        elif self.config.route == "random":
            is_routed = torch.rand(len(hidden_states), device=hidden_states.device) < self.config.route_ratio
        return torch.where(is_routed[:, None], bf16_logits, quantized_logits)

    def load_weights(
        self,
        weights: Iterable[tuple[str, Tensor]],
    ) -> set[str]:
        return AutoWeightsLoader(self).load_weights(weights)


def main() -> None:
    # transformers로 두 모델을 번갈아 돌리면 매우 느리고, vllm은 모델 하나만 올릴 수 있음.
    # 따라서 merge.py에서 합친 모델로 위치마다 logit 2개 뽑아 route에 따라 고름.
    config = parse(Config)
    file_name = f"seed={config.seed}"
    gptq_path = make_gptq_path(config.model_name, config.num_bits, config.group_size, config.num_samples)
    if config.quantization == "gptq":
        quantized_path = gptq_path
    elif config.quantization == "mxfp4":
        quantized_path = make_mxfp4_path(config.model_name, config.mxfp4_group_size)
    # bf16으로 보낼 위치의 엔트로피 경계는 teacher forcing seed 0에서 정함.
    # hf_overrides로 모델에 넘겨 compute_logits에서 씀.
    # 엔트로피는 bf16의 것이라 양자화와 무관하므로, 양자화끼리 경계를 맞추려고 gptq 결과의 것을 씀.
    entropy = load_tensors(f"teacher_forcing/{gptq_path}", "seed=0")["entropy"]
    entropy_threshold = None
    if config.route == "high":
        entropy_threshold = entropy.quantile(1 - config.route_ratio).item()
    elif config.route == "low":
        entropy_threshold = entropy.quantile(config.route_ratio).item()
    samples = load_math500()

    # register_model은 현재 프로세스에만 적용되므로 vllm 모델도 현재 프로세스에서 돌림.
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    # 합친 모델은 vllm에 없는 구조이므로 직접 만든 class로 등록.
    ModelRegistry.register_model(RoutedQwen3ForCausalLM.__name__, RoutedQwen3ForCausalLM)
    hf_overrides = {"route": config.route, "entropy_threshold": entropy_threshold, "route_ratio": config.route_ratio}
    model, tokenizer = load_vllm(download_model(f"merge/{quantized_path}"), config.model_name, hf_overrides=hf_overrides)
    think_end_id = tokenizer.convert_tokens_to_ids("</think>")

    log_time = time.perf_counter()
    outputs = generate_vllm(model, [prompt for prompt, _ in samples], config.max_generation_len, config.temperature, config.seed)
    result = evaluate_math500(outputs, [answer for _, answer in samples], think_end_id)

    path = make_routing_path(quantized_path, config.route_ratio, config.route)
    print(f"{path} accuracy {result['accuracy']:.2%} think_len {mean(result['think_lens']):.0f} truncated {mean(result['truncateds']):.2%} time {format_duration(time.perf_counter() - log_time)}")
    save_result(path, file_name, result)


if __name__ == "__main__":
    main()
