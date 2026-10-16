import torch
from torch import Tensor
from torch.nn import Linear, Module
from torch.nn.functional import linear


def find_grid(
    weight: Tensor,
    num_bits: int,
) -> tuple[Tensor, Tensor]:
    weight_min = weight.amin(dim=-1).clamp(max=0)
    weight_max = weight.amax(dim=-1).clamp(min=0)
    scale = ((weight_max - weight_min) / (2**num_bits - 1)).clamp(min=1e-8)
    zero = torch.round(-weight_min / scale)
    return scale, zero


def round_to_grid(
    weight: Tensor,
    scale: Tensor,
    zero: Tensor,
    num_bits: int,
) -> Tensor:
    level = torch.clamp(torch.round(weight / scale) + zero, 0, 2**num_bits - 1)
    return (level - zero) * scale


def find_mxfp4_scale(
    weight: Tensor,
) -> Tensor:
    weight_max = weight.abs().amax(dim=-1).clamp(min=2**-125)
    return 2 ** (torch.floor(torch.log2(weight_max)) - 2)


def round_to_mxfp4_grid(
    weight: Tensor,
    scale: Tensor,
) -> Tensor:
    magnitude = (weight.abs() / scale).clamp(max=6)
    step = torch.where(magnitude < 2, 0.5, torch.where(magnitude < 4, 1.0, 2.0))
    return torch.sign(weight) * torch.round(magnitude / step) * step * scale


class STELinear(Module):
    def __init__(
        self,
        linear: Linear,
        scale: Tensor,
        zero: Tensor,
        num_bits: int,
        group_size: int,
    ) -> None:
        super().__init__()
        self.weight = linear.weight
        self.bias = linear.bias
        self.scale = scale
        self.zero = zero
        self.num_bits = num_bits
        self.group_size = group_size

    def quantize(
        self,
    ) -> Tensor:
        groups = self.weight.reshape(self.weight.shape[0], -1, self.group_size)
        quantized = round_to_grid(groups, self.scale[..., None], self.zero[..., None], self.num_bits).reshape(self.weight.shape)
        # forward는 양자화한 weight를 쓰고, backward는 gradient를 그대로 weight에 넘김.
        return self.weight + (quantized - self.weight).detach()

    def forward(
        self,
        x: Tensor,
    ) -> Tensor:
        return linear(x, self.quantize(), self.bias)
