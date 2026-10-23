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


def round_ste(
    x: Tensor,
) -> Tensor:
    return x + (x.round() - x).detach()


def round_to_grid(
    weight: Tensor,
    scale: Tensor,
    zero: Tensor,
    num_bits: int,
) -> Tensor:
    # torch.clamp는 backward에서 양 끝 level(4비트면 0, 15)에도 기울기 0을 줌.
    # 그래서 양 끝 level의 weight는 학습되지 않음(바깥쪽은 어차피 잘리지만, 안쪽으로도 못 움직임).
    # is_inside로 양 끝까지는 기울기를 살리고, 범위를 벗어난 경우만 막음.
    # 예)
    # 2비트(level 0~3), scale 0.3, zero 1
    # weight            [-0.3, -0.1, 0.0, 0.2, 0.6]
    # level             [   0,    1,   1,   2,   3]
    # torch.clamp 기울기  [   0,    1,   1,   1,   0]
    # is_inside 기울기    [   1,    1,   1,   1,   1]
    # 0.6이 0.8이 되면 level 4로 벗어나 3으로 잘리고, 이때부터 기울기 0.
    level = round_ste(weight / scale) + zero
    is_inside = (level >= 0) & (level <= 2**num_bits - 1)
    level = torch.where(is_inside, level, level.clamp(0, 2**num_bits - 1).detach())
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
        # 이전에는 quantized 전체를 detach해, backward에서 q = w로 봄.
        # 이는 식에 scale이 남지 않아 scale은 학습할 수 없음.
        # 그래서 detach를 round로 내림.
        # q = round(w/s) * s라 scale이 식에 남아 학습할 수 있게 됨.

        # 범위 안: dq/dw = 1, dq/ds = round(w/s) - w/s
        # 범위 밖: dq/dw = 0, dq/ds = level - zero (level은 양 끝 상수)
        quantized = round_to_grid(groups, self.scale[..., None], self.zero[..., None], self.num_bits)
        return quantized.reshape(self.weight.shape)

    def forward(
        self,
        x: Tensor,
    ) -> Tensor:
        return linear(x, self.quantize(), self.bias)
