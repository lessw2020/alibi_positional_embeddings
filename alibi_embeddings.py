import torch
import torch.nn as nn
from torch import Tensor
import math


class Alibi(nn.Module):
    """Attention with Linear Biases (ALiBi)

    # Softmax(qiKT + m · [-(i - 1), ..., -2, -1, 0]),
    where m = fixed specific slope per head

    as proposed in:
    https://arxiv.org/abs/2108.12409
    Train Short, Test Long: Attention with Linear Biases
    Enables Input Length Extrapolation

    derived from Ofir Press (author) codebase:
    https://github.com/ofirpress/attention_with_linear_biases

    and LucidRains impl:
    https://github.com/lucidrains/x-transformers/tree/main


    """

    def __init__(self, head: int, num_heads: int) -> torch.Tensor:
        super().__init__()
        self.head = head
        self.num_heads = num_heads

        slopes = Tensor(self.get_slopes(head))
        slopes = slopes.unsqueeze(-1).unsqueeze(-1)

        self.register_buffer("slopes", slopes, persistent=False)
        self.register_buffer("bias", None, persistent=False)

    def get_bias(self, i: int, j: int, device: torch.device) -> torch.Tensor:
        """generate the bias matrix based on the distance between q and k"""
        # integer values between i-> j
        i_range = torch.arange(j - i, j, device=device)
        # values up to j
        j_range = torch.arange(j, device=device)
        # Expand dimensions to make them broadcastable
        j_range = j_range.unsqueeze(0).unsqueeze(0)  # Shape: (1, 1, j)
        i_range = i_range.unsqueeze(1).unsqueeze(2)  # Shape: (1, i, 1)

        bias = -torch.abs(j_range - i_range)
        return bias

    @staticmethod
    def get_slopes(num_heads):
        """for n heads, a set of slopes is the geometric sequence that starts
        2^(-8/n) and uses this same value as its ratio

        example: num_heads =4
        result: [0.25, 0.0625, 0.015625, 0.00390625]

        """

        def get_slopes_power_of_2(n):
            start = 2 ** (-(2 ** -(math.log2(n) - 3)))
            ratio = start
            return [start * ratio**i for i in range(n)]

        if math.log2(num_heads).is_integer():
            return get_slopes_power_of_2(num_heads)

        # paper authors note they only trained models that have 2^a heads for some a.
        # This has beneficial properties related to input being power of 2.
        # Closest power of 2 below is workaround for when num of heads is not power of 2

        closest_power_of_2 = 2 ** math.floor(math.log2(num_heads))
        return (
            get_slopes_power_of_2(closest_power_of_2)
            + get_slopes_power_of_2(2 * closest_power_of_2)[0::2][
                : num_heads - closest_power_of_2
            ]
        )
