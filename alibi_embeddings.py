import torch
import torch.nn as nn
import torch.nn.functional as F
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

    """

    def __init__(
        self, max_seq_len: int, num_heads: int, batch_size: int
    ) -> torch.Tensor:
        """recommended usage:  create alibi mask before transformer block loop and integrate"""
        super().__init__()

        self.num_heads = num_heads
        self.batch_size = batch_size
        self.max_seq_len = max_seq_len

        # self.causal_mask = self.build_causal_mask(self.max_seq_len, self.num_heads)
        self.alibi_mask = self.build_alibi_mask(self.max_seq_len, self.num_heads)
        self.register_buffer("alibi_mask", self.alibi_mask, persistent=False)

    def build_causal_attention_mask(self, seq_len: int, num_heads: int) -> torch.Tensor:
        causal_mask = torch.ones(seq_len, seq_len).tril()
        causal_mask = causal_mask.masked_fill(causal_mask == 0, -float("inf"))
        attn_mask = causal_mask.repeat(num_heads, 1, 1)
        return attn_mask

    def build_alibi_mask(self, seq_len: int, num_heads: int) -> torch.Tensor:
        """generate the alibi mask by computing a distance matrix multiplied by each head's m (slope)"""
        distance_matrix = torch.arange(seq_len) - torch.arange(seq_len).view(-1, 1)
        slope_per_head = Tensor(self.get_slopes(num_heads)).view(-1, 1, 1)
        alibi_mask = distance_matrix * slope_per_head
        return alibi_mask

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
    def get_slopes(num_heads: int) -> torch.Tensor:
        """for n heads, a range from (0,1) and is the geometric sequence
        that starts at 2^(-8/n) and uses this same value as its ratio

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

    @property
    def device(self):
        return next(self.buffers()).device

    def pad_at_dim(t, pad, dim=-1, value=0.0):
        dims_from_right = (-dim - 1) if dim < 0 else (t.ndim - dim - 1)
        zeros = (0, 0) * dims_from_right
        return F.pad(t, (*zeros, *pad), value=value)

    def forward(self, i, j):
        h, device = self.total_heads, self.device

        if self.bias and self.bias.shape[-1] >= j and self.bias.shape[-2] >= i:
            return self.bias[..., :i, :j]

        bias = self.get_bias(i, j, device)
        bias = bias * self.slopes

        num_heads_unalibied = h - bias.shape[0]
        bias = self.pad_at_dim(bias, (0, num_heads_unalibied), dim=0)
        self.register_buffer("bias", bias, persistent=False)

        return self.bias
