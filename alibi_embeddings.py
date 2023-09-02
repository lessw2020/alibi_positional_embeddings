# derived in part from LucidRains impl:
https://github.com/lucidrains/x-transformers/tree/main


class Alibi(nn.Module):
    def __init__(self, heads, num_heads):
        super().__init__()
        self.heads = heads
        self.num_heads = num_heads

        slopes = Tensor(self._get_slopes(heads))
        slopes = slopes.unsqueeze(-1).unsqueeze(-1)

        self.register_buffer('slopes', slopes, persistent=False)
        self.register_buffer('bias', None, persistent=False)

    def get_bias(self, i, j, device):
        # integer values between i-> j
        i_range = torch.arange(j-i, j, device=device)
        # values up to j
        j_range = torch.arange(j, device=device)
        # Expand dimensions to make them broadcastable
        j_range = j_range.unsqueeze(0).unsqueeze(0)  # Shape: (1, 1, j)
        i_range = i_range.unsqueeze(1).unsqueeze(2)  # Shape: (1, i, 1)

        bias = -torch.abs(j_range - i_range)
        return bias
