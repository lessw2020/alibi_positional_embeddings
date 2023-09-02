# derived in part from LucidRains impl:
# https://github.com/lucidrains/x-transformers/tree/main
# and the inventor / source of alibi:
# https://github.com/ofirpress/attention_with_linear_biases

import torch
import torch.nn as nn


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

    @staticmethod
    def get_slopes(num_heads):
        def get_slopes_power_of_2(n):
            start = (2**(-2**-(math.log2(n)-3)))
            ratio = start
            return [start*ratio**i for i in range(n)]

        if math.log2(num_heads).is_integer():
            return get_slopes_power_of_2(num_heads)
            
        # paper authors note they only trained models that have 2^a heads for some a.  
        # This has beneficial properties related to input being power of 2. 
        # Closest power of 2 below is workaround for when num of heads is not power of 2
        
        closest_power_of_2 = 2 ** math.floor(math.log2(num_heads))
        return get_slopes_power_of_2(closest_power_of_2) + get_slopes_power_of_2(2 * closest_power_of_2)[0::2][:num_heads-closest_power_of_2]

    # original alibi
    def get_slopes(n):
            def get_slopes_power_of_2(n):
                start = (2**(-2**-(math.log2(n)-3)))
                ratio = start
                return [start*ratio**i for i in range(n)]

            if math.log2(n).is_integer():
                return get_slopes_power_of_2(n)                   #In the paper, we only train models that have 2^a heads for some a. This function has
            else:                                                 #some good properties that only occur when the input is a power of 2. To maintain that even
                closest_power_of_2 = 2**math.floor(math.log2(n))  #when the number of heads is not a power of 2, we use this workaround. 
                return get_slopes_power_of_2(closest_power_of_2) + get_slopes(2*closest_power_of_2)[0::2][:n-closest_power_of_2]
 

