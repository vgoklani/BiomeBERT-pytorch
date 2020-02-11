import torch.nn as nn
import torch.nn.functional as F
import torch
import pdb
import math


class Attention(nn.Module):
    """
    Compute 'Scaled Dot Product Attention
    """

    def forward(self, query, key, value, mask=None, dropout=None):
        pdb.set_trace()
        print("step 1")
        scores = torch.matmul(query, key.transpose(-2, -1)) \
                 / math.sqrt(query.size(-1))
        print("step 2")
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        print("step 3")
        p_attn = F.softmax(scores, dim=-1)
        print("step 4")
        if dropout is not None:
            p_attn = dropout(p_attn)
        print("returning")
        return torch.matmul(p_attn, value), p_attn
