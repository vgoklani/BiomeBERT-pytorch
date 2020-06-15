import torch.nn as nn
import torch.nn.functional as F
import torch
import pdb
import math


class Attention(nn.Module):
    """
    Compute 'Scaled Dot Product Attention
    """

    def forward(self, query, key, value,freq, mask=None, dropout=None):
        #pdb.set_trace()
        #print("calculating attention scores")
        scores = torch.matmul(query, key.transpose(-2, -1)) \
                 / math.sqrt(query.size(-1))
        scores = scores+torch.log(freq)
        #print("performing masked fill")
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        #print("putting scores through softmax")
        p_attn = F.softmax(scores, dim=-1)
        #print("applying dropout")
        if dropout is not None:
            p_attn = dropout(p_attn)
        #print("returning attention values")
        return torch.matmul(p_attn, value), p_attn
