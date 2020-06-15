import torch.nn as nn
import torch
from .transformer import TransformerBlock
import pdb

class BERT(nn.Module):
    """
    BERT model : Bidirectional Encoder Representations from Transformers.
    """

    def __init__(self, vocab_size, hidden=768, n_layers=12, attn_heads=12, dropout=0.1):
        """
        :param vocab_size: vocab_size of total words
        :param hidden: BERT model hidden size
        :param n_layers: numbers of Transformer blocks(layers)
        :param attn_heads: number of attention heads
        :param dropout: dropout rate
        """

        super().__init__()
        self.hidden = hidden
        self.n_layers = n_layers
        self.attn_heads = attn_heads

        # paper noted they used 4*hidden_size for ff_network_hidden_size
        self.feed_forward_hidden = hidden * 4

        # multi-layers transformer blocks, deep network
        self.transformer_blocks = nn.ModuleList(
            [TransformerBlock(hidden, attn_heads, hidden * 4, dropout) for _ in range(n_layers)])

    def forward(self, x,freq):
        # attention masking for padded token
        # torch.ByteTensor([batch_size, 1, seq_len, seq_len)
        zero_boolean = torch.eq(x,0).all(2)
        #pdb.set_trace()
        mask = zero_boolean.clone()
        mask[zero_boolean == 0] = 1
        mask[zero_boolean == 1] = 0
        mask = mask.unsqueeze(1).repeat(1,x.size(1),1).unsqueeze(1)
        freq = freq.unsqueeze(1).repeat(1,x.size(1),1).unsqueeze(1)
        # running over multiple transformer blocks
        for transformer in self.transformer_blocks:
            x = transformer.forward(x,freq, mask)

        return x
