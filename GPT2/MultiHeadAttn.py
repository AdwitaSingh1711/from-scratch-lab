# from Building an LLM from Scratch
import torch
import torch.nn as nn

class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out, context_len, dropout, num_heads, qkv_bias=False):
        super().__init__()

        assert (d_out % num_heads == 0), \
            "d_out must be divisible by num_heads"
        
        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads

        self.W_q = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_k = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_v = nn.Linear(d_in, d_out, bias=qkv_bias)
        # combines head outputs
        self.out_proj = nn.Linear(d_out, d_out)
        self.dropout = nn.Dropout(dropout)

        # 
        self.register_buffer(
            "mask",
            torch.triu(torch.ones(context_len, context_len), diagonal=1)
        )
    
    def forward(self, x):
        b, seq_len, d_in = x.shape

        # [batch_size, seq_len, vocab_size/d_out]
        keys = self.W_k(x) 
        queries = self.W_q(x)
        values = self.W_v(x)

        # since head_dim = d_out/num_heads
        # we extend the last dimension
        keys = keys.view(b, seq_len, self.num_heads, self.head_dim)
        queries = queries.view(b, seq_len, self.num_heads, self.head_dim)
        values = values.view(b, seq_len, self.num_heads, self.head_dim)

        # Transpose and swap dimensions at seq_len and num_heads
        # new dim [batch_dize, num_heads, seq_len, head_dim]
        keys = keys.transpose(1,2)
        queries = queries.transpose(1,2)
        values = values.transpose(1,2)

        # attn_scores = [batch_dize, num_heads, seq_len, head_dim] @ [batch_dize, num_heads, head_dim, seq_len]
        # attn_scores = [batch_dize, num_heads, seq_len, seq_len]
        attn_scores = queries @ keys.transpose(2,3)

        mask_bool = self.mask.bool()[: seq_len, :seq_len]

        attn_scores.masked_fill_(mask_bool, -torch.inf)

        # divide by sqrd(d) where d aka d_out aka which we unrolled so last dimension of keys
        attn_weights = torch.softmax(attn_scores/keys.shape[-1]**0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # to get [batch_size, seq_len, num_heads, head_dim] again
        # [batch_dize, num_heads, seq_len, seq_len] @ [batch_dize, num_heads, seq_len, head_dim]
        # context_vec =  [batch_dize, num_heads, seq_len, head_dim].transpose(1,2)
        # context_vec = [batch_size, seq_len, num_heads, head_dim]
        context_vec = (attn_weights @ values).transpose(1,2)

        # comvine the heads as per d_out = head_dim * num_heads
        context_vec = context_vec.contiguous().view(b,seq_len, self.d_out)
        # optional linear transformation
        context_vec = self.proj_out(context_vec)

        return context_vec




