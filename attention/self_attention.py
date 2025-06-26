# implements self attention with causal mask
import torch
import torch.nn as nn

class SelfAttention(nn.Module):

    def __init__(self, d_in, d_out, dropout, context_length):
        super().__init__()

        self.d_scale = d_out
        # self.q = nn.Linear(d_in, d_out)
        self.w_q = nn.Parameter(torch.rand(d_in,d_out))
        # self.k = nn.Linear(d_in, d_out)
        # self.v = nn.Linear(d_in, d_out)
        self.w_k = nn.Parameter(torch.rand(d_in, d_out))
        self.w_v = nn.Parameter(torch.rand(d_in, d_out))
        self.dropout = nn.Dropout(dropout)
        self.register_buffer ('mask', torch.triu(torch.ones(context_length, context_length), diagonal = 1))

    def forward(self, x):
        a ,seq_len, c= x.shape
        q = x @ self.w_q
        k = x @ self.w_k
        v = x @ self.w_v 

        print(f"Dimension of x: {x.shape}\n")
        print(f"Dimension of k: {k.shape}\n")
        print(f"Dimension of k: {k.transpose(1,2).shape}\n")
        print(f"Dimension of q: {q.shape}\n")
        print(f"Dimension of v: {v.shape}\n")

        attn_score = q @ k.transpose(1,2)

        attn_score.masked_fill(
            self.mask.bool()[:seq_len, :seq_len], -torch.inf
        )
        # attn_weights = atten_score / self.d_scale
        attn_weights = torch.softmax(
            attn_score / k.shape[-1]**0.5, dim=-1
        )

        attn_weights = self.dropout(attn_weights)
        context_vector = attn_weights @ v

        # attn_weights = attn_weights.softmax(dim=-1)

        return context_vector


# torch.manual_seed(123)

# inputs = torch.tensor(
#   [[0.43, 0.15, 0.89], # Your     (x^1)
#    [0.55, 0.87, 0.66], # journey  (x^2)
#    [0.57, 0.85, 0.64], # starts   (x^3)
#    [0.22, 0.58, 0.33], # with     (x^4)
#    [0.77, 0.25, 0.10], # one      (x^5)
#    [0.05, 0.80, 0.55]] # step     (x^6)
# )
# context_length = 4
# batch = torch.stack((inputs, inputs), dim=0)

# sa_v1 = SelfAttention(3,2, 0.0, context_length)
# print(sa_v1(inputs))



