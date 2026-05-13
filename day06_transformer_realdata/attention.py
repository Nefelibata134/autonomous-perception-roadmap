"""
attention.py
手撕 Self-Attention 和 Multi-Head Attention
"""
import torch
import torch.nn as nn
import math


class SelfAttention(nn.Module):
    """
    单头 Self-Attention
    输入: x (batch, seq_len, embed_dim)
    输出: attn_output (batch, seq_len, embed_dim), attn_weights (batch, seq_len, seq_len)
    """
    def __init__(self, embed_dim):
        super(SelfAttention, self).__init__()
        self.embed_dim = embed_dim

        # 三个线性投影：把输入 x 分别映射为 Q/K/V
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)

        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward (self, x):
        batch_size, seq_len, embed_dim = x.size()

        # 1. 投影得到 Q/K/V
        Q = self.q_proj(x)
        K = self.k_proj(x)
        V = self.v_proj(x)

        # 2. 计算注意力分数: QK^T / sqrt(d_k)
        # K.transpose(1,2): (B, embed, seq)
        scores = torch.matmul(Q, K.transpose(1, 2)) / math.sqrt(embed_dim)
        # scores: (B, seq, seq)

        # 3. softmax 归一化
        attn_weights = torch.softmax(scores, dim=-1)
        # attn_weights: (B, seq, seq)，每行之和为 1

        # 4. 加权求和: attn_weights · V
        attn_output = torch.matmul(attn_weights, V)
        # attn_output: (B, seq, embed)

        # 5. 输出投影
        output = self.out_proj(attn_output)

        return output, attn_weights


class MultiHeadAttention(nn.Module):
    """
    多头注意力
    把 Q/K/V 分成 h 个头，每个头独立做 Attention，最后拼接
    """
    def __init__(self, embed_dim, num_heads):
        super(MultiHeadAttention, self).__init__()
        assert embed_dim % num_heads == 0

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads # 每个头的维度

        #总投影： 把embed_dim 映射到 embed_dim(内部会分成num_heads 份)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)

        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward (self, x):
        batch_size, seq_len, embed_dim = x.size()

        # 1. 投影并分头
        # (B, seq, embed) -> (B, seq, num_heads, head_dim) -> (B, num_heads, seq, head_dim)
        Q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        # Q/K/V: (B, num_heads, seq, head_dim)

        # 2. 每个头独立计算 Attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        # scores: (B, num_heads, seq, seq)

        attn_weights = torch.softmax(scores, dim=-1)

        attn_output = torch.matmul(attn_weights, V)
        # attn_output: (B, num_heads, seq, head_dim)

        # 3. 拼接多头: (B, num_heads, seq, head_dim) -> (B, seq, num_heads, head_dim) -> (B, seq, embed)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, embed_dim)

        # 4. 输出投影
        output = self.out_proj(attn_output)

        return output, attn_weights


def test_attention():
    """测试"""
    batch_size = 2
    seq_len = 4  # 序列长度（比如 4 个单词/图像块）
    embed_dim = 8  # 嵌入维度
    num_heads = 2  # 2 个头

    x = torch.randn(batch_size, seq_len, embed_dim)

    print("===== Self-Attention =====")
    sa = SelfAttention(embed_dim)
    out, weights = sa(x)
    print(f"输入: {x.shape}")
    print(f"输出: {out.shape}")
    print(f"注意力权重: {weights.shape}")
    print(f"权重每行和: {weights.sum(dim=-1)[0]}")  # 应该接近 1

    print("\n===== Multi-Head Attention =====")
    mha = MultiHeadAttention(embed_dim, num_heads)
    out, weights = mha(x)
    print(f"输入: {x.shape}")
    print(f"输出: {out.shape}")
    print(f"注意力权重: {weights.shape}")  # (B, num_heads, seq, seq)
    print("✅ Attention 测试通过！")


if __name__ == "__main__":
    test_attention()


