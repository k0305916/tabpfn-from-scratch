"""多头注意力模块 for TabPFN"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class MultiHeadAttention(nn.Module):
    """多头注意力模块

    支持：
    - 3D 输入: (batch, seq, d_model)
    - Flash Attention (PyTorch 2.0+)
    - 手动实现作为 fallback
    """

    def __init__(
        self,
        d_model,
        nhead,
        dropout=0.1,
        bias=True,
    ):
        super().__init__()
        assert d_model % nhead == 0, "d_model must be divisible by nhead"

        self.d_model = d_model
        self.nhead = nhead
        self.d_k = d_model // nhead
        self.d_v = self.d_k

        # Q, K, V 投影（分开投影，避免 cat 的问题）
        self.q_proj = nn.Linear(d_model, d_model, bias=bias)
        self.k_proj = nn.Linear(d_model, d_model, bias=bias)
        self.v_proj = nn.Linear(d_model, d_model, bias=bias)

        # 输出投影
        self.out_proj = nn.Linear(d_model, d_model, bias=bias)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        query,
        key,
        value,
        mask=None,
        need_weights=False,
    ):
        """
        Args:
            query: (batch, q_len, d_model) 或 (q_len, d_model)
            key: (batch, k_len, d_model) 或 (k_len, d_model)
            value: (batch, k_len, d_model) 或 (k_len, d_model)
            mask: (q_len, k_len) 可选
            need_weights: 是否返回注意力权重

        Returns:
            output: (batch, q_len, d_model)
        """
        # 处理 2D 输入（单个序列）
        squeeze_batch = False
        if query.dim() == 2:
            query = query.unsqueeze(0)
            key = key.unsqueeze(0)
            value = value.unsqueeze(0)
            squeeze_batch = True

        batch_size = query.shape[0]
        q_len = query.shape[1]
        k_len = key.shape[1]

        # Q, K, V 投影
        q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(value)

        # reshape to (batch, nhead, seq, d_k)
        q = q.view(batch_size, q_len, self.nhead, self.d_k).transpose(1, 2)
        k = k.view(batch_size, k_len, self.nhead, self.d_k).transpose(1, 2)
        v = v.view(batch_size, k_len, self.nhead, self.d_v).transpose(1, 2)

        # 缩放
        scale = math.sqrt(self.d_k)
        q = q / scale

        # 计算注意力
        use_torch_attn = hasattr(F, 'scaled_dot_product_attention')

        if use_torch_attn and mask is None:
            attn_output = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=None,
                dropout_p=self.dropout.p if self.training else 0.0,
            )
        else:
            # 手动计算
            attn_output, _ = self._attention(q, k, v, mask)

        # 输出
        output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, q_len, self.d_model
        )
        output = self.out_proj(output)
        output = self.dropout(output)

        if squeeze_batch:
            output = output.squeeze(0)

        return output

    def _attention(self, q, k, v, mask=None):
        """手动实现注意力"""
        # q: (batch, nhead, q_len, d_k)
        # k: (batch, nhead, k_len, d_k)
        # v: (batch, nhead, k_len, d_v)

        scores = torch.matmul(q, k.transpose(-2, -1))

        if mask is not None:
            scores = scores + mask

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        output = torch.matmul(attn_weights, v)
        return output, attn_weights


class SelfAttention(nn.Module):
    """自注意力"""

    def __init__(self, d_model, nhead, dropout=0.1, bias=True):
        super().__init__()
        self.attn = MultiHeadAttention(d_model, nhead, dropout, bias)

    def forward(self, x, mask=None):
        """
        Args:
            x: (batch, seq_len, d_model) 或 (seq_len, d_model)
        """
        return self.attn(x, x, x, mask=mask)


if __name__ == '__main__':
    # 测试
    batch_size = 2
    seq_len = 10
    d_model = 64
    nhead = 4

    x = torch.randn(batch_size, seq_len, d_model)

    attn = MultiHeadAttention(d_model, nhead)
    output = attn(x, x, x)

    print(f"Input: {x.shape} -> Output: {output.shape}")
    print(f"Params: {sum(p.numel() for p in attn.parameters()):,}")