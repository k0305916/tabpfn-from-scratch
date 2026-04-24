"""多头注意力模块 for TabPFN"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class MultiHeadAttention(nn.Module):
    """多头注意力，支持 Flash Attention 和多种优化"""

    def __init__(
        self,
        d_model,
        nhead,
        dropout=0.1,
        bias=True,
        share_kv=False,
    ):
        """
        Args:
            d_model: 模型维度
            nhead: 注意力头数
            dropout: dropout 概率
            bias: 是否使用偏置
            share_kv: 是否共享 K/V（Grouped Query Attention）
        """
        super().__init__()
        assert d_model % nhead == 0, "d_model must be divisible by nhead"

        self.d_model = d_model
        self.nhead = nhead
        self.d_k = d_model // nhead
        self.d_v = self.d_k
        self.share_kv = share_kv

        # Q, K, V 投影
        if share_kv:
            # Grouped Query Attention: 1 组 KV，多组 Q
            self.q_proj = nn.Linear(d_model, d_model, bias=bias)
            self.kv_proj = nn.Linear(d_model, 2 * self.d_k, bias=bias)
        else:
            self.qkv_proj = nn.Linear(d_model, 3 * d_model, bias=bias)

        # 输出投影
        self.out_proj = nn.Linear(d_model, d_model, bias=bias)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        query,
        key,
        value,
        mask=None,
        key_padding_mask=None,
        need_weights=False,
    ):
        """
        Args:
            query: (batch, q_len, d_model)
            key: (batch, k_len, d_model)
            value: (batch, k_len, d_model)
            mask: (q_len, k_len) 用于 causal attention
            key_padding_mask: (batch, k_len) 用于 padding mask

        Returns:
            output: (batch, q_len, d_model)
            weights: (batch, nhead, q_len, k_len) if need_weights=True
        """
        batch_size = query.shape[0]
        q_len = query.shape[1]
        k_len = key.shape[1]

        # Q, K, V 投影
        if self.share_kv:
            q = self.q_proj(query)
            kv = self.kv_proj(key)
            k, v = kv[:, :, :self.d_k], kv[:, :, self.d_k:]
        else:
            qkv = self.qkv_proj(torch.cat([query, key, value], dim=-2))
            q, k, v = qkv.split(self.d_model, dim=-1)

        # reshape to (batch, nhead, seq, d_k)
        q = q.view(batch_size, q_len, self.nhead, self.d_k).transpose(1, 2)
        k = k.view(batch_size, k_len, self.nhead, self.d_k).transpose(1, 2)
        v = v.view(batch_size, k_len, self.nhead, self.d_v).transpose(1, 2)

        # 缩放
        scale = math.sqrt(self.d_k)
        q = q / scale

        # 计算注意力
        if mask is not None:
            mask = mask.to(q.device)

        # 尝试使用 scaled_dot_product_attention（PyTorch 2.0+）
        use_torch_attn = hasattr(F, 'scaled_dot_product_attention')

        if use_torch_attn and mask is None:
            # PyTorch 2.0 的 flash attention 实现
            attn_output = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=None,
                dropout_p=self.dropout.p if self.training else 0.0,
            )
        else:
            # 手动计算 attention
            attn_output, attn_weights = self._attention(
                q, k, v, mask, key_padding_mask
            )

        # 输出
        output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, q_len, self.d_model
        )
        output = self.out_proj(output)
        output = self.dropout(output)

        if need_weights:
            return output, attn_weights
        return output

    def _attention(self, q, k, v, mask=None, key_padding_mask=None):
        """手动实现注意力"""
        batch_size = q.shape[0]
        nhead = q.shape[1]
        q_len = q.shape[2]
        k_len = k.shape[2]

        # (batch, nhead, q_len, k_len)
        scores = torch.matmul(q, k.transpose(-2, -1))

        if mask is not None:
            scores = scores + mask

        if key_padding_mask is not None:
            # key_padding_mask: (batch, k_len) -> (batch, 1, 1, k_len)
            mask = key_padding_mask.unsqueeze(1).unsqueeze(2)
            scores = scores.masked_fill(mask, float('-inf'))

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        output = torch.matmul(attn_weights, v)
        return output, attn_weights


class SelfAttention(nn.Module):
    """自注意力的简化接口"""

    def __init__(self, d_model, nhead, dropout=0.1, bias=True):
        super().__init__()
        self.attn = MultiHeadAttention(d_model, nhead, dropout, bias)

    def forward(self, x, mask=None):
        """
        Args:
            x: (batch, seq_len, d_model)
            mask: (seq_len, seq_len) causal mask

        Returns:
            output: (batch, seq_len, d_model)
        """
        return self.attn(x, x, x, mask=mask)


if __name__ == '__main__':
    # 测试注意力
    batch_size = 2
    seq_len = 10
    d_model = 64
    nhead = 4

    x = torch.randn(batch_size, seq_len, d_model)

    attn = MultiHeadAttention(d_model, nhead)
    output = attn(x, x, x)

    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Parameters: {sum(p.numel() for p in attn.parameters())}")