"""PerFeatureTransformer 核心模块 for TabPFN"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from .attention import MultiHeadAttention


class MLP(nn.Module):
    """FFN 模块 (GELU 激活)"""

    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()

    def forward(self, x):
        return self.dropout(self.linear2(self.activation(self.linear1(x))))


class LayerNorm(nn.Module):
    """Post-norm LayerNorm"""

    def __init__(self, d_model, eps=1e-6):
        super().__init__()
        self.norm = nn.LayerNorm(d_model, eps=eps)

    def forward(self, x):
        return self.norm(x)


class PerFeatureEncoderLayer(nn.Module):
    """Per-Feature Transformer 层：双注意力 + FFN

    每层有两个注意力操作：
    1. self_attn_between_features: 在特征维度做自注意力
    2. self_attn_between_items: 在样本（item）维度做自注意力
    """

    def __init__(self, d_model, nhead, d_ff, dropout=0.1, bias=True):
        super().__init__()
        self.d_model = d_model

        # 特征间注意力
        self.self_attn_between_features = MultiHeadAttention(
            d_model, nhead, dropout, bias
        )
        # 样本间注意力
        self.self_attn_between_items = MultiHeadAttention(
            d_model, nhead, dropout, bias
        )

        # FFN
        self.mlp = MLP(d_model, d_ff, dropout)

        # LayerNorms (post-norm)
        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.norm3 = LayerNorm(d_model)

    def forward(self, x, mask=None):
        """
        Args:
            x: (batch, items, features, d_model)
            mask: 暂未使用

        Returns:
            output: (batch, items, features, d_model)
        """
        batch_size, num_items, num_features, d_model = x.shape

        # ===== 1. 特征间注意力 =====
        # 输入: (batch, items, features, d_model)
        # 转为: (batch * items, features, d_model) 进行注意力计算
        x_flat = x.view(batch_size * num_items, num_features, d_model)
        attn_out = self.self_attn_between_features(x_flat, x_flat, x_flat)
        attn_out = attn_out.view(batch_size, num_items, num_features, d_model)
        x = x + attn_out  # 残差连接
        x = self.norm1(x)

        # ===== 2. 样本间注意力 =====
        # 输入: (batch, items, features, d_model)
        # 转置: (batch, features, items, d_model)
        # 注意力: (batch * features, items, d_model)
        x_transposed = x.transpose(1, 2)  # (batch, features, items, d_model)
        x_flat = x_transposed.reshape(batch_size * num_features, num_items, d_model)
        attn_out = self.self_attn_between_items(x_flat, x_flat, x_flat)
        attn_out = attn_out.view(batch_size, num_features, num_items, d_model)
        attn_out = attn_out.transpose(1, 2)  # (batch, items, features, d_model)
        x = x + attn_out  # 残差连接
        x = self.norm2(x)

        # ===== 3. FFN =====
        x_flat = x.reshape(batch_size * num_items * num_features, d_model)
        mlp_out = self.mlp(x_flat)
        mlp_out = mlp_out.view(batch_size, num_items, num_features, d_model)
        x = x + mlp_out  # 残差连接
        x = self.norm3(x)

        return x


class FeatureEncoder(nn.Module):
    """特征编码器：将原始特征转换为 embedding"""

    def __init__(self, d_model, bias=True):
        super().__init__()
        self.linear = nn.Linear(1, d_model, bias=bias)  # 每个特征单独编码

    def forward(self, x):
        """
        Args:
            x: (batch, seq, features) 原始特征

        Returns:
            output: (batch, seq, features, d_model)
        """
        # x: (batch, seq, features) -> (batch, seq, features, 1)
        x = x.unsqueeze(-1)
        # linear: (batch, seq, features, 1) -> (batch, seq, features, d_model)
        return self.linear(x)


class TargetEncoder(nn.Module):
    """目标变量编码器：编码 y 值"""

    def __init__(self, d_model, bias=True):
        super().__init__()
        # y 编码：包含原始值 + NaN indicator
        self.linear = nn.Linear(2, d_model, bias=bias)  # y + nan_indicator

    def forward(self, y, nan_indicator=None):
        """
        Args:
            y: (batch, seq) 目标值
            nan_indicator: (batch, seq) NaN 标记，可选

        Returns:
            output: (batch, seq, d_model)
        """
        if nan_indicator is None:
            nan_indicator = torch.isnan(y).float()

        # 填充 NaN 为 0
        y_filled = torch.where(torch.isnan(y), torch.zeros_like(y), y)

        # 拼接 y 和 nan_indicator: (batch, seq, 2)
        y_input = torch.stack([y_filled, nan_indicator], dim=-1)
        return self.linear(y_input)


class FeaturePositionalEmbedding(nn.Module):
    """特征位置编码 - Subspace 方式"""

    def __init__(self, num_features, d_model):
        super().__init__()
        # 每个特征一个可学习的 embedding
        self.embeddings = nn.Parameter(torch.randn(num_features, d_model // 4))
        self.proj = nn.Linear(d_model // 4, d_model)

    def forward(self, x):
        """
        Args:
            x: (batch, seq, features, d_model)

        Returns:
            output: (batch, seq, features, d_model) 带位置编码
        """
        batch_size, seq_len, num_features, d_model = x.shape

        # embeddings: (num_features, d_model // 4)
        # 扩展到 (batch, seq, features, d_model // 4)
        pos_emb = self.proj(self.embeddings)  # (features, d_model)
        pos_emb = pos_emb[None, None, :, :]  # (1, 1, features, d_model)

        return x + pos_emb


class PerFeatureTransformer(nn.Module):
    """Per-Feature Transformer 核心模型

    关键设计：
    - 每个特征作为一个 token
    - 特征间注意力 + 样本间注意力
    - 支持 In-Context Learning
    """

    def __init__(
        self,
        num_features,
        num_classes,
        emsize=64,
        nhead=4,
        nlayers=4,
        nhid_factor=4,
        dropout=0.1,
        feature_positional_embedding='subspace',
    ):
        """
        Args:
            num_features: 最大特征数
            num_classes: 类别数
            emsize: embedding dimension
            nhead: 注意力头数
            nlayers: transformer 层数
            nhid_factor: FFN hidden size = emsize * nhid_factor
            dropout: dropout 概率
            feature_positional_embedding: 特征位置编码方式
        """
        super().__init__()
        self.num_features = num_features
        self.num_classes = num_classes
        self.emsize = emsize
        self.nhead = nhead
        self.nlayers = nlayers

        d_ff = emsize * nhid_factor

        # 编码器：输入是 1 维原始特征
        self.x_encoder = FeatureEncoder(emsize, bias=True)
        # y 编码：y 值 + nan indicator = 2 维
        self.y_encoder = TargetEncoder(emsize, bias=True)

        # 特征位置编码
        if feature_positional_embedding == 'subspace':
            self.feature_pos_emb = FeaturePositionalEmbedding(num_features, emsize)
        else:
            self.feature_pos_emb = None

        # Transformer 层
        self.layers = nn.ModuleList([
            PerFeatureEncoderLayer(emsize, nhead, d_ff, dropout)
            for _ in range(nlayers)
        ])

        # 分类头
        self.classifier = nn.Linear(emsize, num_classes)

        # 输出头
        self.out_activation = nn.Identity()

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, X, y, single_eval_pos=None, categorical_mask=None):
        """
        Args:
            X: (batch, seq_len, num_features) 特征矩阵
            y: (batch, seq_len) 目标值（训练部分有标签，测试部分为 NaN）
            single_eval_pos: 评估位置（测试集开始的位置）
            categorical_mask: (batch, 1, num_features) 特征是否为类别型

        Returns:
            output: (batch, num_classes) 预测 logits
        """
        batch_size, seq_len, num_features = X.shape

        if single_eval_pos is None:
            single_eval_pos = seq_len // 2  # 默认前半是训练，后半是测试

        # ===== 1. 编码 X =====
        # x_enc: (batch, seq, features, emsize)
        x_enc = self.x_encoder(X)

        # ===== 2. 编码 y =====
        # y_enc: (batch, seq, emsize)
        nan_indicator = torch.isnan(y).float()
        y_filled = torch.where(torch.isnan(y), torch.zeros_like(y), y)
        y_enc = self.y_encoder(y_filled, nan_indicator)

        # ===== 3. 添加特征位置编码 =====
        if self.feature_pos_emb is not None:
            x_enc = self.feature_pos_emb(x_enc)

        # ===== 4. 将 y 作为额外 token 拼接 =====
        # y_enc: (batch, seq, 1, emsize) -> 拼接在特征维度
        y_enc = y_enc.unsqueeze(2)  # (batch, seq, 1, emsize)
        # 拼接: (batch, seq, features + 1, emsize)
        combined = torch.cat([x_enc, y_enc], dim=2)

        # ===== 5. 通过 Transformer 层 =====
        state = combined
        for layer in self.layers:
            state = layer(state)

        # ===== 6. 取测试位置（single_eval_pos）的输出 =====
        # state: (batch, items, features+1, d_model)
        # 取 single_eval_pos 位置的 y token 输出用于分类
        eval_output = state[:, single_eval_pos, -1, :]  # (batch, d_model)

        # ===== 7. 分类 =====
        logits = self.classifier(eval_output)  # (batch, num_classes)
        return logits

    def get_num_params(self):
        """返回模型参数量"""
        return sum(p.numel() for p in self.parameters())


if __name__ == '__main__':
    # 测试模型
    batch_size = 4
    seq_len = 200  # 100 train + 100 test
    num_features = 50
    num_classes = 2

    model = PerFeatureTransformer(
        num_features=num_features,
        num_classes=num_classes,
        emsize=64,
        nhead=4,
        nlayers=4,
    )

    print(f"Model parameters: {model.get_num_params():,}")

    # 创建输入
    X = torch.randn(batch_size, seq_len, num_features)
    y = torch.randn(batch_size, seq_len)
    y[seq_len // 2:] = float('nan')  # 后半是测试集

    # 前向
    output = model(X, y, single_eval_pos=seq_len // 2)
    print(f"Output shape: {output.shape}")  # (batch, num_classes)