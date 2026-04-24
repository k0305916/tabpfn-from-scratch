"""In-Context 数据集 for TabPFN 训练"""
import torch
from torch.utils.data import Dataset
import numpy as np
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.synthetic import generate_classification, sample_dataset_config, feature_shift


class TabPFNDataset(Dataset):
    """TabPFN 训练数据集

    每个样本是一个 in-context 示例：
    - X_all: 全部特征 (n_samples, n_features)
    - y_all: 全部标签 (n_samples,)
    - 在使用时随机划分 support / query
    """

    def __init__(
        self,
        X,
        y,
        n_support=100,
        n_query=100,
        max_features=100,
        feature_shift=True,
        seed=None
    ):
        """
        Args:
            X: (N, D) 特征矩阵
            y: (N,) 标签
            n_support: support set 大小
            n_query: query set 大小
            max_features: 最大特征数（用于位置编码）
            feature_shift: 是否应用特征移位
            seed: 随机种子
        """
        self.X = X.astype(np.float32)
        self.y = y.astype(np.int64)
        self.n_support = n_support
        self.n_query = n_query
        self.max_features = max_features
        self.feature_shift = feature_shift
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return len(self.X) // (self.n_support + self.n_query)

    def __getitem__(self, idx):
        """获取一个 in-context 样本"""
        # 随机选择 support 和 query 样本
        n_total = len(self.X)
        indices = self.rng.permutation(n_total)[:self.n_support + self.n_query]
        support_idx = indices[:self.n_support]
        query_idx = indices[self.n_support:self.n_support + self.n_query]

        X_support = self.X[support_idx].copy()
        y_support = self.y[support_idx].copy()
        X_query = self.X[query_idx].copy()
        y_query = self.y[query_idx].copy()

        # 特征移位
        if self.feature_shift:
            shift = self.rng.integers(0, X_support.shape[1])
            X_support = np.roll(X_support, shift, axis=1)
            X_query = np.roll(X_query, shift, axis=1)

        # 填充/截取到 max_features
        n_feat = X_support.shape[1]
        if n_feat < self.max_features:
            pad_width = self.max_features - n_feat
            X_support = np.pad(X_support, ((0, 0), (0, pad_width)), mode='constant', constant_values=0)
            X_query = np.pad(X_query, ((0, 0), (0, pad_width)), mode='constant', constant_values=0)
        else:
            X_support = X_support[:, :self.max_features]
            X_query = X_query[:, :self.max_features]

        # 转换
        X_support = torch.from_numpy(X_support)
        X_query = torch.from_numpy(X_query)
        y_support = torch.from_numpy(y_support).float()
        y_query = torch.from_numpy(y_query).long()

        return X_support, y_support, X_query, y_query


def collate_in_context(batch):
    """将多个 in-context 样本合并为一个 batch"""
    X_support, y_support, X_query, y_query = zip(*batch)

    # X_support: (batch, n_support, n_features)
    # y_support: (batch, n_support)
    # X_query: (batch, n_query, n_features)
    # y_query: (batch, n_query)

    X_support = torch.stack(X_support)
    y_support = torch.stack(y_support)
    X_query = torch.stack(X_query)
    y_query = torch.stack(y_query)

    return X_support, y_support, X_query, y_query


class SyntheticDataGenerator:
    """在线生成合成数据"""

    def __init__(self, config=None, seed=None):
        if config is None:
            config = {}
        self.config = {
            'num_samples': config.get('num_samples', 1000),
            'num_features': config.get('num_features', 50),
            'num_classes': config.get('num_classes', 2),
            'num_informative': config.get('num_informative', 20),
            'num_redundant': config.get('num_redundant', 10),
            'feature_dist': config.get('feature_dist', 'normal'),
            'class_sep': config.get('class_sep', 1.0),
            'flip_y': config.get('flip_y', 0.0),
        }
        self.rng = np.random.default_rng(seed)

    def generate_batch(self):
        """生成一个 batch 的数据"""
        cfg = self.config.copy()
        cfg['class_sep'] = self.rng.uniform(0.5, 2.0)
        cfg['flip_y'] = self.rng.uniform(0, 0.05)
        X, y = generate_classification(cfg, seed=None)
        return X.astype(np.float32), y.astype(np.int64)

    def __iter__(self):
        return self

    def __next__(self):
        return self.generate_batch()


if __name__ == '__main__':
    # 测试数据集
    X, y = generate_classification(
        sample_dataset_config(np.random.default_rng(42))
    )
    print(f"Generated: X.shape={X.shape}, y.shape={y.shape}")

    dataset = TabPFNDataset(X, y, n_support=100, n_query=100, seed=42)
    print(f"Dataset length: {len(dataset)}")

    # 测试 __getitem__
    X_s, y_s, X_q, y_q = dataset[0]
    print(f"Support: X.shape={X_s.shape}, y.shape={y_s.shape}")
    print(f"Query: X.shape={X_q.shape}, y.shape={y_q.shape}")