"""合成数据生成模块 for TabPFN 训练"""
import numpy as np
from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler


def sample_dataset_config(rng=None):
    """随机采样数据集配置"""
    if rng is None:
        rng = np.random.default_rng()

    return {
        'num_samples': rng.integers(500, 2000),       # 样本数
        'num_features': rng.integers(20, 100),        # 特征数
        'num_classes': rng.choice([2, 3, 4, 5, 10]),  # 类别数
        'num_informative': rng.integers(10, 50),      # 有效特征数
        'num_redundant': rng.integers(5, 30),         # 冗余特征数
        'feature_dist': rng.choice(['normal', 'skewed', 'uniform']),
        'class_sep': rng.uniform(0.5, 2.0),          # 类别分离度
        'flip_y': rng.uniform(0, 0.05),               # 标签噪声
    }


def generate_classification(config=None, seed=None):
    """生成分类合成数据

    Args:
        config: 配置字典，如果为 None 则随机采样
        seed: 随机种子

    Returns:
        X: (n_samples, n_features) 特征矩阵
        y: (n_samples,) 标签
    """
    if config is None:
        config = sample_dataset_config()

    if seed is not None:
        rng = np.random.default_rng(seed)
    else:
        rng = np.random.default_rng()

    # 生成数据
    X, y = make_classification(
        n_samples=config['num_samples'],
        n_features=config['num_features'],
        n_informative=config.get('num_informative', 20),
        n_redundant=config.get('num_redundant', 10),
        n_classes=config['num_classes'],
        n_clusters_per_class=rng.integers(2, 5),
        class_sep=config.get('class_sep', 1.0),
        flip_y=config.get('flip_y', 0.0),
        random_state=int(rng.integers(0, 2**31)),
    )

    # 特征变换
    if config.get('feature_dist') == 'skewed':
        # 添加偏斜
        X = np.sign(X) * np.abs(X) ** 0.5
        X = X * (1 + rng.normal(0, 0.1, size=X.shape))
    elif config.get('feature_dist') == 'uniform':
        X = (X - X.min()) / (X.max() - X.min()) * 2 - 1

    return X, y


def generate_batch(n_samples, n_features, n_classes, n_batches=100, seed=None):
    """批量生成合成数据

    Args:
        n_samples: 每个 batch 的样本数
        n_features: 特征数
        n_classes: 类别数
        n_batches: batch 数量
        seed: 随机种子

    Returns:
        X_all: (n_samples * n_batches, n_features)
        y_all: (n_samples * n_batches,)
    """
    all_X, all_y = [], []
    rng = np.random.default_rng(seed)

    for _ in range(n_batches):
        X, y = generate_classification({
            'num_samples': n_samples,
            'num_features': n_features,
            'num_classes': n_classes,
            'num_informative': min(n_features // 2, 20),
            'num_redundant': min(n_features // 4, 10),
            'feature_dist': 'normal',
            'class_sep': rng.uniform(0.8, 1.5),
            'flip_y': rng.uniform(0, 0.02),
        }, seed=None)

        all_X.append(X)
        all_y.append(y)

    return np.vstack(all_X), np.hstack(all_y)


def feature_shift(X, method='rotate', shift=None, rng=None):
    """特征列移位（数据增强）

    Args:
        X: (n_samples, n_features) 特征矩阵
        method: 'rotate' (循环移位) 或 'shuffle' (随机打乱)
        shift: 循环移位的位数
        rng: 随机数生成器

    Returns:
        X_shifted: 移位后的特征矩阵
    """
    if rng is None:
        rng = np.random.default_rng()

    if method == 'rotate':
        if shift is None:
            shift = rng.integers(0, X.shape[1])
        return np.roll(X, shift, axis=1)
    else:  # shuffle
        perm = rng.permutation(X.shape[1])
        return X[:, perm]


class InContextDataset:
    """In-Context Learning 格式的数据集

    每个样本包含 support set（有标签）和 query set（无标签）
    """

    def __init__(
        self,
        X,
        y,
        n_support=100,
        n_query=100,
        feature_shift=True,
        seed=None
    ):
        """
        Args:
            X: 特征矩阵
            y: 标签
            n_support: support set 大小
            n_query: query set 大小
            feature_shift: 是否应用特征移位
            seed: 随机种子
        """
        self.X = X
        self.y = y
        self.n_support = n_support
        self.n_query = n_query
        self.feature_shift = feature_shift
        self.rng = np.random.default_rng(seed)

    def __getitem__(self, idx):
        """获取一个 in-context 样本"""
        # 随机选择 support 和 query 样本
        n_total = len(self.X)
        indices = self.rng.permutation(n_total)
        support_idx = indices[:self.n_support]
        query_idx = indices[self.n_support:self.n_support + self.n_query]

        X_support = self.X[support_idx]
        y_support = self.y[support_idx]
        X_query = self.X[query_idx]
        y_query = self.y[query_idx]

        # 特征移位
        if self.feature_shift:
            X_support = feature_shift(X_support, method='rotate', rng=self.rng)
            X_query = feature_shift(X_query, method='rotate', rng=self.rng)

        return {
            'X_support': X_support,      # (n_support, n_features)
            'y_support': y_support,      # (n_support,)
            'X_query': X_query,          # (n_query, n_features)
            'y_query': y_query,          # (n_query,)
        }

    def __len__(self):
        return len(self.X) // (self.n_support + self.n_query)


if __name__ == '__main__':
    # 测试数据生成
    config = sample_dataset_config()
    print("Dataset config:", config)

    X, y = generate_classification(config)
    print(f"Generated: X.shape={X.shape}, y.shape={y.shape}")
    print(f"Classes: {np.unique(y)}")
    print(f"Class distribution: {np.bincount(y)}")