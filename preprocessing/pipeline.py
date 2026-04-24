"""预处理模块 for TabPFN"""
import numpy as np
from sklearn.preprocessing import StandardScaler, QuantileTransformer


class Normalizer:
    """数据归一化器"""

    def __init__(self, method='standard'):
        """
        Args:
            method: 'standard' (Z-score) 或 'quantile' (均匀分布)
        """
        self.method = method
        self.scaler = None

    def fit(self, X):
        """拟合归一化参数

        Args:
            X: (n_samples, n_features)
        """
        if self.method == 'standard':
            self.scaler = StandardScaler()
        elif self.method == 'quantile':
            self.scaler = QuantileTransformer(n_quantiles=1000, output_distribution='uniform')

        self.scaler.fit(X)
        return self

    def transform(self, X):
        """应用归一化

        Args:
            X: (n_samples, n_features)

        Returns:
            X_norm: 归一化后的数据
        """
        return self.scaler.transform(X)

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)


class TabPFNPreprocessingPipeline:
    """TabPFN 预处理管道

    包括：
    1. 移除常数特征
    2. 归一化
    3. 异常值移除
    4. 特征移位
    """

    def __init__(
        self,
        normalize='standard',
        remove_outliers=True,
        outlier_std=4.0,
        feature_shift=True,
        seed=None,
    ):
        """
        Args:
            normalize: 'standard', 'quantile', 或 None
            remove_outliers: 是否移除异常值
            outlier_std: 异常值判定阈值（标准差倍数）
            feature_shift: 是否应用特征移位
            seed: 随机种子
        """
        self.normalize = normalize
        self.remove_outliers = remove_outliers
        self.outlier_std = outlier_std
        self.feature_shift = feature_shift
        self.rng = np.random.default_rng(seed)

        self.normalizer = None
        self.constant_features = []
        self.feature_means = None

    def fit(self, X, y=None):
        """拟合预处理参数

        Args:
            X: (n_samples, n_features)
        """
        X = X.copy()

        # 1. 移除常数特征
        self.constant_features = np.where(X.std(axis=0) == 0)[0]
        if len(self.constant_features) > 0:
            X = np.delete(X, self.constant_features, axis=1)

        # 2. 保存特征均值（用于填充 NaN）
        self.feature_means = X.mean(axis=0)

        # 3. 归一化
        if self.normalize:
            self.normalizer = Normalizer(method=self.normalize)
            X = self.normalizer.fit_transform(X)

        self.n_features_ = X.shape[1]
        return self

    def transform(self, X):
        """应用预处理

        Args:
            X: (n_samples, n_features)

        Returns:
            X_proc: 预处理后的数据
        """
        X = X.copy()

        # 1. 移除常数特征
        if len(self.constant_features) > 0:
            X = np.delete(X, self.constant_features, axis=1)

        # 2. 填充 NaN（使用特征均值）
        X = np.where(np.isnan(X), self.feature_means[None, :], X)

        # 3. 异常值移除
        if self.remove_outliers:
            mean = X.mean(axis=0)
            std = X.std(axis=0)
            X = np.clip(X, mean - self.outlier_std * std, mean + self.outlier_std * std)

        # 4. 归一化
        if self.normalizer:
            X = self.normalizer.transform(X)

        return X

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)


def preprocess_for_inference(X, fit_pipeline=True, pipeline=None, seed=None):
    """推理时的预处理

    Args:
        X: (n_samples, n_features)
        fit_pipeline: 是否拟合新的 pipeline
        pipeline: 已拟合的 pipeline
        seed: 随机种子

    Returns:
        X_proc: 预处理后的数据
        pipeline: 使用的 pipeline
    """
    if pipeline is None or fit_pipeline:
        pipeline = TabPFNPreprocessingPipeline(seed=seed)
        X_proc = pipeline.fit_transform(X)
    else:
        X_proc = pipeline.transform(X)

    return X_proc, pipeline


if __name__ == '__main__':
    # 测试预处理
    X = np.random.randn(1000, 50)
    X[:, 10] = 0  # 常数特征
    X[0, 5] = 100  # 异常值

    pipeline = TabPFNPreprocessingPipeline(normalize='standard', remove_outliers=True)
    X_proc = pipeline.fit_transform(X)

    print(f"Original shape: {X.shape}")
    print(f"Processed shape: {X_proc.shape}")
    print(f"Constant features removed: {len(pipeline.constant_features)}")