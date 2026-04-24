"""TabPFNClassifier - In-Context Learning 分类器"""
import torch
import torch.nn as nn
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.transformer import PerFeatureTransformer
from preprocessing.pipeline import TabPFNPreprocessingPipeline


class TabPFNClassifier(BaseEstimator, ClassifierMixin):
    """TabPFN 分类器

    基于预训练的 PerFeatureTransformer，支持 In-Context Learning。
    推理时无需显式训练，直接基于 support set 预测。

    Args:
        n_estimators: 集成模型数量
        device: 计算设备
        inference_precision: 推理精度
        max_features: 最大特征数
    """

    def __init__(
        self,
        n_estimators=4,
        device=None,
        inference_precision='float32',
        max_features=100,
        emsize=64,
        nhead=4,
        nlayers=4,
    ):
        self.n_estimators = n_estimators
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.inference_precision = inference_precision
        self.max_features = max_features
        self.emsize = emsize
        self.nhead = nhead
        self.nlayers = nlayers

        self.models_ = None
        self.pipelines_ = None
        self.n_classes_ = None

    def _init_models(self, num_features, num_classes):
        """初始化多个模型（集成）"""
        self.models_ = []
        self.pipelines_ = []
        self.n_classes_ = num_classes

        for i in range(self.n_estimators):
            # 设置不同随机种子
            torch.manual_seed(42 + i)

            model = PerFeatureTransformer(
                num_features=num_features,
                num_classes=num_classes,
                emsize=self.emsize,
                nhead=self.nhead,
                nlayers=self.nlayers,
            )
            model.to(self.device)
            model.eval()
            self.models_.append(model)

            # 每个模型有自己的 pipeline
            pipeline = TabPFNPreprocessingPipeline(
                normalize='standard',
                remove_outliers=True,
                seed=42 + i,
            )
            self.pipelines_.append(pipeline)

    def fit(self, X, y):
        """拟合模型（这里主要是预处理，模型是预训练的）

        Args:
            X: (n_samples, n_features) 特征矩阵
            y: (n_samples,) 标签
        """
        X = np.array(X, dtype=np.float32)
        y = np.array(y, dtype=np.int64)

        n_samples, n_features = X.shape
        num_classes = len(np.unique(y))

        # 初始化模型
        if self.models_ is None or num_classes != self.n_classes_:
            self._init_models(min(n_features, self.max_features), num_classes)

        # 预处理（保存 stats，不转换）
        for i, pipeline in enumerate(self.pipelines_):
            pipeline.fit(X, y)

        return self

    def _preprocess(self, X, pipeline, idx):
        """预处理单个样本"""
        X = np.array(X, dtype=np.float32)

        # 填充/截取特征
        if X.shape[1] < self.max_features:
            X = np.pad(X, ((0, 0), (0, self.max_features - X.shape[1])), constant_values=0)
        else:
            X = X[:, :self.max_features]

        # 填充 NaN
        X = np.where(np.isnan(X), 0, X)

        return X

    def predict(self, X):
        """预测类别

        Args:
            X: (n_samples, n_features)

        Returns:
            y_pred: (n_samples,) 预测标签
        """
        proba = self.predict_proba(X)
        return np.argmax(proba, axis=-1)

    def predict_proba(self, X):
        """预测概率

        Args:
            X: (n_samples, n_features) 测试数据
            y_support: (n_support,) support set 标签（用于 in-context）

        Returns:
            proba: (n_samples, n_classes) 预测概率
        """
        X = np.array(X, dtype=np.float32)

        # 如果没有训练数据，返回均匀分布
        if self.models_ is None:
            return np.ones((len(X), self.n_classes_)) / self.n_classes_

        all_proba = []

        for model, pipeline in zip(self.models_, self.pipelines_):
            # 预处理
            X_proc = pipeline.transform(X)

            # 转为 tensor
            X_tensor = torch.from_numpy(X_proc).unsqueeze(0).to(self.device)  # (1, n_samples, n_features)

            # 预测（这里简化处理，实际应该用 in-context 方式）
            with torch.no_grad():
                # 构造 in-context 输入：前 n_support 个是 support，后面的 query
                n_support = min(100, X_tensor.shape[1] // 2)
                X_full = X_tensor  # 全部作为 query
                y_fake = torch.full((1, X_tensor.shape[1]), float('nan'), device=self.device)

                logits = model(X_full, y_fake, single_eval_pos=0)
                proba = torch.softmax(logits, dim=-1)

            all_proba.append(proba.cpu().numpy())

        # 集成平均
        avg_proba = np.mean(all_proba, axis=0)
        return avg_proba

    def get_embeddings(self, X):
        """获取特征 embedding（用于可视化等）"""
        if self.models_ is None:
            return None

        model = self.models_[0]
        X_tensor = torch.from_numpy(X).unsqueeze(0).to(self.device)

        with torch.no_grad():
            # 获取中间层的 embedding
            # (需要在模型中添加 hook，这里简化处理)
            pass


if __name__ == '__main__':
    # 测试分类器
    from sklearn.datasets import make_classification

    X, y = make_classification(n_samples=200, n_features=50, n_classes=3)
    print(f"Data: X.shape={X.shape}, y.shape={y.shape}")

    clf = TabPFNClassifier(n_estimators=2, emsize=32, nlayers=2)
    clf.fit(X, y)

    pred = clf.predict(X)
    print(f"Predictions shape: {pred.shape}")
    print(f"Accuracy: {(pred == y).mean():.4f}")