"""TabPFN 从头实现 - 主入口"""
import torch
import numpy as np
import sys
import os

from models.transformer import PerFeatureTransformer
from train.trainer import Trainer
from train.dataset import TabPFNDataset, collate_in_context, SyntheticDataGenerator
from data.synthetic import generate_batch, sample_dataset_config
from torch.utils.data import DataLoader

from classifier import TabPFNClassifier
from preprocessing.pipeline import TabPFNPreprocessingPipeline


def train_example():
    """训练示例"""
    print("=" * 50)
    print("TabPFN 训练示例")
    print("=" * 50)

    # 配置
    num_features = 50
    num_classes = 3
    batch_size = 16
    num_epochs = 10

    # 生成数据
    print("\n[1] 生成合成数据...")
    X, y = generate_batch(
        n_samples=500,
        n_features=num_features,
        n_classes=num_classes,
        n_batches=100,  # 50k 样本
        seed=42
    )
    print(f"Generated: X.shape={X.shape}, y.shape={y.shape}")

    # 创建数据集
    print("\n[2] 创建数据集...")
    dataset = TabPFNDataset(
        X, y,
        n_support=100,
        n_query=100,
        max_features=num_features,
        seed=42
    )
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_in_context,
        num_workers=0
    )
    print(f"Dataset size: {len(dataset)} batches")

    # 创建模型
    print("\n[3] 创建模型...")
    model = PerFeatureTransformer(
        num_features=num_features,
        num_classes=num_classes,
        emsize=64,
        nhead=4,
        nlayers=4,
    )
    print(f"Model parameters: {model.get_num_params():,}")

    # 训练
    print("\n[4] 开始训练...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    trainer = Trainer(
        model,
        num_classes=num_classes,
        lr=1e-3,
        device=device
    )

    trainer.train(dataloader, num_epochs=num_epochs)

    print("\n[5] 训练完成!")

    # 保存模型
    checkpoint_path = 'tabpfn_model.pt'
    trainer.save_checkpoint(checkpoint_path)
    print(f"Model saved to {checkpoint_path}")

    return model


def inference_example():
    """推理示例"""
    print("\n" + "=" * 50)
    print("TabPFN 推理示例")
    print("=" * 50)

    from sklearn.datasets import make_classification

    # 生成测试数据
    X_train, y_train = make_classification(n_samples=200, n_features=50, n_classes=3)
    X_test, y_test = make_classification(n_samples=50, n_features=50, n_classes=3)

    print(f"Train: X.shape={X_train.shape}, y.shape={y_train.shape}")
    print(f"Test: X.shape={X_test.shape}")

    # 使用分类器
    clf = TabPFNClassifier(
        n_estimators=2,
        emsize=32,
        nlayers=2,
    )
    clf.fit(X_train, y_train)

    pred = clf.predict(X_test)
    print(f"Predictions: {pred[:10]}...")
    print(f"Accuracy: {(pred == y_test).mean():.4f}")


def quick_test():
    """快速测试"""
    print("=" * 50)
    print("快速测试")
    print("=" * 50)

    # 测试模型前向
    print("\n[1] 测试模型前向...")
    batch_size = 2
    seq_len = 200
    num_features = 50
    num_classes = 2

    model = PerFeatureTransformer(
        num_features=num_features,
        num_classes=num_classes,
        emsize=64,
        nhead=4,
        nlayers=4,
    )
    print(f"Parameters: {model.get_num_params():,}")

    X = torch.randn(batch_size, seq_len, num_features)
    y = torch.randn(batch_size, seq_len)
    y[:, seq_len // 2:] = float('nan')

    output = model(X, y, single_eval_pos=seq_len // 2)
    print(f"Output shape: {output.shape}")

    # 测试数据生成
    print("\n[2] 测试数据生成...")
    from data.synthetic import generate_classification, sample_dataset_config
    config = sample_dataset_config(np.random.default_rng(42))
    X, y = generate_classification(config)
    print(f"Generated: X.shape={X.shape}, y.shape={y.shape}")

    print("\n[OK] 所有测试通过!")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='TabPFN 从头实现')
    parser.add_argument('--mode', choices=['train', 'inference', 'test'], default='test')
    args = parser.parse_args()

    if args.mode == 'train':
        train_example()
    elif args.mode == 'inference':
        inference_example()
    else:
        quick_test()