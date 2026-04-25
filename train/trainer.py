"""训练器 for TabPFN"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import sys
import os
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.transformer import PerFeatureTransformer
from train.loss import InContextLoss


class Trainer:
    """TabPFN 训练器"""

    def __init__(
        self,
        model,
        num_classes=2,
        lr=1e-3,
        weight_decay=1e-5,
        device=None,
    ):
        """
        Args:
            model: PerFeatureTransformer 模型
            num_classes: 类别数
            lr: 学习率
            weight_decay: 权重衰减
            device: 计算设备
        """
        self.model = model
        self.num_classes = num_classes

        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.device = device
        self.model.to(device)

        # 优化器
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
        )

        # 学习率调度
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=100, eta_min=1e-6
        )

        # 损失函数
        self.criterion = InContextLoss(num_classes, loss_type='ce')

    def train_step(self, X_support, y_support, X_query, y_query):
        """训练一步

        Args:
            X_support: (batch, n_support, n_features)
            y_support: (batch, n_support)
            X_query: (batch, n_query, n_features)
            y_query: (batch, n_query)
        """
        batch_size = X_support.shape[0]
        n_support = X_support.shape[1]
        n_query = X_query.shape[1]

        # 拼接 support 和 query
        # X: (batch, n_support + n_query, n_features)
        X = torch.cat([X_support, X_query], dim=1)
        # y: support 有标签，query 设为 NaN（表示未知）
        y = torch.cat([y_support, torch.full((batch_size, n_query), float('nan'))], dim=1)

        X = X.to(self.device)
        y = y.to(self.device)

        # 前向传播
        logits = self.model(X, y, single_eval_pos=n_support)  # (batch, num_classes)

        # 计算 loss：取第一个 query 样本的标签（简化处理）
        y_query_single = y_query[:, 0].to(self.device).long()  # (batch,)
        loss = self.criterion(logits, y_query_single)

        # 反向传播
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()

        return loss.item()

    def train_epoch(self, dataloader, epoch=None):
        """训练一个 epoch"""
        self.model.train()
        total_loss = 0
        num_batches = 0

        pbar = tqdm(dataloader, desc=f'Epoch {epoch}' if epoch else 'Training')
        for batch in pbar:
            X_support, y_support, X_query, y_query = batch

            loss = self.train_step(X_support, y_support, X_query, y_query)
            total_loss += loss
            num_batches += 1

            pbar.set_postfix({'loss': f'{loss:.4f}'})

        return total_loss / num_batches

    def train(
        self,
        dataloader,
        num_epochs=50,
        eval_every=5,
        eval_dataloader=None,
        resume_from=None,
    ):
        """完整训练流程

        Args:
            dataloader: 训练数据加载器
            num_epochs: 训练轮数
            eval_every: 每隔几个 epoch 评估一次
            eval_dataloader: 评估数据加载器
            resume_from: 从 checkpoint 恢复训练的路径，格式 "checkpoint.pt,epoch"
        """
        start_epoch = 0
        best_loss = float('inf')

        # 恢复检查点
        if resume_from is not None:
            if ',' in resume_from:
                checkpoint_path, epoch_str = resume_from.rsplit(',', 1)
                start_epoch = int(epoch_str)
                _, saved_best_loss = self.load_checkpoint(checkpoint_path)
                print(f"Resumed from epoch {start_epoch}, checkpoint loaded")
            else:
                checkpoint_path = resume_from
                saved_epoch, saved_best_loss = self.load_checkpoint(checkpoint_path)
                if saved_epoch is not None:
                    start_epoch = saved_epoch
                if saved_best_loss is not None:
                    best_loss = saved_best_loss
                print(f"Checkpoint loaded (epoch {start_epoch}, best_loss={best_loss:.4f}), continuing training")

        for epoch in range(start_epoch, num_epochs):
            # 训练
            train_loss = self.train_epoch(dataloader, epoch + 1)
            self.scheduler.step()

            # 评估（每次 epoch 都评估，以便及时捕捉突破）
            if eval_dataloader is not None:
                eval_loss = self.evaluate(eval_dataloader)

                # 检查是否突破
                if eval_loss < best_loss:
                    best_loss = eval_loss
                    self.save_checkpoint('best_model.pt', epoch=epoch + 1, best_loss=best_loss)
                    print(f"Epoch {epoch+1}: train_loss={train_loss:.4f}, eval_loss={eval_loss:.4f} -> New best, saved!")
                elif (epoch + 1) % eval_every == 0:
                    print(f"Epoch {epoch+1}: train_loss={train_loss:.4f}, eval_loss={eval_loss:.4f}")
            else:
                if (epoch + 1) % eval_every == 0:
                    print(f"Epoch {epoch+1}: train_loss={train_loss:.4f}")

        return best_loss

    def evaluate(self, dataloader):
        """评估模型"""
        self.model.eval()
        total_loss = 0
        num_batches = 0

        with torch.no_grad():
            for batch in dataloader:
                X_support, y_support, X_query, y_query = batch

                batch_size = X_support.shape[0]
                n_support = X_support.shape[1]
                n_query = X_query.shape[1]

                X = torch.cat([X_support, X_query], dim=1)
                y = torch.cat([y_support, torch.full((batch_size, n_query), float('nan'))], dim=1)

                X = X.to(self.device)
                y = y.to(self.device)

                logits = self.model(X, y, single_eval_pos=n_support)
                y_query_single = y_query[:, 0].to(self.device).long()
                loss = self.criterion(logits, y_query_single)

                total_loss += loss.item()
                num_batches += 1

        return total_loss / num_batches

    def save_checkpoint(self, path, epoch=None, best_loss=None):
        """保存模型"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'epoch': epoch,
            'best_loss': best_loss,
        }, path)

    def load_checkpoint(self, path):
        """加载模型"""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        return checkpoint.get('epoch', None), checkpoint.get('best_loss', None)


def train_model(
    num_features=100,
    num_classes=2,
    num_epochs=50,
    batch_size=32,
    lr=1e-3,
    device=None,
):
    """训练 TabPFN 模型的快捷函数"""
    from train.dataset import TabPFNDataset, collate_in_context
    from data.synthetic import generate_batch

    # 生成数据
    print("Generating synthetic data...")
    X, y = generate_batch(
        n_samples=500,
        n_features=num_features,
        n_classes=num_classes,
        n_batches=200,  # 共 100k 样本
    )

    # 创建数据集
    dataset = TabPFNDataset(X, y, n_support=100, n_query=100, seed=42)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_in_context)

    # 创建模型
    model = PerFeatureTransformer(
        num_features=num_features,
        num_classes=num_classes,
        emsize=64,
        nhead=4,
        nlayers=4,
    )
    print(f"Model parameters: {model.get_num_params():,}")

    # 创建训练器
    trainer = Trainer(model, num_classes=num_classes, lr=lr, device=device)

    # 训练
    trainer.train(dataloader, num_epochs=num_epochs)

    return model


if __name__ == '__main__':
    # 测试训练
    print("Starting training test...")
    model = train_model(num_features=50, num_classes=3, num_epochs=5)
    print("Training done!")