"""损失函数 for TabPFN"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ClassificationLoss(nn.Module):
    """分类损失：支持 Binary CE 和 Multi-class CE"""

    def __init__(self, num_classes):
        super().__init__()
        self.num_classes = num_classes

    def forward(self, logits, targets):
        """
        Args:
            logits: (batch, num_classes) 模型输出
            targets: (batch,) 标签

        Returns:
            loss: scalar
        """
        if self.num_classes == 2:
            # Binary: BCEWithLogitsLoss
            loss = F.binary_cross_entropy_with_logits(
                logits[:, 0], targets.float(), reduction='mean'
            )
        else:
            # Multi-class: CrossEntropyLoss
            loss = F.cross_entropy(logits, targets.long(), reduction='mean')

        return loss


def get_loss_criterion(num_classes, num_buckets=None):
    """获取损失函数"""
    if num_classes == 2:
        return nn.BCEWithLogitsLoss(reduction='none')
    elif num_classes > 2:
        return nn.CrossEntropyLoss(reduction='none')
    else:
        # 回归：用分桶 loss（后续实现 Bar Distribution）
        raise NotImplementedError("Regression not implemented yet")


class InContextLoss(nn.Module):
    """In-Context Learning 损失：只计算 query 位置的 loss"""

    def __init__(self, num_classes, loss_type='ce'):
        super().__init__()
        self.num_classes = num_classes
        self.loss_type = loss_type

        if loss_type == 'ce':
            if num_classes == 2:
                self.criterion = nn.BCEWithLogitsLoss(reduction='none')
            else:
                self.criterion = nn.CrossEntropyLoss(reduction='none')

    def forward(self, logits, targets, query_mask=None):
        """
        Args:
            logits: (batch, num_classes) query 位置的预测
            targets: (batch,) query 位置的真实标签
            query_mask: 可选，用于忽略某些位置

        Returns:
            loss: scalar
        """
        if self.loss_type == 'ce':
            if self.num_classes == 2:
                loss = self.criterion(logits[:, 0], targets.float())
            else:
                loss = self.criterion(logits, targets.long())
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")

        # 平均
        if query_mask is not None:
            loss = loss * query_mask
            loss = loss.sum() / (query_mask.sum() + 1e-8)
        else:
            loss = loss.mean()

        return loss


if __name__ == '__main__':
    # 测试损失
    batch_size = 32
    num_classes = 3

    logits = torch.randn(batch_size, num_classes)
    targets = torch.randint(0, num_classes, (batch_size,))

    criterion = InContextLoss(num_classes, loss_type='ce')
    loss = criterion(logits, targets)
    print(f"Loss: {loss.item():.4f}")