# train_tinyvit_kd_kaggle.py
# -------------------------------------------------------
# 用于在 CIFAR-10 上对 TinyViT（或 ViT-tiny）进行知识蒸馏训练
# 可运行在 Kaggle Notebook 或本地
# -------------------------------------------------------

import os
import argparse
import time
import math
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets
from datetime import datetime


# ---------------------------
# 命令行参数解析
# ---------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--logits', type=str, required=True, help='teacher logits .npz 路径')
    p.add_argument('--data-root', type=str, default='./data', help='CIFAR10 数据目录')
    p.add_argument('--epochs', type=int, default=50)
    p.add_argument('--batch-size', type=int, default=128)
    p.add_argument('--num-workers', type=int, default=4)
    p.add_argument('--lr', type=float, default=3e-4)
    p.add_argument('--weight-decay', type=float, default=0.05)
    p.add_argument('--warmup-steps', type=int, default=500)
    p.add_argument('--alpha', type=float, default=0.7)
    p.add_argument('--T', type=float, default=4.0)
    p.add_argument('--save-dir', type=str, default='./kd_ckpt')
    p.add_argument('--device', type=str, default='cuda')
    p.add_argument('--log_path', type=str, default='train_log.txt')
    return p.parse_args()


# ---------------------------
# Dataset：CIFAR10 + teacher logits
# ---------------------------
class CIFAR10WithLogits(Dataset):
    def __init__(self, root, logits_npz, train=True, transform=None):
        # 原始 CIFAR10 数据
        self.base = datasets.CIFAR10(root=root, train=train, download=True)
        self.transform = transform

        # 如果 logits 文件存在，加载
        if os.path.exists(logits_npz):
            arr = np.load(logits_npz)
            self.logits = arr['logits']
            self.labels = arr['labels']
            print(f"[INFO] Loaded teacher logits from {logits_npz}")
        else:
            self.logits = None
            self.labels = np.array([x[1] for x in self.base])  # fallback 使用 CIFAR10 标签

        print(f"[INFO] CIFAR10 train={train}, base_len={len(self.base)}, logits_len={len(self.logits) if self.logits is not None else 'None'}")

        # 如果存在 logits，则校验样本数量是否匹配
        if self.logits is not None:
            assert len(self.base) == len(self.logits), "CIFAR10 样本数与 logits 数量不匹配"

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        img, label = self.base[idx]
        if self.transform:
            img = self.transform(img)
        if self.logits is not None:
            teacher_logits = torch.from_numpy(self.logits[idx]).float()
        else:
            teacher_logits = torch.zeros(10)
        return img, int(label), teacher_logits


# ---------------------------
# 学生模型构建
# ---------------------------
def build_student(num_classes=10):
    try:
        from models.tiny_vit import tiny_vit_5m_224
        model = tiny_vit_5m_224(num_classes=num_classes)
        print("[INFO] Using local TinyViT implementation.")
    except Exception:
        import timm
        print("[WARN] Local TinyViT not found, fallback to timm vit_tiny_patch16_224.")
        model = timm.create_model('vit_tiny_patch16_224', pretrained=False, num_classes=num_classes)
    return model


# ---------------------------
# Warmup + Cosine Scheduler
# ---------------------------
class WarmupCosineScheduler:
    def __init__(self, optimizer, warmup_steps, total_steps, base_lr):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = max(total_steps, 1)
        self.base_lr = base_lr
        self.step_num = 0

    def step(self):
        self.step_num += 1
        if self.step_num <= self.warmup_steps:
            lr_mult = self.step_num / float(max(1, self.warmup_steps))
        else:
            t = (self.step_num - self.warmup_steps) / float(max(1, self.total_steps - self.warmup_steps))
            t = min(max(t, 0.0), 1.0)
            lr_mult = 0.5 * (1.0 + math.cos(math.pi * t))
        lr = self.base_lr * lr_mult
        for g in self.optimizer.param_groups:
            g['lr'] = lr
        return lr


# ---------------------------
# 单 epoch 训练（含 KD）
# ---------------------------
def train_one_epoch(model, loader, optimizer, device, epoch, scheduler, alpha, T, ce_loss_fn, kd_loss_fn):
    model.train()
    running_loss = 0.0
    total = 0
    pbar = tqdm(loader, desc=f"Epoch {epoch}")
    for imgs, labels, teacher_logits in pbar:
        imgs = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        teacher_logits = teacher_logits.to(device, non_blocking=True)

        # forward
        logits = model(imgs)

        # loss 计算
        ce_loss = ce_loss_fn(logits, labels)
        student_logp = nn.functional.log_softmax(logits / T, dim=1)
        teacher_prob = nn.functional.softmax(teacher_logits / T, dim=1)
        kd_loss = kd_loss_fn(student_logp, teacher_prob) * (T * T)
        loss = alpha * ce_loss + (1 - alpha) * kd_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 更新 lr
        cur_lr = scheduler.step() if scheduler else optimizer.param_groups[0]['lr']

        running_loss += loss.item() * imgs.size(0)
        total += imgs.size(0)
        if total > 0:
            pbar.set_postfix({'loss': running_loss / total, 'lr': cur_lr})
    return running_loss / total


# ---------------------------
# 验证函数
# ---------------------------
def validate(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for imgs, labels, _ in loader:
            imgs = imgs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            preds = model(imgs).argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return correct / total


# ---------------------------
# 主函数
# ---------------------------
def main():
    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)
    device = args.device if torch.cuda.is_available() else 'cpu'
    print(f"[INFO] Using device: {device}")

    # 数据增强
    transform_train = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandAugment(num_ops=2, magnitude=9),
        transforms.RandomErasing(p=0.25),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    ])
    transform_val = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    ])

    # 数据集加载
    train_set = CIFAR10WithLogits(root=args.data_root, logits_npz=args.logits, train=True, transform=transform_train)
    val_set = CIFAR10WithLogits(root=args.data_root, logits_npz=args.logits, train=False, transform=transform_val)

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True)

    # 模型与优化器
    model = build_student(num_classes=10).to(device)
    ce_loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)
    kd_loss_fn = nn.KLDivLoss(reduction='batchmean')
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    total_steps = args.epochs * len(train_loader)
    scheduler = WarmupCosineScheduler(optimizer, args.warmup_steps, total_steps, args.lr)

    # 训练循环
    best_acc = 0.0
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss = train_one_epoch(model, train_loader, optimizer, device, epoch,
                                     scheduler, args.alpha, args.T, ce_loss_fn, kd_loss_fn)
        val_acc = validate(model, val_loader, device)
        t1 = time.time()
        print(f"[Epoch {epoch}] train_loss={train_loss:.4f}, val_acc={val_acc:.4f}, time={(t1-t0):.1f}s")

        # 保存 checkpoint
        ckpt = {
            'epoch': epoch,
            'model_state': model.state_dict(),
            'optimizer_state': optimizer.state_dict(),
            'val_acc': val_acc
        }
        torch.save(ckpt, os.path.join(args.save_dir, f'ckpt_epoch_{epoch}.pth'))
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), os.path.join(args.save_dir, 'best_student.pth'))
            print(f"[INFO] Saved best model with acc={best_acc:.4f}")

        # 写日志
        with open(args.log_path, 'a') as f:
            f.write(f"{epoch},{train_loss:.4f},{val_acc:.4f},{datetime.now().strftime('%Y-%m-%d %H:%M:%S')},{(t1-t0):.1f}s\n")

    print(f"[DONE] Total best val acc = {best_acc:.4f}")


if __name__ == '__main__':
    main()
