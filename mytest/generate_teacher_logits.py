# ===============================================================
# generate_teacher_logits.py
# 功能：支持从本地加载权重、保存最佳模型与日志记录
# ===============================================================

import os
import argparse
import numpy as np
from tqdm import tqdm
from datetime import datetime
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms, datasets


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--batch-size', type=int, default=64)
    p.add_argument('--num-workers', type=int, default=2)
    p.add_argument('--model', type=str, default='resnet50', choices=['resnet50'])
    p.add_argument('--resume', type=str, default=None, help='path to resume checkpoint')
    p.add_argument('--pretrained', action='store_true', help='use ImageNet pretrained weights')
    p.add_argument('--finetune-epochs', type=int, default=0, help='number of fine-tune epochs')
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--save-path', type=str, default='./teacher_logits.npz')
    p.add_argument('--device', type=str, default='cuda')
    p.add_argument('--log-path', type=str, default='./train_log.txt')
    return p.parse_args()


def build_teacher(model_name='resnet50', pretrained=True, num_classes=10):
    if model_name == 'resnet50':
        model = torchvision.models.resnet50(
            weights='IMAGENET1K_V1' if pretrained else None)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    else:
        raise NotImplementedError
    return model


def get_cifar10_dataloaders(batch_size=64, num_workers=2, resize=224):
    train_tf = transforms.Compose([
        transforms.Resize((resize, resize)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406),
                             std=(0.229, 0.224, 0.225))
    ])
    test_tf = transforms.Compose([
        transforms.Resize((resize, resize)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406),
                             std=(0.229, 0.224, 0.225))
    ])
    train_set = datasets.CIFAR10(root='./data', train=True, transform=train_tf, download=True)
    val_set = datasets.CIFAR10(root='./data', train=False, transform=test_tf, download=True)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True)
    full_train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=False,
                                   num_workers=num_workers, pin_memory=True)
    return train_loader, full_train_loader, val_loader


def finetune_teacher(model, train_loader, val_loader, device='cuda', epochs=3, lr=1e-3, log_path='train_log.txt'):
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss()

    os.makedirs('checkpoints', exist_ok=True)
    best_acc = 0.0
    best_path = None

    with open(log_path, 'w') as f:
        f.write("Epoch,Loss,ValAcc,Time,Duration(s)\n")

    for epoch in range(1, epochs + 1):
        start_t = time.time()
        model.train()
        total_loss, total_samples = 0, 0
        pbar = tqdm(train_loader, desc=f"[Finetune] Epoch {epoch}/{epochs}")

        for imgs, labels in pbar:
            imgs, labels = imgs.to(device), labels.to(device)
            logits = model(imgs)
            loss = criterion(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * imgs.size(0)
            total_samples += imgs.size(0)
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        train_loss = total_loss / total_samples

        # 验证阶段
        model.eval()
        correct, tot = 0, 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                preds = model(imgs).argmax(dim=1)
                correct += (preds == labels).sum().item()
                tot += labels.size(0)
        val_acc = correct / tot

        # 保存模型
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        duration = time.time() - start_t

        # 最后模型
        last_path = f"checkpoints/last_teacher_epoch{epoch}.pth"
        torch.save({'model_state': model.state_dict()}, last_path)

        # 最佳模型
        if val_acc > best_acc:
            best_acc = val_acc
            if best_path and os.path.exists(best_path):
                os.remove(best_path)  # 删除旧的最佳权重
            best_path = f"checkpoints/best_teacher_epoch{epoch}_acc{val_acc:.4f}.pth"
            torch.save({'model_state': model.state_dict()}, best_path)
            print(f"✨ New best model saved: {best_path}")

        # 写日志
        with open(log_path, 'a') as f:
            f.write(f"{epoch},{train_loss:.4f},{val_acc:.4f},{now},{duration:.2f}\n")

        print(f"[Epoch {epoch}] Loss={train_loss:.4f} | ValAcc={val_acc:.4f} | Duration={duration:.1f}s")

    print(f"✅ Finetune finished. Best Acc={best_acc:.4f}")
    return model


def generate_logits(model, dataloader, device='cuda'):
    model.eval()
    model.to(device)
    all_logits, all_labels = [], []
    with torch.no_grad():
        for imgs, labels in tqdm(dataloader, desc="Generating logits"):
            imgs = imgs.to(device)
            logits = model(imgs)
            all_logits.append(logits.cpu().numpy())
            all_labels.append(labels.numpy())
    all_logits = np.concatenate(all_logits)
    all_labels = np.concatenate(all_labels)
    return all_logits, all_labels


def main():
    args = parse_args()
    device = args.device if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    train_loader, full_train_loader, val_loader = get_cifar10_dataloaders(
        batch_size=args.batch_size, num_workers=args.num_workers)

    teacher = build_teacher(model_name=args.model, pretrained=args.pretrained, num_classes=10)

    # 从本地加载权重
    if args.resume and os.path.exists(args.resume):
        ckpt = torch.load(args.resume, map_location='cpu')
        state_dict = ckpt.get('model_state', ckpt)
        teacher.load_state_dict(state_dict, strict=False)
        print(f"✅ Loaded weights from {args.resume}")

    # 微调阶段
    if args.finetune_epochs > 0:
        teacher = finetune_teacher(
            teacher, train_loader, val_loader,
            device=device, epochs=args.finetune_epochs,
            lr=args.lr, log_path=args.log_path
        )

    # 生成 logits
    print("-> Generating teacher logits ...")
    logits, labels = generate_logits(teacher, full_train_loader, device=device)
    os.makedirs(os.path.dirname(args.save_path) or '.', exist_ok=True)
    np.savez_compressed(args.save_path, logits=logits, labels=labels)
    print(f"✅ Saved logits+labels to {args.save_path}")
    print(f"logits shape={logits.shape}, labels shape={labels.shape}")
    print(f"Approx size: {logits.nbytes/1e6:.2f} MB")


if __name__ == '__main__':
    main()