# ===============================================================
# generate_teacher_logits_optimized.py
# 优化版：支持分阶段微调、RandAugment、CosineLR、Label Smoothing
# ===============================================================
# !python generate_teacher_logits_optimized.py --model resnet50 --pretrained --finetune-epochs 20 --lr 3e-4 --batch-size 256 --device cuda
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
from torchvision.transforms import RandAugment


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--batch-size', type=int, default=128)
    p.add_argument('--num-workers', type=int, default=2)
    p.add_argument('--model', type=str, default='resnet50', choices=['resnet50'])
    p.add_argument('--resume', type=str, default=None, help='path to resume checkpoint')
    p.add_argument('--pretrained', action='store_true', help='use ImageNet pretrained weights')
    p.add_argument('--finetune-epochs', type=int, default=100, help='number of fine-tune epochs')
    p.add_argument('--lr', type=float, default=3e-4)
    p.add_argument('--save-path', type=str, default='./teacher_logits.npz')
    p.add_argument('--device', type=str, default='cuda')
    p.add_argument('--log-path', type=str, default='./train_log_optimized.txt')
    return p.parse_args()


def build_teacher(model_name='resnet50', pretrained=True, num_classes=10):
    if model_name == 'resnet50':
        model = torchvision.models.resnet50(
            weights='IMAGENET1K_V1' if pretrained else None)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    else:
        raise NotImplementedError
    return model


def get_cifar10_dataloaders(batch_size=128, num_workers=2, resize=224):
    train_tf = transforms.Compose([
        transforms.Resize((resize, resize)),
        transforms.RandomHorizontalFlip(),
        RandAugment(num_ops=2, magnitude=9),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406),
                             std=(0.229, 0.224, 0.225)),
        transforms.RandomErasing(p=0.25)
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


def finetune_teacher(model, train_loader, val_loader, device='cuda', epochs=100, lr=3e-4, log_path='train_log.txt'):
    model.to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.05)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    os.makedirs('checkpoints', exist_ok=True)
    best_acc = 0.0
    best_path = None

    with open(log_path, 'a') as f:
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

        scheduler.step()
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

        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        duration = time.time() - start_t

        # 保存模型
        last_path = f"checkpoints/last_teacher_epoch{epoch}.pth"
        torch.save({'model_state': model.state_dict()}, last_path)

        if val_acc > best_acc:
            best_acc = val_acc
            if best_path and os.path.exists(best_path):
                os.remove(best_path)
            best_path = f"checkpoints/best_teacher_epoch{epoch}_acc{val_acc:.4f}.pth"
            torch.save({'model_state': model.state_dict()}, best_path)
            print(f"✨ New best model saved: {best_path}")

        with open(log_path, 'a') as f:
            f.write(f"{epoch},{train_loss:.4f},{val_acc:.4f},{now},{duration:.2f}\n")

        print(f"[Epoch {epoch}] Loss={train_loss:.4f} | ValAcc={val_acc:.4f} | Duration={duration:.1f}s")

    print(f"✅ Finetune finished. Best Acc={best_acc:.4f}")
    return model


def staged_finetune(model, train_loader, val_loader, device='cuda', total_epochs=100, lr=3e-4, log_path='train_log.txt'):
    print("🧊 Stage 1: train only the FC layer ...")
    for name, param in model.named_parameters():
        param.requires_grad = False
    for name, param in model.fc.named_parameters():
        param.requires_grad = True
    finetune_teacher(model, train_loader, val_loader, device, epochs=10, lr=1e-3, log_path=log_path)

    print("🔥 Stage 2: unfreeze all layers and fine-tune ...")
    for param in model.parameters():
        param.requires_grad = True
    return finetune_teacher(model, train_loader, val_loader, device, epochs=total_epochs, lr=lr, log_path=log_path)


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
    return np.concatenate(all_logits), np.concatenate(all_labels)


def main():
    args = parse_args()
    device = args.device if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    train_loader, full_train_loader, val_loader = get_cifar10_dataloaders(
        batch_size=args.batch_size, num_workers=args.num_workers)

    teacher = build_teacher(model_name=args.model, pretrained=args.pretrained, num_classes=10)

    if args.resume and os.path.exists(args.resume):
        ckpt = torch.load(args.resume, map_location='cpu')
        state_dict = ckpt.get('model_state', ckpt)
        teacher.load_state_dict(state_dict, strict=False)
        print(f"✅ Loaded weights from {args.resume}")

    if args.finetune_epochs > 0:
        teacher = staged_finetune(
            teacher, train_loader, val_loader,
            device=device, total_epochs=args.finetune_epochs,
            lr=args.lr, log_path=args.log_path
        )

    print("-> Generating teacher logits ...")
    # 生成训练集 logits
    logits_train, labels_train = generate_logits(teacher, full_train_loader, device=device)
    np.savez_compressed("teacher_logits_train.npz", logits=logits_train, labels=labels_train)

    # 生成测试集 logits
    logits_val, labels_val = generate_logits(teacher, val_loader, device=device)
    np.savez_compressed("teacher_logits_val.npz", logits=logits_val, labels=labels_val)

    print(f"✅ Saved logits+labels to {args.save_path}")
    print(f"logits shape={logits_train.shape}, labels shape={labels_train.shape}")
    print(f"Approx size: {logits_train.nbytes/1e6:.2f} MB")
    print(f"logits shape={logits_val.shape}, labels shape={logits_val.shape}")
    print(f"Approx size: {logits_val.nbytes/1e6:.2f} MB")


if __name__ == '__main__':
    main()
