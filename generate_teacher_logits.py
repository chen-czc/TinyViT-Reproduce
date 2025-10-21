# generate_teacher_logits.py
# 用法示例（在 Kaggle notebook cell 中运行）：
# !python generate_teacher_logits.py --batch-size 64 --finetune-epochs 0 --save-path /kaggle/working/teacher_logits.npz

import os
import argparse
import numpy as np
from tqdm import tqdm
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
    p.add_argument('--pretrained', action='store_true', help='use ImageNet pretrained weights (default true if not finetune)')
    p.add_argument('--finetune-epochs', type=int, default=0,
                   help='If >0, fine-tune the teacher on CIFAR-10 for this many epochs before generating logits.')
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--save-path', type=str, default='./teacher_logits.npz',
                   help='path to save a numpy archive containing logits and labels')
    p.add_argument('--device', type=str, default='cuda')
    return p.parse_args()

def build_teacher(model_name='resnet50', pretrained=True, num_classes=10):
    # 使用 torchvision 的 ResNet50
    if model_name == 'resnet50':
        # 加载 ImageNet 预训练权重（新 torchvision API）
        model = torchvision.models.resnet50(weights='IMAGENET1K_V1' if pretrained else None)
        # 将全连接头改为 num_classes（用于 CIFAR-10 微调或直接训练）
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    else:
        raise NotImplementedError
    return model

def get_cifar10_dataloaders(batch_size=64, num_workers=2, resize=224):
    # 注意：我们把 CIFAR-10 图像放到 224x224 以兼容 ImageNet 预训练模型
    train_transform = transforms.Compose([
        transforms.Resize((resize, resize)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225))
    ])
    test_transform = transforms.Compose([
        transforms.Resize((resize, resize)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225))
    ])
    train_set = datasets.CIFAR10(root='./mydata', train=True, transform=train_transform, download=True)
    val_set = datasets.CIFAR10(root='./mydata', train=False, transform=test_transform, download=True)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    full_train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    return train_loader, full_train_loader, val_loader, len(train_set), train_set

def finetune_teacher(model, train_loader, val_loader, device='cuda', epochs=3, lr=1e-3):
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss()
    for epoch in range(epochs):
        model.train()
        pbar = tqdm(train_loader, desc=f'Finetune epoch {epoch+1}/{epochs}')
        for imgs, labels in pbar:
            imgs = imgs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            logits = model(imgs)
            loss = criterion(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            pbar.set_postfix({'loss': loss.item()})
        # 可选：在 val 集上评估
        model.eval()
        acc = 0
        tot = 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs = imgs.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                logits = model(imgs)
                preds = logits.argmax(dim=1)
                acc += (preds == labels).sum().item()
                tot += labels.size(0)
        print(f'Finetune epoch {epoch+1} val acc {acc/tot:.4f}')
    return model

def generate_logits(model, dataloader, device='cuda'):
    model.eval()
    model.to(device)
    all_logits = []
    all_labels = []
    with torch.no_grad():
        for imgs, labels in tqdm(dataloader, desc='Generating logits'):
            imgs = imgs.to(device, non_blocking=True)
            logits = model(imgs)  # shape (B, C)
            all_logits.append(logits.cpu().numpy())
            all_labels.append(labels.numpy())
    all_logits = np.concatenate(all_logits, axis=0)  # (N, C)
    all_labels = np.concatenate(all_labels, axis=0)  # (N,)
    return all_logits, all_labels

def main():
    args = parse_args()
    device = args.device if torch.cuda.is_available() else 'cpu'

    # 1) 构建数据加载器
    train_loader, full_train_loader, val_loader, train_n, train_dataset = get_cifar10_dataloaders(
        batch_size=args.batch_size, num_workers=args.num_workers, resize=224)

    # 2) 构建 teacher
    teacher = build_teacher(model_name=args.model, pretrained=args.pretrained, num_classes=10)

    # 3) 可选：微调 teacher
    if args.finetune_epochs > 0:
        print('-> Finetuning teacher on CIFAR-10 first ...')
        teacher = finetune_teacher(teacher, train_loader, val_loader, device=device, epochs=args.finetune_epochs, lr=args.lr)
        # 保存微调的权重以便复用
        torch.save({'model_state': teacher.state_dict()}, 'teacher_finetuned.pth')
        print('-> finetuned teacher saved to teacher_finetuned.pth')

    # 4) 生成 logits（对训练集做一次遍历）
    print('-> Generating logits for full training set ...')
    logits, labels = generate_logits(teacher, full_train_loader, device=device)
    print('-> logits shape', logits.shape, 'labels shape', labels.shape)

    # 5) 保存到 npz，节省文件数量并便于加载
    os.makedirs(os.path.dirname(args.save_path) or '.', exist_ok=True)
    np.savez_compressed(args.save_path, logits=logits, labels=labels)
    print(f'-> Saved logits+labels to {args.save_path}')
    print('Sizes: logits bytes approx:', logits.nbytes, 'labels bytes approx:', labels.nbytes)

if __name__ == '__main__':
    main()