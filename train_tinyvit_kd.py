# train_tinyvit_kd.py
# 用法示例:
# !python train_tinyvit_kd.py --logits-path ./teacher_logits.npz --batch-size 32 --epochs 50 --save-dir ./kd_checkpoints

import os
import argparse
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--logits-path', type=str, required=True, help='path to npz file produced by generate_teacher_logits.py')
    p.add_argument('--batch-size', type=int, default=32)
    p.add_argument('--num-workers', type=int, default=2)
    p.add_argument('--epochs', type=int, default=50)
    p.add_argument('--lr', type=float, default=3e-4)
    p.add_argument('--weight-decay', type=float, default=0.05)
    p.add_argument('--alpha', type=float, default=0.7, help='weight for CE loss; (1-alpha) for KD loss')
    p.add_argument('--T', type=float, default=4.0, help='KD temperature')
    p.add_argument('--save-dir', type=str, default='./kd_checkpoints')
    p.add_argument('--device', type=str, default='cuda')
    return p.parse_args()

# 自定义 Dataset：读取 CIFAR-10 原始图片 + 预先生成的 teacher logits
class CIFAR10WithLogits(Dataset):
    def __init__(self, root, logits_npz_path, train=True, transform=None):
        self.transform = transform
        # load CIFAR-10 dataset via torchvision
        self.base = datasets.CIFAR10(root=root, train=train, download=True)
        # load logits npz
        arr = np.load(logits_npz_path)
        self.all_logits = arr['logits']  # shape (N, C)
        self.all_labels = arr['labels']  # shape (N,)
        assert len(self.base) == len(self.all_labels) == len(self.all_logits)
    def __len__(self):
        return len(self.base)
    def __getitem__(self, idx):
        img, label = self.base[idx]
        if self.transform:
            img = self.transform(img)
        teacher_logits = torch.from_numpy(self.all_logits[idx]).float()
        return img, int(label), teacher_logits

def build_student_model(num_classes=10):
    # 优先尝试导入官方 tinyvit（如果你已 clone 官方仓库并在 PYTHONPATH 下）
    try:
        from tinyvit import build_tiny_vit  # 假设官方仓库中的构建函数
        # 举例调用：build_tiny_vit(config) ; 这里我们假设存在一个简便的工厂函数
        # 替换下面为你的仓库中实际的构造方式
        model = build_tiny_vit('tinyvit_5m')  # 伪代码：按官方仓库的接口修改
        print('-> Using official TinyViT from local repo.')
    except Exception as e:
        print('-> Official TinyViT not found or failed to import; fallback to timm vit_tiny.')
        import timm
        model = timm.create_model('vit_tiny_patch16_224', pretrained=False, num_classes=num_classes)
    return model

def train_one_epoch(model, loader, optimizer, device, epoch, alpha, T, criterion_ce, criterion_kd):
    model.train()
    total_loss = 0.0
    pbar = tqdm(enumerate(loader), total=len(loader), desc=f'Epoch {epoch}')
    for i, (imgs, labels, teacher_logits) in pbar:
        imgs = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        teacher_logits = teacher_logits.to(device, non_blocking=True)  # shape (B, C)
        logits = model(imgs)  # shape (B, C)
        # CE loss (label smoothing if desired)
        ce_loss = criterion_ce(logits, labels)
        # KD loss (KL between softened distributions)
        # Use log_softmax on student and softmax on teacher
        student_log_prob = nn.functional.log_softmax(logits / T, dim=1)
        teacher_prob = nn.functional.softmax(teacher_logits / T, dim=1)
        kd_loss = criterion_kd(student_log_prob, teacher_prob) * (T * T)
        loss = alpha * ce_loss + (1.0 - alpha) * kd_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * imgs.size(0)
        pbar.set_postfix({'loss': total_loss / ((i+1) * imgs.size(0))})
    return total_loss / len(loader.dataset)

def validate(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for imgs, labels, _ in loader:
            imgs = imgs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            logits = model(imgs)
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return correct / total

def main():
    args = parse_args()
    device = args.device if torch.cuda.is_available() else 'cpu'
    os.makedirs(args.save_dir, exist_ok=True)

    # transforms（同生成 logits 时保持一致）
    transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225))
    ])

    # Datasets / Dataloaders
    train_set = CIFAR10WithLogits(root='./data', logits_npz_path=args.logits_path, train=True, transform=transform)
    val_set = CIFAR10WithLogits(root='./data', logits_npz_path=args.logits_path, train=False, transform=transform)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    # model
    model = build_student_model(num_classes=10)
    model.to(device)

    # losses, optimizer
    criterion_ce = nn.CrossEntropyLoss(label_smoothing=0.1)  # label smoothing 推荐 0.1
    criterion_kd = nn.KLDivLoss(reduction='batchmean')  # expects log-prob & prob
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_acc = 0.0
    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, device, epoch, args.alpha, args.T, criterion_ce, criterion_kd)
        val_acc = validate(model, val_loader, device)
        print(f'Epoch {epoch} train_loss={train_loss:.4f} val_acc={val_acc:.4f}')
        # save checkpoint
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_acc': val_acc
        }, os.path.join(args.save_dir, f'checkpoint_epoch_{epoch}.pth'))
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), os.path.join(args.save_dir, 'best_student.pth'))
            print('-> saved best_student.pth, acc', best_acc)

if __name__ == '__main__':
    main()