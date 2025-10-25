# train_tinyvit_kd_kaggle.py
# 注：此脚本在 Kaggle Notebook / 本地都可运行，要求提前生成 teacher logits (.npz)
# 用法示例:
# python train_tinyvit_kd_kaggle.py --logits ./teacher_logits.npz --epochs 50 --batch-size 64

import os                # 标准库：文件/目录操作
import argparse          # 标准库：解析命令行参数
import time              # 标准库：时间测量
import math              # 标准库：数学函数（cos 等）
import numpy as np       # numpy 用于加载 .npz 离线 logits
from tqdm import tqdm    # 进度条
import torch             # PyTorch 主库
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets
from datetime import datetime

teacher_logits_npz_path = './teacher_logits.npz'  # 可修改为你的 logits 文件路径
 
# ---------------------------
# 命令行参数解析函数
# ---------------------------
def parse_args():
    p = argparse.ArgumentParser()
    # 每个 add_argument: 指定命令行参数名称与类型、默认值、说明
    p.add_argument('--logits', type=str, required=True, help='teacher logits path')   # str
    p.add_argument('--data-root', type=str, default='./data', help='CIFAR 数据目录')
    p.add_argument('--epochs', type=int, default=50)
    p.add_argument('--batch-size', type=int, default=128)
    p.add_argument('--num-workers', type=int, default=4)
    p.add_argument('--lr', type=float, default=3e-4)           # base lr
    p.add_argument('--weight-decay', type=float, default=0.05)
    p.add_argument('--warmup-steps', type=int, default=500)   # step 数；也可设为 epochs*steps*比例
    p.add_argument('--alpha', type=float, default=0.7)        # CE 权重，alpha*CE + (1-alpha)*KD
    p.add_argument('--T', type=float, default=4.0)            # KD 温度（温度越大 soft label 更平滑）
    p.add_argument('--save-dir', type=str, default='./kd_ckpt')
    p.add_argument('--device', type=str, default='cuda')
    p.add_argument('--log_path', type=str, default='train_log.txt')
    return p.parse_args()

# ---------------------------
# 自定义 Dataset：把 CIFAR-10 原图与离线 teacher logits 结合
# ---------------------------
class CIFAR10WithLogits(Dataset):
    """
    Dataset 接口：
      - __init__(...): 初始化，加载 CIFAR-10 原始图片与对应的 teacher logits（npz）
      - __len__(): 返回样本数
      - __getitem__(idx): 返回 (img_tensor, label_int, teacher_logits_tensor)
    类型提示：
      - self.base: torchvision.datasets.CIFAR10 实例
      - self.logits: numpy array (N, C)
      - self.transform: torchvision.transforms.Compose 或 None
    """
    def __init__(self, root, logits_npz, train=True, transform=None):
        # 下载并构建原始 CIFAR10（PIL 图片 + label）
        self.base = datasets.CIFAR10(root=root, train=train, download=True)
        # 读取预先生成的离线 logits（numpy .npz）
        arr = np.load(logits_npz)
        # logits: (N, num_classes)，labels: (N,)
        self.logits = arr['logits']
        self.labels = arr['labels']
        # 校验样本数量一致
        assert len(self.base) == len(self.logits), "CIFAR size and logits length must match"
        self.transform = transform

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        # base[idx] 返回 (PIL.Image, int_label)
        img, label = self.base[idx]
        if self.transform:
            img = self.transform(img)                # 转为 tensor 并做增强/归一化
        teacher_logits = torch.from_numpy(self.logits[idx]).float()  # 转为 float tensor
        # 返回 (image_tensor, label_int, teacher_logits_tensor)
        return img, int(label), teacher_logits

# ---------------------------
# 学生模型构建（尝试本地 TinyViT，否则 fallback 到 timm 的 tiny-vit）
# ---------------------------
def build_student(num_classes=10):
    """
    尝试导入本地 TinyViT 实现（如果你 clone 了仓库并设置 PYTHONPATH），
    否则用 timm 提供的 vit_tiny_patch16_224 作为替代（注意不是 TinyViT 原版，但可作学生）。
    返回：
      - model: torch.nn.Module，输出 logits 形状 (B, num_classes)
    """
    try:
        # 这是假设你有本地实现；若有请根据仓库接口替换
        from models.tiny_vit import tiny_vit_5m_224
        model = tiny_vit_5m_224()  # TODO 可自定义参数
        print("Using local TinyViT implementation.")
    except Exception as e:
        # fallback: timm 的 ViT tiny（便于快速测试）
        print("Local TinyViT not found, fallback to timm vit_tiny.")
        import timm
        model = timm.create_model('vit_tiny_patch16_224', pretrained=False, num_classes=num_classes)
    return model

# ---------------------------
# Learning rate scheduler: linear warmup then cosine decay (按 step 更新)
# ---------------------------
class WarmupCosineScheduler:
    """
    Scheduler 按 step 调整 lr：
      - 前 warmup_steps 步线性从 0 -> base_lr
      - 后续做 cosine decay 从 base_lr -> 0
    方法：
      - __init__(optimizer, warmup_steps, total_steps, base_lr)
      - step(): 更新内部计数并设定 optimizer 中的 lr，返回当前 lr（float）
    注意：
      - 该 scheduler 是按 step（optimizer.step）调用，而不是按 epoch
    """
    def __init__(self, optimizer, warmup_steps, total_steps, base_lr):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = max(total_steps, 1)
        self.base_lr = base_lr
        self.step_num = 0

    def step(self):
        self.step_num += 1
        if self.step_num <= self.warmup_steps and self.warmup_steps > 0:
            lr_mult = float(self.step_num) / float(max(1, self.warmup_steps))
        else:
            # cosine decay from base_lr to 0
            t = (self.step_num - self.warmup_steps) / float(max(1, self.total_steps - self.warmup_steps))
            t = min(max(t, 0.0), 1.0)
            lr_mult = 0.5 * (1.0 + math.cos(math.pi * t))
        lr = self.base_lr * lr_mult
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        return lr

# ---------------------------
# 单 epoch 训练函数（蒸馏损失）
# ---------------------------
def train_one_epoch(model, loader, optimizer, device, epoch_idx, scheduler, alpha, T, criterion_ce, criterion_kd):
    """
    训练一个 epoch：
      - model: 学生模型（nn.Module）
      - loader: DataLoader，返回 (imgs, labels, teacher_logits)
      - optimizer: 优化器
      - device: 'cuda' 或 'cpu'
      - alpha: CE 权重（float）
      - T: 温度（float）
      - criterion_ce: CrossEntropyLoss
      - criterion_kd: KLDivLoss (期待 student_logp, teacher_prob)
    返回：平均 loss（float）
    """
    model.train()
    running_loss = 0.0
    total = 0
    # tqdm 自动显示进度条，enumerate(loader) 返回 (batch_idx, batch)
    pbar = tqdm(enumerate(loader), total=len(loader), desc=f'Epoch[{epoch_idx}]')
    for i, (imgs, labels, teacher_logits) in pbar:
        # imgs: tensor (B, C, H, W), labels: tensor (B,), teacher_logits: tensor (B, num_classes)
        imgs = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        teacher_logits = teacher_logits.to(device, non_blocking=True)

        # forward 学生
        logits = model(imgs)  # (B, C_out) where C_out=num_classes

        # 交叉熵损失（硬标签）
        ce_loss = criterion_ce(logits, labels)

        # KD 损失（软标签）
        # student_logp: log softmax(student_logits / T)
        student_logp = nn.functional.log_softmax(logits / T, dim=1)        # (B, C)
        # teacher_prob: softmax(teacher_logits / T)
        teacher_prob = nn.functional.softmax(teacher_logits / T, dim=1)   # (B, C)
        # KLDivLoss expects (log_prob, prob)
        kd_loss = criterion_kd(student_logp, teacher_prob) * (T * T)      # multiply by T^2 (Hinton et al.)
        # 组合总损失：alpha*CE + (1-alpha)*KD
        loss = alpha * ce_loss + (1.0 - alpha) * kd_loss

        # 反向与优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # step scheduler（按 step 更新 LR）
        if scheduler is not None:
            cur_lr = scheduler.step()
        else:
            cur_lr = optimizer.param_groups[0]['lr']

        # 聚合统计
        running_loss += loss.item() * imgs.size(0)
        total += imgs.size(0)
        # 每 50 step 更新一次进度条信息
        if i % 50 == 0:
            pbar.set_postfix({'loss': running_loss / total, 'lr': cur_lr})
    # 返回 epoch 平均 loss
    return running_loss / total

# ---------------------------
# 验证函数（仅硬标签）
# ---------------------------
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
    return correct / total  # 返回准确率

# ---------------------------
# 主函数：组装数据/模型/训练循环
# ---------------------------
def main():
    args = parse_args()
    # 选择设备：如果可用用 args.device，否则回退到 cpu
    device = args.device if torch.cuda.is_available() else 'cpu'
    os.makedirs(args.save_dir, exist_ok=True)
    print("Device:", device)
    if device.startswith('cuda'):
        # 打印 GPU 名称与显存大小（便于调试）
        print(torch.cuda.get_device_name(0), " total mem (GB):", torch.cuda.get_device_properties(0).total_memory / 1024**3)

    # 数据变换：训练集与验证集
    transform_train = transforms.Compose([
        transforms.Resize((224, 224)),                # ResNet/Vit 通常需要 224x224 输入
        transforms.RandomHorizontalFlip(),            # 数据增强：随机水平翻转
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

    # 加载带 logits 的数据集
    train_set = CIFAR10WithLogits(root=args.data_root, logits_npz=args.logits, train=True, transform=transform_train)
    val_set = CIFAR10WithLogits(root=args.data_root, logits_npz=args.logits, train=False, transform=transform_val)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    # 构建学生模型并放到 device
    model = build_student(num_classes=10).to(device)

    # 损失与优化器
    criterion_ce = nn.CrossEntropyLoss(label_smoothing=0.1)   # 带标签平滑，有助泛化
    criterion_kd = nn.KLDivLoss(reduction='batchmean')       # KLDivLoss for KD (expects log-prob and prob)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # 计算总 step 数并构建 scheduler（按 step 更新）
    total_steps = args.epochs * len(train_loader)
    scheduler = WarmupCosineScheduler(optimizer, warmup_steps=args.warmup_steps, total_steps=total_steps, base_lr=args.lr)

    # 训练循环：保存检查点并输出最优模型
    best_acc = 0.0
    start = time.time()
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss = train_one_epoch(model, train_loader, optimizer, device, epoch, scheduler, args.alpha, args.T, criterion_ce, criterion_kd)
        t1 = time.time()
        print(f"Epoch {epoch} finished, train_loss={train_loss:.4f}, time={(t1-t0):.1f}s")
        # 验证并保存
        val_acc = validate(model, val_loader, device)
        print(f"Validation accuracy: {val_acc:.4f}")
        # 保存 checkpoint（包含模型状态与优化器状态）
        ckpt = {'epoch': epoch, 'model_state': model.state_dict(), 'optim_state': optimizer.state_dict(), 'val_acc': val_acc}
        torch.save(ckpt, os.path.join(args.save_dir, f'ckpt_epoch_{epoch}.pth'))
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), os.path.join(args.save_dir, 'best_student.pth'))
            print("Saved best_student.pth on epoch {epoch} with acc: {best_acc:.4f}")
        with open(args.log_path, 'a') as f:
                f.write(f"{epoch},{train_loss:.4f},{val_acc:.4f},{datetime.now().strftime('%Y-%m-%d %H:%M:%S')},{(t1-t0):.1f}s\n")
    total_time = time.time() - start
    print(f"Training finished. Total time: {total_time/3600:.2f} hours. Best val acc: {best_acc:.4f}")

if __name__ == '__main__':
    main()
