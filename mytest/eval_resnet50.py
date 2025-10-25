import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torchvision.models import resnet50
import os
from tqdm import tqdm

# =====================================================
# 配置区域
# =====================================================

# ✅ 权重文件路径（可以是本地路径或 None 使用随机初始化）
# 示例：
# WEIGHTS_PATH = "/kaggle/working/resnet50_cifar10.pth"
# 若使用官方 ImageNet 预训练权重，则设置为 "imagenet"
WEIGHTS_PATH = "/kaggle/working/TinyViT-Reproduce/teacher_finetuned.pth"   # 可修改

# 是否使用 GPU
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 128

# =====================================================
# 数据加载
# =====================================================

print("📦 加载 CIFAR-10 测试集中 ...")

transform_test = transforms.Compose([
    transforms.Resize(224),   # ResNet50 输入 224x224
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

testset = torchvision.datasets.CIFAR10(
    root="./data",
    train=False,
    download=True,
    transform=transform_test
)
testloader = torch.utils.data.DataLoader(
    testset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=2
)

# =====================================================
# 模型加载
# =====================================================

print("🧠 初始化 ResNet50 模型 ...")

if WEIGHTS_PATH == "imagenet":
    model = resnet50(weights="IMAGENET1K_V1")  # torchvision 官方预训练
else:
    model = resnet50(weights=None)

# CIFAR10 有 10 类，需修改最后的全连接层
model.fc = nn.Linear(model.fc.in_features, 10)

# 如果有自定义权重则加载
if WEIGHTS_PATH and os.path.isfile(WEIGHTS_PATH) and WEIGHTS_PATH != "imagenet":
    state_dict = torch.load(WEIGHTS_PATH, map_location="cpu")
    model.load_state_dict(state_dict)
    print(f"✅ 已加载权重：{WEIGHTS_PATH}")
elif WEIGHTS_PATH == "imagenet":
    print("✅ 使用 ImageNet 预训练权重")
else:
    print("⚠️ 未加载任何权重，模型为随机初始化")

model = model.to(DEVICE)
model.eval()

# =====================================================
# 推理与评估
# =====================================================

print("🚀 开始在 CIFAR-10 测试集上验证 ...")
correct = 0
total = 0

with torch.no_grad():
    for images, labels in tqdm(testloader):
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

acc = 100 * correct / total
print(f"\n✅ CIFAR-10 测试集准确率: {acc:.2f}% ({correct}/{total})")
