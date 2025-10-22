"""Model Inference."""
import warnings
from pydantic.warnings import PydanticDeprecatedSince20
warnings.filterwarnings("ignore", category=PydanticDeprecatedSince20)
warnings.filterwarnings("ignore", module="pydantic._internal._generate_schema")

import torch
import numpy as np
from PIL import Image

from models.tiny_vit import tiny_vit_21m_224
from data import build_transform, imagenet_classnames
from config import get_config



config = get_config()


# Build model
model = tiny_vit_21m_224(pretrained=True)
model.eval()

# Load Image
fname = './.figure/cat.jpg'
image = Image.open(fname)
transform = build_transform(is_train=False, config=config)

# (1, 3, img_size, img_size)
batch = transform(image)[None]

with torch.no_grad():
    logits = model(batch)

# print top-5 classification names
probs = torch.softmax(logits, -1)
scores, inds = probs.topk(5, largest=True, sorted=True)

# 将推理结果保存到文件
output_file = 'inf_result.txt'

with open(output_file, 'w') as f:
    f.write('=' * 30 + '\n')
    f.write(fname + '\n')
    for score, ind in zip(scores[0].numpy(), inds[0].numpy()):
        f.write(f'{imagenet_classnames[ind]}: {score:.2f}\n')

print('=' * 30)
print(fname)
for score, ind in zip(scores[0].numpy(), inds[0].numpy()):
    print(f'{imagenet_classnames[ind]}: {score:.2f}')
