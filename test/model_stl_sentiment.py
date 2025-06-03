import logging
import torch
from linmult import LinMulT, load_config

logging.basicConfig(level=logging.DEBUG)

config = load_config("configs/STL/sentiment/model.yaml")
model = LinMulT(config)

x = [
    torch.randn(1, 500, 768),
    torch.randn(1, 300, 1024),
    torch.randn(1, 120, 768)
]

y = model(x)

print(model)
print([head.shape for head in y])