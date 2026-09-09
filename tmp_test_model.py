import torch
from v6_smartphone_idr.src.models_v6 import PINODeadReckoningNetV6

m = PINODeadReckoningNetV6()
n = sum(p.numel() for p in m.parameters() if p.requires_grad)
print('params', n)
print('budget ok', n <= 25000)
x = torch.randn(2, 20, 6)
out = m(x)
print('outputs', [tuple(o.shape) for o in out])
