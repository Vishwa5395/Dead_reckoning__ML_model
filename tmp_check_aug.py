import ast
with open('v6_smartphone_idr/src/augmentation_v6.py', 'r', encoding='utf-8') as f:
    src = f.read()
print('char count:', len(src))
try:
    ast.parse(src)
    print('SYNTAX OK')
except SyntaxError as e:
    print('SYNTAX ERROR:', e)
# Test imports and the SMOTE function on tiny data
import numpy as np
import sys
sys.path.insert(0, '.')
from v6_smartphone_idr.src.augmentation_v6 import IMUAugmentor, compute_maneuver_weights, generate_smote_upsamples

# Small quick test
X = np.random.rand(100, 20, 6).astype(np.float32)
dv = np.random.rand(100, 1).astype(np.float32)
w = np.random.rand(100, 1).astype(np.float32)
z = (np.random.rand(100, 1) > 0.9).astype(np.float32)
ba = np.random.rand(100, 1).astype(np.float32)
bw = np.random.rand(100, 1).astype(np.float32)
w_phys = (np.random.rand(100, 1) - 0.5).astype(np.float32)
dv_phys = (np.random.rand(100, 1) - 0.5).astype(np.float32)

ws = compute_maneuver_weights(w_phys, dv_phys, z)
print('weights shape', ws.shape, 'mean', float(ws.mean()))

aug = IMUAugmentor()
xw = np.random.rand(20, 6).astype(np.float32)
xaug = aug.augment_window(xw)
print('augmented', xaug.shape, xaug.dtype)

X2, dv2, w2, z2, ba2, bw2 = generate_smote_upsamples(X, dv, w, z, ba, bw, w_phys, dv_phys, tau_turn=1.0)
print('after smoke', X2.shape, dv2.shape)
