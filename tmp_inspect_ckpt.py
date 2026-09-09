import torch, json
p = 'v6_smartphone_idr/checkpoints/best_model_v6.pth'
ckpt = torch.load(p, map_location='cpu', weights_only=False)
print('ckpt keys:', list(ckpt.keys()))
sd = ckpt['model_state_dict']
print('--- state_dict ---')
for k, v in sd.items():
    print(f'{k}: {list(v.shape)}')
print('--- config ---')
print(json.dumps(ckpt.get('config', {}), indent=2))
print('--- training ---')
print(json.dumps(ckpt.get('training', {}), indent=2))
