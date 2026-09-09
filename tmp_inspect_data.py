import pickle, numpy as np, json

# Scalers
with open('v6_smartphone_idr/data/scalers_v6.pkl', 'rb') as f:
    s = pickle.load(f)
print('=== SCALERS ===')
print('X data_min (first 20):', s['X'].data_min_[:20])
print('X data_max (first 20):', s['X'].data_max_[:20])
print('X n_features_in:', s['X'].n_features_in_)
for k in ['y_dv', 'y_w', 'y_ba', 'y_bw']:
    print(f'{k} min={s[k].data_min_}, max={s[k].data_max_}')
print('turn_stats:', s['turn_stats'])

# Test scenarios
with open('v6_smartphone_idr/data/test_scenarios_v6.pkl', 'rb') as f:
    ts = pickle.load(f)
print('=== TEST SCENARIOS ===')
for k, v in ts.items():
    print(k, '->', [j['name'] for j in v], 'samples:', sum(len(j['v_true']) for j in v))
    if v:
        j = v[0]
        print('   keys:', list(j.keys()))
        print('   a_fwd mean/std:', np.mean(j['a_fwd']), np.std(j['a_fwd']))
        print('   w_yaw mean/std:', np.mean(j['w_yaw']), np.std(j['w_yaw']))
        print('   v_true mean/std:', np.mean(j['v_true']), np.std(j['v_true']))
        print('   len:', len(j['v_true']))
