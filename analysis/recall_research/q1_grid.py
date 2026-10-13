import pickle, random, sys
sys.path.insert(0, '.')
from grid import render
R = pickle.load(open('label_stats.pkl','rb'))
random.seed(0)
for kind in ('ceil', 'wall'):
    rem = [(r['id'], x['k'], f"{x['status']} {x['frac']*100:.2f}% s{x['sig']:.2f}") for r in R for x in r['removed'] if x['kind']==kind and x['frac']<0.02]
    random.shuffle(rem)
    render(rem[:48], f'q1_removed_{kind}.jpg')
    kept = [(r['id'], x['k'], f"kept {x['frac']*100:.2f}% s{x['sig']:.2f}") for r in R for x in r['kept_struct'] if x['kind']==kind and x['frac']<0.02 and x['frac']>=0.001]
    random.shuffle(kept)
    render(kept[:32], f'q1_kept_{kind}.jpg')
