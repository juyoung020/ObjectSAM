import pickle, random, sys
sys.path.insert(0,'.')
from grid import render
S=pickle.load(open('q2_sel.pkl','rb')); random.seed(1)
for key in ('cand','enc','struct'):
    L=S[key]; random.shuffle(L)
    render([(i,x['k'],f"s{x['score']:.2f} {x['frac']*100:.2f}% ig{x['in_ignore']:.1f}") for i,x in L[:40]], f'q2_low_{key}.jpg')
