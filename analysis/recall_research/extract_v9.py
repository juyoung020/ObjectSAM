"""fail_explore/extract.py 를 그대로 import 해서 v9@0.1·v1@0.1 기록만 만든다(원본 수정 없음)."""
import os, sys
from multiprocessing import Pool
sys.path.insert(0, '/home/juyoung/meridian_ws/analysis/fail_explore')
import extract as E
E.RUNS = {'v901': 'runs/full_exp_v9_01', 'v101': 'runs/frontend_eval_osam010'}
E.OUT = '/home/juyoung/ObjectSAM/analysis/recall_research/recs'
if __name__ == '__main__':
    jobs = [(r, s) for r in E.RUNS for s in E.SEQS]
    with Pool(8) as p:
        for x in p.imap_unordered(E.job, jobs):
            print(*x, flush=True)
