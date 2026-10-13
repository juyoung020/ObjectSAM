"""학습 라벨(v5 판정 파일)에서 '배경으로 학습되는 물체 같은 화소' 세기(CPU 만).
화소 나누기: 양성(남은 선생 마스크 ∪ 사람 정답·부품) / 구조물(WCF) / 무시(crowd) / 나머지 = 고아(음성으로 학습됨).
고아 화소가 어느 선생 마스크에 속하는지(점수 구간, 판정에서 버려졌는지)로 나누고,
'고아 비율 ≥ 70 %, ≥ 100 px 인 선생 마스크' = 배경으로 학습되는 물체 후보를 센다.
  python -I orphan_check.py --plan data/teacher/train2017_human_os_parts.plan --every 200 --out DIR
"""
import argparse, io, json, os, struct, sys, zipfile, collections
import numpy as np, cv2
sys.path.insert(0, '/home/juyoung/ObjectSAM/pytools')
import ftl
from label_plan import WCF

def leb_runs(b):
    vals, p, v, s = [], 0, 0, 0
    out = []
    while p < len(b):
        c = b[p]; p += 1
        v |= (c & 0x7F) << s; s += 7
        if not c & 0x80:
            out.append(v); v = s = 0
    return out

def rle_mask(b, n):
    m = np.zeros(n, bool); pos = 0; one = False
    for r in leb_runs(b):
        if one and r: m[pos:pos + r] = True
        pos += r; one = not one
    return m

def read_plan(path):
    data = open(path, 'rb').read(); p = 0
    while p < len(data):
        assert data[p:p+4] == b'FTP1'
        iid, nd, nc, ne, _ = struct.unpack_from('<IHHHH', data, p + 4); q = p + 16
        drop = np.frombuffer(data, '<u2', nd, q); q += 2 * nd + 2 * nc
        ex = []
        for e in range(ne + 1):
            (ln,) = struct.unpack_from('<I', data, q); q += 4
            ex.append(data[q:q+ln]); q += ln
        yield iid, drop.copy(), ex[:-1], ex[-1]
        p = q

