"""선생 라벨(FTL1) 읽기. 형식은 teacher/label.cpp 머리말.

  for rec in ftl.read('data/teacher/train2017.ftl'): rec['id'], rec['scores'], rec['areas'], ftl.idmap(rec)
"""
import struct

import numpy as np


def _leb(b, p):
    v = s = 0
    while True:
        c = b[p]
        p += 1
        v |= (c & 0x7F) << s
        s += 7
        if not c & 0x80:
            return v, p


def read(path):
    data = open(path, 'rb').read()
    p = 0
    while p + 16 <= len(data):
        assert data[p:p + 4] == b'FTL1', p
        iid, H, W, n, _ = struct.unpack_from('<IHHHH', data, p + 4)
        q = p + 16
        sa = np.frombuffer(data, dtype=np.dtype([('s', '<f4'), ('a', '<u4')]), count=n, offset=q)
        q += 8 * n
        (nb,) = struct.unpack_from('<I', data, q)
        q += 4
        yield {'id': iid, 'H': H, 'W': W, 'scores': sa['s'].copy(), 'areas': sa['a'].copy(), 'rle': data[q:q + nb]}
        p = q + nb


def idmap(rec):
    """[H, W] int32: -1 = 마스크 없음, k = k 번 마스크(점수 순)."""
    b, p = rec['rle'], 0
    vals, runs = [], []
    while p < len(b):
        v, p = _leb(b, p)
        r, p = _leb(b, p)
        vals.append(v)
        runs.append(r)
    out = np.repeat(np.asarray(vals, np.int32) - 1, runs)
    return out.reshape(rec['H'], rec['W'])
