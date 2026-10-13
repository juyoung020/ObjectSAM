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


def read_parts(path):
    """부품 후보(teacher_label --parts, 'FTP2') 읽기. 형식은 teacher/label.cpp 머리말.

    사진마다 {'id', 'H', 'W', 'parts': [{'parent', 'rank', 'score', 'full', 'vis', 'inside', 'rle'}]};
    rle 은 coco_build 형식(0 부터 교대 길이, LEB128) — unrle(rle, H*W) 로 펼친다.
    """
    data = open(path, 'rb').read()
    p = 0
    while p + 16 <= len(data):
        assert data[p:p + 4] == b'FTP2', p
        iid, H, W, n, _ = struct.unpack_from('<IHHHH', data, p + 4)
        q = p + 16
        parts = []
        for _ in range(n):
            par, rk, sc, full, vis, ins, nb = struct.unpack_from('<HHfIIfI', data, q)
            q += 24
            parts.append({'parent': par, 'rank': rk, 'score': sc, 'full': full, 'vis': vis, 'inside': ins,
                          'rle': data[q:q + nb]})
            q += nb
        yield {'id': iid, 'H': H, 'W': W, 'parts': parts}
        p = q


def unrle(b, n):
    """coco_build RLE(0 부터 교대 길이) → [n] bool."""
    vals, v, s = [], 0, 0
    for c in b:
        v |= (c & 0x7F) << s
        s += 7
        if not c & 0x80:
            vals.append(v)
            v = s = 0
    runs = np.asarray(vals, np.int64)
    vv = np.zeros(len(runs), bool)
    vv[1::2] = True
    out = np.repeat(vv, runs)
    assert out.size == n, (out.size, n)
    return out
