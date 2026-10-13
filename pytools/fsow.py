"""FSOW: C++ 학습기와 주고받는 단순 텐서 묶음 형식.

파일 = b'FSOW0001' + u32 개수 + 텐서마다 [u16 이름 길이, 이름(utf-8), u8 차원 수, i64 크기들, u8 형(0 f32, 1 i32, 2 u8), 원시 데이터].
리틀 엔디언. 가중치는 PyTorch 모양 그대로(OIHW) 저장하고, 배치는 C++ 쪽이 바꾼다.
"""
import struct

import numpy as np

MAGIC = b'FSOW0001'
_DT = {np.dtype(np.float32): 0, np.dtype(np.int32): 1, np.dtype(np.uint8): 2}
_TD = {v: k for k, v in _DT.items()}


def save(path, tensors):
    with open(path, 'wb') as f:
        f.write(MAGIC)
        f.write(struct.pack('<I', len(tensors)))
        for name, a in tensors.items():
            a = np.ascontiguousarray(a)
            if a.dtype not in _DT:
                a = a.astype(np.float32)
            nb = name.encode()
            f.write(struct.pack('<H', len(nb)) + nb)
            f.write(struct.pack('<B', a.ndim) + struct.pack(f'<{a.ndim}q', *a.shape))
            f.write(struct.pack('<B', _DT[a.dtype]))
            f.write(a.tobytes())


def load(path):
    out = {}
    with open(path, 'rb') as f:
        assert f.read(8) == MAGIC, path
        (n,) = struct.unpack('<I', f.read(4))
        for _ in range(n):
            (ln,) = struct.unpack('<H', f.read(2))
            name = f.read(ln).decode()
            (nd,) = struct.unpack('<B', f.read(1))
            shape = struct.unpack(f'<{nd}q', f.read(8 * nd)) if nd else ()
            (dt,) = struct.unpack('<B', f.read(1))
            dt = _TD[dt]
            cnt = int(np.prod(shape)) if nd else 1
            out[name] = np.frombuffer(f.read(cnt * dt.itemsize), dt).reshape(shape).copy()
    return out
