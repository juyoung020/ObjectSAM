"""선생 마스크 표본 격자 그림: items = [(image_id, k, caption)], 마스크 윤곽(초록)과 반투명 칠, 상자 주변 잘라 보기."""
import sys, cv2, numpy as np
sys.path.insert(0, '/home/juyoung/ObjectSAM/pytools')
import ftl
IMG = '/home/juyoung/ObjectSAM/data/coco/train2017'

def render(items, out, cell=220, cols=8):
    need = {}
    for it in items:
        need.setdefault(it[0], []).append(it)
    tiles = {}
    for r in ftl.read('/home/juyoung/ObjectSAM/data/teacher/train2017.ftl'):
        if r['id'] not in need:
            continue
        m = ftl.idmap(r)
        im = cv2.imread(f"{IMG}/{r['id']:012d}.jpg")
        for it in need[r['id']]:
            k, cap = it[1], it[2]
            mk = (m == k)
            ys, xs = np.nonzero(mk)
            if not len(ys):
                continue
            x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
            s = max(x1 - x0, y1 - y0, 40)
            cx, cy = (x0 + x1) // 2, (y0 + y1) // 2
            h = int(s * 1.6) // 2 + 10
            X0, Y0 = max(0, cx - h), max(0, cy - h)
            X1, Y1 = min(im.shape[1], cx + h), min(im.shape[0], cy + h)
            v = im.copy()
            ov = v.copy(); ov[mk] = (0, 255, 0)
            v = cv2.addWeighted(v, 0.65, ov, 0.35, 0)
            cs, _ = cv2.findContours(mk.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            cv2.drawContours(v, cs, -1, (0, 255, 0), 1)
            c = v[Y0:Y1, X0:X1]
            sc = cell / max(c.shape[:2])
            c = cv2.resize(c, (max(1, int(c.shape[1] * sc)), max(1, int(c.shape[0] * sc))))
            t = np.full((cell + 18, cell, 3), 255, np.uint8)
            t[:c.shape[0], :c.shape[1]] = c
            cv2.putText(t, cap[:34], (2, cell + 13), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (0, 0, 0), 1)
            tiles[(it[0], k)] = t
        if len(tiles) >= len(items):
            break
    ts = [tiles[(i[0], i[1])] for i in items if (i[0], i[1]) in tiles]
    rows = []
    for i in range(0, len(ts), cols):
        row = ts[i:i + cols]
        row += [np.full_like(ts[0], 255)] * (cols - len(row))
        rows.append(np.hstack(row))
    cv2.imwrite(out, np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 85])
    print(out, len(ts))
