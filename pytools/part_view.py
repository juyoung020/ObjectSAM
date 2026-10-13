"""부품 후보(teacher_label --parts) 살펴보기: 사진마다 부모(통째 마스크) 테두리 + 부품 채움 그림과 개수.

  .venv/bin/python pytools/part_view.py PARTS [--out analysis/part_teacher] [--per 6]
부품끼리 IoU ≥ 0.5 면 점수 높은 것 하나만(label_v1parts.py 와 같은 규칙).
"""
import argparse
import json
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ftl  # noqa: E402

COLORS = [(255, 64, 64), (64, 200, 64), (64, 128, 255), (255, 200, 0), (255, 0, 255), (0, 220, 220), (255, 128, 0),
          (160, 64, 255), (128, 255, 128), (255, 128, 192)]


def dedup(masks, scores, thr=0.5):
    keep = []
    for i in np.argsort(-np.asarray(scores)):
        if all((masks[i] & masks[j]).sum() / max((masks[i] | masks[j]).sum(), 1) < thr for j in keep):
            keep.append(int(i))
    return sorted(keep)


def tile(img, idm, parts, masks, keep, W=480):
    H0, W0 = idm.shape
    vis = (img.astype(np.float32) * 0.55).astype(np.uint8)
    out = vis.copy()
    for c, i in enumerate(keep):
        col = COLORS[c % len(COLORS)]
        m = masks[i]
        out[m] = (0.45 * np.array(col) + 0.55 * img[m]).astype(np.uint8)
        par = (idm == parts[i]['parent']).astype(np.uint8)
        cs, _ = cv2.findContours(par, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(out, cs, -1, (255, 255, 255), 2)
        cs, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(out, cs, -1, col, 1)
    s = W / W0
    out = cv2.resize(out, (W, int(H0 * s)), interpolation=cv2.INTER_AREA)
    for c, i in enumerate(keep):
        ys, xs = np.nonzero(masks[i])
        cv2.putText(out, f"{c}:{parts[i]['score']:.2f}", (int(xs.mean() * s), int(ys.mean() * s)), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                    (255, 255, 255), 1, cv2.LINE_AA)
    side = cv2.resize(img, (W, int(H0 * s)), interpolation=cv2.INTER_AREA)
    return np.concatenate([side, out], 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('parts')
    ap.add_argument('--ftl', default=None, help='부모 id 지도를 읽을 FTL1(기본: PARTS 와 같이 만든 .ftl)')
    ap.add_argument('--out', default='analysis/part_teacher')
    ap.add_argument('--per', type=int, default=5)
    a = ap.parse_args()
    ftlp = a.ftl or os.path.splitext(a.parts)[0] + '.ftl'
    recs = {r['id']: r for r in ftl.read(ftlp)}
    rows, page, stats = [], 0, []
    for pr in ftl.read_parts(a.parts):
        r = recs[pr['id']]
        H, W = pr['H'], pr['W']
        idm = ftl.idmap(r)
        masks = [ftl.unrle(p['rle'], H * W).reshape(H, W) for p in pr['parts']]
        keep = dedup(masks, [p['score'] for p in pr['parts']])
        stats.append({'id': pr['id'], 'raw': len(masks), 'dedup': len(keep),
                      'parts': [{k: (round(v, 3) if isinstance(v, float) else v) for k, v in pr['parts'][i].items() if k != 'rle'}
                                | {'area_frac_parent': round(pr['parts'][i]['full'] / max(int(r['areas'][pr['parts'][i]['parent']]), 1), 3)}
                                for i in keep]})
        img = cv2.imread(f"data/coco/train2017/{pr['id']:012d}.jpg")[..., ::-1]
        t = tile(img, idm, pr['parts'], masks, keep)
        cv2.putText(t, f"{pr['id']}  parts {len(keep)} (raw {len(masks)})", (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 1,
                    cv2.LINE_AA)
        rows.append(t)
        if len(rows) == a.per:
            write(rows, a.out, page)
            rows, page = [], page + 1
    if rows:
        write(rows, a.out, page)
    json.dump(stats, open(f'{a.out}/proto_stats.json', 'w'), indent=1)
    raw = np.array([s['raw'] for s in stats])
    dd = np.array([s['dedup'] for s in stats])
    print(f'{len(stats)} images: raw {raw.sum()} ({raw.mean():.2f}/img), dedup {dd.sum()} ({dd.mean():.2f}/img), '
          f'images with ≥1 part {(dd > 0).mean() * 100:.0f}%, max {dd.max()}')


def write(rows, out, page):
    W = max(r.shape[1] for r in rows)
    rows = [np.pad(r, ((0, 4), (0, W - r.shape[1]), (0, 0))) for r in rows]
    cv2.imwrite(f'{out}/grid_{page:02d}.jpg', np.concatenate(rows, 0)[..., ::-1], [cv2.IMWRITE_JPEG_QUALITY, 85])


if __name__ == '__main__':
    main()
