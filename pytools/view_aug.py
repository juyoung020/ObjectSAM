"""aug_dump 결과 그림: 증강 영상 위에 정답 마스크(색)와 상자.

  .venv/bin/python pytools/view_aug.py data/aug.fsow data/aug.jpg
"""
import sys

import cv2
import numpy as np

import fsow


def main(src, dst):
    t = fsow.load(src)
    img = t['img']
    rows = []
    rng = np.random.default_rng(0)
    for i in range(img.shape[0]):
        im = img[i][..., ::-1].astype(np.float32)
        S = im.shape[0]
        m = t[f'mask{i}']
        bx = t[f'box{i}']
        over = im.copy()
        for j in rng.permutation(len(m)):
            mm = cv2.resize(m[j], (S, S), interpolation=cv2.INTER_NEAREST).astype(bool)
            over[mm] = 0.45 * over[mm] + 0.55 * rng.uniform(40, 255, 3)
        if 'ignore' in t:  # 무시 영역: 자홍
            ig = cv2.resize(t['ignore'][i], (S, S), interpolation=cv2.INTER_NEAREST).astype(bool)
            over[ig] = 0.3 * over[ig] + 0.7 * np.array([255, 0, 255.])
        over = over.astype(np.uint8)
        for b in bx:
            cv2.rectangle(over, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (255, 255, 255), 1)
        cv2.putText(over, f'{len(m)} gt', (8, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
        rows.append(np.concatenate([im.astype(np.uint8), over], 1))
    out = np.concatenate(rows, 0)
    out = cv2.resize(out, (out.shape[1] // 2, out.shape[0] // 2), interpolation=cv2.INTER_AREA)
    cv2.imwrite(dst, out, [cv2.IMWRITE_JPEG_QUALITY, 85])
    print(dst)


if __name__ == '__main__':
    main(*sys.argv[1:3])
