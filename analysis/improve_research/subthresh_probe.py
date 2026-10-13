"""CPU ONNX 탐침: v5(@0.1)가 놓쳤고 FastSAM-s(@0.1)는 맞힌 물체 GT 에 대해, v5 가 문턱 아래(또는 NMS·dedup 로 지워진) 후보를
내고 있었는지 본다. frontend 전처리(가운데 640 crop → 1024x768, 위 128 줄 회색)·마스크 조립(proto 32:224 행, 상자 밖 소거)을 그대로 따른다.
결과: GT 마다 'IoU>0.5 후보의 최대 점수' 분포. 점수가 0.03~0.1 에 몰리면 = 보이지만 확신이 낮음(음성 라벨 잡음·보정 문제),
후보가 아예 없으면 = 모양을 못 배움(양성 라벨이 필요).
  python -I subthresh_probe.py <seq> <max_kf> <model.onnx> <tag>"""
import sys, glob, json, pickle, collections
import numpy as np, cv2, onnxruntime as ort
WS = '/home/juyoung/meridian_ws'
GH, GW, CELL = 192, 256, 2.5   # score_2d.py 와 같은 격자
_VI = (np.arange(GH) * CELL + CELL / 2).astype(int); _UI = (np.arange(GW) * CELL + CELL / 2).astype(int)
def gt_grid(p):
    return cv2.imread(str(p), cv2.IMREAD_UNCHANGED)[_VI][:, _UI].astype(np.int32)
seq, MAXKF, onnx, tag = sys.argv[1], int(sys.argv[2]), sys.argv[3], sys.argv[4]
ROLE = json.load(open(f'{WS}/tools/kimera/gt_role.json'))[seq]
v5 = pickle.load(open(f'{WS}/analysis/fail_explore/recs/v501__{seq}.pkl', 'rb'))['gt']
fs = pickle.load(open(f'{WS}/analysis/fail_explore/recs/fs01__{seq}.pkl', 'rb'))['gt']
fs_tp = {(g['frame'], g['ep']) for g in fs if g['status'] == 'tp'}
tgt = collections.defaultdict(list)
for g in v5:
    if g['status'] != 'tp' and (g['frame'], g['ep']) in fs_tp:
        tgt[g['frame']].append(g)
frames = sorted(tgt)[::max(1, len(tgt) // MAXKF)][:MAXKF]
so = ort.SessionOptions(); so.intra_op_num_threads = 6
sess = ort.InferenceSession(onnx, so, providers=['CPUExecutionProvider'])
rgbs = sorted(glob.glob(f'{WS}/datasets/unpacked/uHumans2_{seq}/left_cam_rgb/*'))
Cg = np.arange(GW)[None, None]; Rg = np.arange(GH)[None, :, None]
rows = []
for fr in frames:
    img = cv2.imread(rgbs[fr])[:, 40:680, ::-1].astype(np.float32) / 255.
    can = np.full((1024, 1024, 3), 114 / 255., np.float32)
    can[128:896] = cv2.resize(img, (1024, 768), interpolation=cv2.INTER_LINEAR)
    o0, o1 = sess.run(None, {'images': can.transpose(2, 0, 1)[None]})
    p0 = o0[0]; sc = p0[4]
    sel = np.where(sc > 0.003)[0]; sel = sel[np.argsort(-sc[sel])][:600]
    cx, cy, w, h = p0[0, sel], p0[1, sel], p0[2, sel], p0[3, sel]
    bb = np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], 1) / 4.0; bb[:, [1, 3]] -= 32
    m = (p0[5:, sel].T @ o1[0][:, 32:224, :].reshape(32, -1)).reshape(-1, GH, GW) > 0
    m &= (Cg >= bb[:, 0, None, None]) & (Cg < bb[:, 2, None, None]) & (Rg >= bb[:, 1, None, None]) & (Rg < bb[:, 3, None, None])
    area = m.reshape(len(sel), -1).sum(1)
    Graw = gt_grid(f'{WS}/runs/full_exp_parts_01/gt_labels_2d/uHumans2_{seq}/{fr:06d}.png')
    void = np.zeros_like(Graw, bool)
    for e in np.unique(Graw):
        if e == 0 or ROLE.get(str(e - 1), 'ignore') != 'obj': void |= Graw == e
    for g in tgt[fr]:
        gm = Graw == g['ep'] + 1
        inter = (m & gm).reshape(len(sel), -1).sum(1); ign = (m & void).reshape(len(sel), -1).sum(1)
        iou = inter / np.maximum(area + gm.sum() - inter - ign, 1)
        ok = (iou > 0.5) & (area >= 256)
        best = float(sc[sel][ok].max()) if ok.any() else 0.0
        best_any = float(sc[sel][iou > 0.5].max()) if (iou > 0.5).any() else 0.0
        rows.append(dict(frame=fr, ep=g['ep'], status=g['status'], cls=g.get('cls'), fam=g.get('fam'), area=g['area'],
                         best_conf=best, best_conf_anyarea=best_any, max_iou=float(iou.max())))
pickle.dump(rows, open(f'/home/juyoung/ObjectSAM/analysis/improve_research/probe_{tag}_{seq}.pkl', 'wb'))
b = np.array([r['best_conf'] for r in rows])
print(f'{tag} {seq}: kf {len(frames)}  target GTs {len(rows)} (v5@0.1 miss, FS@0.1 hit)')
for lo, hi, nm in [(0.1, 1.1, '>=0.1 (있었지만 NMS·dedup·면적 등으로 빠짐)'), (0.03, 0.1, '0.03-0.1'), (0.01, 0.03, '0.01-0.03'), (0.003, 0.01, '0.003-0.01'), (-1, 0.003, 'IoU>0.5 후보 없음')]:
    print(f'  {nm}: {((b >= lo) & (b < hi)).mean() * 100:.1f}%')
print('  max IoU of any candidate (median):', np.median([r['max_iou'] for r in rows]).round(3))
