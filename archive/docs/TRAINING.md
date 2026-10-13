# ObjectSAM: training and evaluation details

This page explains how ObjectSAM was made and how to retrain it. To *use* the model, see the [README](../README.md).

## Model

- YOLO26n-seg: 2.7 M parameters, about 3.8 GFLOPs at 640. It starts from Ultralytics `yolo26n-seg.pt` with a fresh one-class head (`object`).
- Exported with `end2end=False`, so it has the same raw outputs as FastSAM-s and the same post-processing applies.
- The ONNX (opset 13) uses only Conv, Mul, Sigmoid, Add, Concat, Reshape, Split, Transpose, MatMul, Resize, MaxPool, Softmax, Slice, Sub, ConvTranspose and Div. It builds with TensorRT 8.2 (Jetson Nano, JetPack 4.6) and later.

## Labels: self-distillation plus whole-object ground truth

`tools/build_data.py` builds the target set for every training image from two sources.

1. **All ground-truth instance masks** where a dataset provides them: COCO things, LVIS, ADE20K instances, and simulator instances. Doors, windows and stairs are included.
   - Instances are never merged.
   - Masks smaller than about half of the runtime minimum mask area are dropped (area < 0.13 % of the image).
2. **FastSAM-s masks** (the teacher, conf 0.25), except:
   - masks ≥ 50 % on wall, ceiling or floor (structure);
   - masks ≥ 50 % inside one GT instance or ≥ 70 % inside the union of GT instances. These are fragments, and the whole GT mask replaces them;
   - masks covering ≥ 20 % of any GT instance. These are merges of an object with its surroundings.

   Everything else is kept: unlabeled objects and unknown categories. The student therefore keeps what FastSAM found.

## Data

| source | used | role |
|---|---|---|
| COCO 2017 + COCO-panoptic | 12,000 train images (9,000 "indoor", i.e. wall/floor/ceiling ≥ 10 %, + 3,000 others) | things = GT, wall/floor/ceiling stuff = background, door/window/stairs stuff = GT |
| LVIS v1 | instances on the same images (1,203 categories) | GT. 120 non-COCO categories are **held out** (labels removed) to test unseen-category recall |
| ADE20K (SceneParsing 2016 + instance 2017) | 6,000 indoor train images | instances = GT, wall/floor/ceiling = background |
| BEHAVIOR-1K scenes (rendered with OmniGibson) | 36 train scenes, about 6.4 k frames; 10 held-out scenes for evaluation | robot-height RGB + exact instance masks |

**Simulator frames** (`tools/sim_render.py`, `tools/label_sim.py`)
- Camera: 0.18 m high, level, 67.9° horizontal FOV, 640×480. That is the camera of a small mobile robot.
- Poses are sampled in free space from the scene traversability map:
  - 20 % close to walls or furniture;
  - 10 % pitched up toward the ceiling;
  - 10 % raised 0.3–1.2 m.
- Instance masks come from ray-casting the scene's visual meshes (embree), not from the simulator's segmentation output.
- A depth cross-check against the rendered depth rejects stale or mismatched frames. Agreement is 99.3–99.6 %.

## Training (`tools/train.py`)

Ultralytics training at 416, SGD, cosine schedule, mosaic, `overlap_mask=False`, single class. Three stages:

1. **60 epochs** from `yolo26n-seg.pt`, simulator frames ×3.
2. **30 epochs** with simulator frames ×2 and frames showing a door/window/stairs ×2 more.
3. **15 epochs with an under-segmentation penalty** (`--under-w 3`).
   - In the mask BCE, pixels that belong to *another* labelled instance in the same image (and not to the target) are weighted ×3.
   - A chair mask that spills onto the table or a person costs more.
   - Nested targets (clothes inside a person) are not penalised, because their pixels are positive for the outer object.
   - This halved under-segmentation (merged masks per frame: simulator 3.42 → 1.69, COCO 7.33 → 4.76, ADE20K 8.22 → 6.31). The cost is more conservative masks (simulator recall @ IoU 0.5: 0.698 → 0.633).

Two things were tried and did **not** help:
- **Training longer.** The curves were flat after about 20 epochs, which looks like the capacity of a 2.7 M-parameter model.
- **A cleaner teacher.** Masks from a FastSAM-s retrained the same way, used as pseudo-labels instead of the original FastSAM-s masks, gave slightly more false masks and more under-segmentation.

## Score calibration (`tools/calibrate.py`)

- A whole-object model emits one mask per object instead of several fragments, so its scores are lower.
- `tools/sweep_t.py` finds the threshold `t` that keeps recall equal to FastSAM-s; here t = 0.03.
- `logit(0.25) − logit(t)` is added to the class bias, so the released model is used at the usual conf 0.25.
- This is exactly equivalent to thresholding the uncalibrated model at `t`, since ranking, NMS and mask de-duplication are unchanged.

## Evaluation

Both models use the same inference path:
- letterbox 416, conf 0.25;
- class-agnostic NMS 0.7, at most 100 masks;
- mask = prototype logit > 0 inside the box;
- masks under 24 grid cells dropped, duplicate masks (IoU > 0.7) removed.

`tools/ref_detect.py` reproduces this with ONNX Runtime.

**Metrics** (held-out data, every 2nd evaluation frame). Each ground-truth object is matched against all predicted masks of its frame.

| metric | type | definition | better |
|---|---|---|---|
| **Recall** | fraction of GT objects (0–1) | the object counts as found if at least one predicted mask has ≥ 50 % of its pixels on the object. This is the condition under which a mapping pipeline creates a node for it | higher |
| **Recall @ IoU 0.5** | fraction of GT objects (0–1) | the best-matching predicted mask has mask IoU ≥ 0.5 with the object, i.e. the object is found *whole* | higher |
| **Structure false masks / frame** | count per image | predicted masks with ≥ 50 % of their pixels on wall, ceiling or floor | lower |
| **Masks per found object** (over-segmentation) | count | number of predicted masks lying on each found object, averaged | lower (1 = one mask per object) |
| **Merged masks / frame** (under-segmentation) | count per image | predicted masks that cover ≥ 20 % of two GT objects (e.g. chair + table), or ≥ 20 % of one GT object while ≥ 30 % of the mask is wall/ceiling/floor (e.g. picture + wall). Masks of objects nested inside another (clothes on a person) are not counted | lower |

No precision or F1 is reported. The model is class-agnostic and open-world, so many correct masks fall on objects that the datasets do not label, and they would be counted as false positives. False positives are measured on what is known to be wrong instead: masks on wall, ceiling or floor.

| | FastSAM-s | ObjectSAM |
|---|---|---|
| **BEHAVIOR held-out scenes** (10 scenes) | | |
| Recall: all / doors-windows-stairs / unseen categories | 0.793 / 0.866 / 0.668 | 0.803 / 0.864 / 0.695 |
| Structure false masks / frame | 4.44 | **1.55** |
| Masks per found object | 2.77 | **2.51** |
| Merged masks / frame | 1.66 | 1.69 |
| Recall @ IoU 0.5 | **0.676** | 0.633 |
| **COCO val2017** (1,500 images) | | |
| Recall: all / small / large / doors-windows-stairs / held-out LVIS categories | 0.499 / 0.055 / 0.962 / 0.863 / 0.346 | **0.530** / 0.068 / 0.976 / 0.863 / 0.358 |
| Structure false masks / frame | 6.06 | 6.91 |
| Merged masks / frame | 3.33 | 4.76 |
| Recall @ IoU 0.5 | 0.422 | 0.431 |
| **ADE20K val, indoor** (1,021 images) | | |
| Recall: all / doors-windows-stairs | 0.581 / 0.785 | **0.628** / **0.816** |
| Structure false masks / frame | 10.05 | 10.81 |
| Merged masks / frame | 3.75 | 6.31 |
| Recall @ IoU 0.5 | 0.501 | **0.545** |
| ONNX / TensorRT FP16 engine size | 47 MB / 25 MB | **11 MB / 7 MB** |

**Recall gate**
- Recall of ObjectSAM minus recall of FastSAM-s, per bucket (size classes, doors/windows/stairs, unseen categories) and per dataset, with a paired frame bootstrap.
- It passes on all three datasets.
- The weakest buckets are BEHAVIOR doors/windows/stairs at −0.2 recall points (percentage points, ObjectSAM minus FastSAM-s) (95 % CI [−3.0, +2.3]) and BEHAVIOR medium objects at −0.5 [−4.4, +3.3].

## Precision and quantization (`tools/quant_engine.py`)

**FP16** needs no FP32-pinned layers. TensorRT FP16 vs ONNX Runtime FP32 on BEHAVIOR: recall 0.805 vs 0.806; recall @ IoU 0.5 0.629 vs 0.636.

**INT8 post-training quantization**
- Method: explicit Q/DQ with onnxruntime static quantization (Conv only, symmetric, per-channel weights) over 512 calibration images (half simulator, half COCO/ADE), then TensorRT `--int8 --fp16`.
- TensorRT 10.16's implicit (calibrator) INT8 fails to build this graph, which is why explicit Q/DQ is used.
- Entropy calibration, entropy with the head (`model.23`) left in FP16, and the 99.99 percentile all pass the recall gate. MinMax fails it (doors/windows/stairs recall −1.8 points).
- The cost of INT8 is about +0.3 wall/floor false masks per frame on simulator images.
- The released `ObjectSAM-416-int8-qdq.onnx` is entropy with the head in FP16. Quantization-aware training was not needed.

**Speed** (RTX 5070 Ti, 416, network only with CUDA graphs): FP16 0.39 ms, INT8 0.44 ms. INT8 is *not* faster on a desktop GPU for a network this small; it may pay off on Jetson Orin (not measured). Jetson Nano (Maxwell, sm_53) has no fast INT8 path.

**Re-quantizing with your own calibration images**
```bash
python tools/quant_engine.py calib --out calib512.npy --list train_list.txt --n 512
python tools/quant_engine.py qdq ObjectSAM-416.onnx my-int8-qdq.onnx --calib calib512.npy --algo entropy --exclude 'model\.23/'
python tools/quant_engine.py build my-int8-qdq.onnx my-int8.plan --mode int8   # handles TensorRT 8.2 and 10 APIs
```

## Rebuild from scratch

Everything is in `tools/`.
- Set `FASTSAM_DATA` (work directory) and `B1K_ROOT` (a BEHAVIOR-1K checkout with its datasets).
- For the simulator, also set `OG_LOCK` (lock file) and `OG_CONDA_ENV` (OmniGibson environment).

```bash
cd tools
# 0) teacher: FastSAM-s.pt from Ultralytics -> models/FastSAM-s.pt, and its 416 ONNX -> models/FastSAM-s-416.onnx
# 1) simulator frames + exact instance masks (needs OmniGibson / BEHAVIOR-1K and its dataset license)
bash render_all.sh eval; bash render_all.sh train; bash render_tasks.sh
# 2) real images: put COCO panoptic annotations, LVIS v1 json zips, ADEChallengeData2016 and the 2017 instance
#    annotations in $FASTSAM_DATA/raw, then select + download the COCO subset
python select_real.py
# 3) labels (self-distillation from models/FastSAM-s-416.onnx + GT)
for s in coco_train ade_train sim_train sim_val coco_val; do python build_data.py --source $s; done
# 4) train the student in three stages
python train.py --name n1 --init yolo26n-seg.pt --epochs 60 --sim-rep 3 --lr0 0.01
python train.py --name n2 --init $FASTSAM_DATA/runs/n1/weights/last.pt --epochs 30 --sim-rep 2 --dws-rep 2 --lr0 0.003
python train.py --name n3 --init $FASTSAM_DATA/runs/n2/weights/last.pt --list $FASTSAM_DATA/yolo/train_n2.txt --epochs 15 --lr0 0.002 --under-w 3
# 5) pick the threshold, calibrate, export
OUT_PT=out OUT_PLAN=out bash export.sh $FASTSAM_DATA/runs/n3/weights/last.pt cand    # ONNX (+ TensorRT if available)
python sweep_t.py out/cand.onnx --ts 0.25 0.1 0.07 0.05 0.04 0.03 --out $FASTSAM_DATA/eval/sweep.json
python calibrate.py $FASTSAM_DATA/runs/n3/weights/last.pt ObjectSAM-416.pt --t <best_t>
OUT_PT=out OUT_PLAN=out bash export.sh ObjectSAM-416.pt ObjectSAM-416
```

The evaluation harness (`eval_det.py`) runs either the ONNX reference detector (`.onnx`) or a TensorRT engine through the `ovdet` C library (`.plan`, optional, via `OVDET_LIB`).
