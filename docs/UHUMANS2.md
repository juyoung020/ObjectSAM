# ObjectSAM on uHumans2: a dataset outside the training distribution

ObjectSAM was trained on COCO, LVIS, ADE20K and BEHAVIOR-1K renders. To check that it generalizes, it was
dropped into a mapping frontend and run on **uHumans2**, a dataset it has never seen:
a different simulator (Unity/TESSE, from the [Kimera](https://github.com/MIT-SPARK/Kimera) project),
different scenes, a different camera.

Short answer: compared with FastSAM-s (the model the frontend used before), ObjectSAM

- covers about **half as much wall/floor/ceiling** (0.673 → 0.357 of structure pixels);
- covers **more of the objects** (0.795 → 0.833 of object pixels), and more objects with a single mask (73 % → 81 %);
- runs **4.6× faster** (5.13 → 1.11 ms per frame for the network);
- but gives **more masks per frame** at the default threshold, so mask precision and tracking scores drop (see [Costs](#costs)).

## Setup

| | |
|---|---|
| data | uHumans2, 6 sequences: `apartment_s1_{00h,01h,02h}`, `office_s1_{00h,06h,12h}` |
| frames | every 10th frame, 3,196 frames, same frames for both models |
| image | 720×480 RGB, centre-cropped to 640×480 (the frontend's input) |
| frontend | a FastSAM-based object mapping frontend: letterbox, score threshold, minimum mask area (about 40×40 px), duplicate removal. Only the segmentation model was swapped |
| ObjectSAM | v1.0 (`ObjectSAM-416.onnx`), default score threshold 0.25 |
| GPU | RTX 3060, TensorRT FP16 |
| ground truth | uHumans2 semantic images, colours mapped to classes with the Kimera TESSE tables (99.8–100 % of pixels mapped) |

- **Structure** = wall, floor, ceiling. A mask is a *structure mask* when ≥ 50 % of its pixels are structure.
- **Objects** = every other ground-truth instance (13,852 instances over the 3,196 frames). Carpets/rugs count as neither.
- The frontend expects FastSAM's 1024×1024 input and output layout. ObjectSAM was fitted in with an ONNX wrapper
  (resize 1024 → 416, rescale boxes, map the 0.25 score threshold onto the frontend's 0.4). This double resize
  changes about 9 % of the masks compared with running ObjectSAM directly.

## Coverage: structure down, objects up

| | FastSAM-s | ObjectSAM |
|---|---|---|
| structure pixels covered by a mask (lower is better) | 0.673 | **0.357** |
| object pixels covered by a mask (higher is better) | 0.795 | **0.833** |
| object pixels covered, structure masks left out | 0.767 | **0.806** |
| per object, mean share covered by one mask | 0.668 | **0.734** |
| objects ≥ 50 % covered by one mask | 0.733 | **0.814** |
| objects ≥ 80 % covered by one mask | 0.511 | **0.556** |

Objects ≥ 50 % covered by one mask, by size and scene:

| | FastSAM-s | ObjectSAM |
|---|---|---|
| small (< 1/48 of the image) | 0.699 | **0.826** |
| medium | 0.763 | **0.820** |
| large (≥ 1/12 of the image) | **0.816** | 0.714 |
| apartment | 0.714 | **0.874** |
| office | 0.744 | **0.781** |

Sizes are measured on a 256×192 grid: small < 1,024 cells, large ≥ 4,096 cells.

## Recall at mask IoU ≥ 0.5

| | FastSAM-s | ObjectSAM |
|---|---|---|
| all objects | 0.662 | **0.694** |
| small | 0.619 | **0.672** |
| medium | 0.699 | **0.729** |
| large | **0.774** | 0.685 |
| mean IoU of matched masks | **0.804** | 0.780 |

- Plants (0.23 → 0.85) and books (0.36 → 0.62) gained most.
- Large, flat objects seen up close are the weak spot: curtains, door and cabinet panels filling much of the view.

## Wall, floor and ceiling masks

| | FastSAM-s | ObjectSAM |
|---|---|---|
| masks per frame | 26.4 | 38.2 |
| structure masks per frame | 14.2 | 14.1 |
| share of masks on structure | 0.536 | **0.369** |
| large structure masks (≥ 4,096 cells), apartment | 658 | **62** |
| large structure masks (≥ 4,096 cells), office | 6,710 | **4,199** |

- The masks that cover whole walls, floors and ceilings mostly disappear. What remains on structure is small pieces.
- In the office scenes, uHumans2 paints glass partitions, glass doors and windows with the **wall** colour.
  ObjectSAM keeps doors and windows as objects on purpose, so some of its office "structure masks" are such doors and partitions.

## Speed

| RTX 3060, TensorRT FP16 | FastSAM-s | ObjectSAM |
|---|---|---|
| network | 5.13 ms | **1.11 ms** |
| segmentation step of the frontend | 5.85 ms | **1.82 ms** |
| engine file | 27.5 MB | **9.8 MB** |
| engine activation memory | 51 MB | **5 MB** |

## Costs

At the default threshold (0.25) ObjectSAM gives about 45 % more masks than FastSAM-s on uHumans2.
Many extra masks fall on things uHumans2 does not label separately (single books, stair steps, shelf compartments),
so they count as false positives:

| | FastSAM-s | ObjectSAM |
|---|---|---|
| mask precision at IoU ≥ 0.5 | 0.368 | 0.211 |
| panoptic quality (PQ) | 0.366 | 0.241 |
| masks merging two objects | 808 | 1,436 |
| objects split into pieces | 1,163 | 1,028 |

Downstream, the frontend's 3D object recall rose (0.824 → 0.880) and its geometry stayed the same (F-score at 20 cm 0.881 → 0.876),
but with more observations to associate, its tracking scores fell (HOTA 0.408 → 0.300). The tracker was tuned for FastSAM-s.

If your downstream step is sensitive to the number of masks, raise the threshold for your camera with
`tools/sweep_t.py` and bake it in with `tools/calibrate.py` (see [TRAINING.md](TRAINING.md)).

## Limits of this check

- One dataset, two scene types, simulated images.
- The office ground truth uses one colour for floor and ceiling, and the wall colour includes doors and windows.
- The model went through a FastSAM-shaped wrapper and the frontend's own filtering, not ObjectSAM's reference decoder.
- No threshold was tuned for uHumans2: both models ran at their defaults.
