#!/usr/bin/env bash
# 선생(CropFormer Hornet-L) PyTorch 환경: 정합성 기준값(pytools/teacher_dump.py)과 가중치 변환에만 쓴다.
# 라벨 만들기는 C++(build/teacher_label)이라 이 환경이 필요 없다.
#   scripts/setup_teacher.sh   (~/ObjectSAM 에서)
set -euo pipefail
cd "$(dirname "$0")/.."
ROOT=$(pwd)
TP=third_party
mkdir -p $TP weights/teacher
[ -d $TP/Entity ] || git clone -q https://github.com/qqlu/Entity.git $TP/Entity
git -C $TP/Entity checkout -q 6e7e13ac91ef508088e1b848167c01f19b00b512
[ -d $TP/detectron2 ] || git clone -q https://github.com/facebookresearch/detectron2.git $TP/detectron2
git -C $TP/detectron2 checkout -q 1e3e13bbf607b54f62205c4c33922521822fb298

uv venv -q -p 3.11 .venv-teacher
export UV_LINK_MODE=hardlink
uv pip install -q -p .venv-teacher/bin/python torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu128
uv pip install -q -p .venv-teacher/bin/python "numpy<2.5" opencv-python-headless==5.0.0.93 scipy shapely "timm==0.9.16" h5py \
  scikit-image cython pycocotools fvcore iopath ninja setuptools wheel gdown
export CUDA_HOME=/usr/local/cuda-12.8 PATH=/usr/local/cuda-12.8/bin:$PATH TORCH_CUDA_ARCH_LIST="12.0" MAX_JOBS=24
uv pip install -q -p .venv-teacher/bin/python --no-build-isolation -e $TP/detectron2

# CropFormer 를 detectron2 프로젝트로 복사하고 고친 점을 적용(scripts/cropformer.patch):
#  - ms_deform_attn_cuda.cu: 최신 PyTorch 에 없는 value.type() → value.scalar_type()
#  - register_entityv2_semseg_150.py: 학습 데이터 등록에만 쓰는 mmcv 를 지연 import
#  - cropformer_model.py: 마스크를 화면마다 키운다(한꺼번에 키우면 약 8 GB) — 결과는 같다
rm -rf $TP/detectron2/projects/CropFormer
cp -r $TP/Entity/Entityv2/CropFormer $TP/detectron2/projects/CropFormer
(cd $TP/detectron2/projects/CropFormer && patch -s -p1 < $ROOT/scripts/cropformer.patch)
(cd $TP/detectron2/projects/CropFormer/mask2former/modeling/pixel_decoder/ops && $ROOT/.venv-teacher/bin/python setup.py build install > /dev/null)

# 가중치(Hugging Face 는 승인제라 저자가 올린 Google Drive 사본에서; 크기 888,996,425 바이트)
W=weights/teacher/CropFormer_hornet_3x_03823a.pth
[ -f $W ] || .venv-teacher/bin/gdown -q 12LO1OsOGu6L7EfsxVflTANhoV7JvO6m9 -O $W
.venv-teacher/bin/python pytools/teacher_dump.py weights
