#!/usr/bin/env bash
# TRELLIS (image to 3D) install for an RTX 50-series card, no sudo.
# Everything lands in ~/trellis, or in $TRELLIS_HOME if that is set. Safe to rerun.
# It differs from Microsoft's setup.sh, which installs a PyTorch too old for these cards.
set -eo pipefail
T=${TRELLIS_HOME:-$HOME/trellis}
# The TRELLIS commit this was tested against.
TRELLIS_COMMIT=442aa1e1afb9014e80681d3bf604e8d728a86ee7
mkdir -p "$T/bin" "$T/ext"
cd "$T"
stage() { echo; echo "=== STAGE: $* === $(date +%T)"; }

stage micromamba
[ -x bin/micromamba ] || curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -xj bin/micromamba
export MAMBA_ROOT_PREFIX=$T/mamba

stage env
if [ ! -x env/bin/python ]; then
  bin/micromamba create -y -p "$T/env" -c nvidia/label/cuda-12.8.1 -c conda-forge python=3.11 cuda-toolkit=12.8.1 pip
fi
export PATH=$T/env/bin:$PATH
export CUDA_HOME=$T/env CUDA_PATH=$T/env
export LD_LIBRARY_PATH=$T/env/lib:${LD_LIBRARY_PATH:-}
nvcc --version | tail -2

stage torch
pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128
python -c "import torch;print(torch.__version__,torch.cuda.is_available(),torch.cuda.get_device_name(0),torch.cuda.get_arch_list())"

stage basic
pip install pillow imageio imageio-ffmpeg tqdm easydict opencv-python-headless scipy ninja rembg onnxruntime trimesh open3d xatlas pyvista pymeshfix igraph transformers
pip install git+https://github.com/EasternJournalist/utils3d.git@9a4eb15e4021b67b12c460c7057d642626897ec8

stage xformers
pip install xformers==0.0.31 --index-url https://download.pytorch.org/whl/cu128

stage spconv
pip install spconv-cu120

stage kaolin
pip install kaolin==0.18.0 -f https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.7.1_cu128.html

stage nvdiffrast
[ -d ext/nvdiffrast ] || git clone --depth 1 https://github.com/NVlabs/nvdiffrast.git ext/nvdiffrast
pip install setuptools wheel && pip install --no-build-isolation ext/nvdiffrast

stage mipgaussian
[ -d ext/mip-splatting ] || git clone --depth 1 https://github.com/autonomousvision/mip-splatting.git ext/mip-splatting
R=ext/mip-splatting/submodules/diff-gaussian-rasterization
grep -q '<cstdint>' $R/cuda_rasterizer/rasterizer_impl.h || sed -i '1i #include <cstdint>' $R/cuda_rasterizer/rasterizer_impl.h
pip install --no-build-isolation $R

stage repo
if [ ! -d TRELLIS ]; then
  git clone https://github.com/microsoft/TRELLIS.git
  git -C TRELLIS checkout "$TRELLIS_COMMIT"
  git -C TRELLIS submodule update --init --recursive
fi

stage verify
python - <<'EOF'
import torch
print('torch', torch.__version__, torch.cuda.get_device_name(0))
import xformers, spconv, kaolin, nvdiffrast, diff_gaussian_rasterization, utils3d
print('xformers', xformers.__version__, 'spconv', spconv.__version__, 'kaolin', kaolin.__version__)
import spconv.pytorch as sp
from spconv.core import ConvAlgo
x = sp.SparseConvTensor(torch.randn(4,8).cuda(), torch.tensor([[0,0,0,0],[0,1,1,1],[0,2,2,2],[0,3,3,3]],dtype=torch.int32).cuda(), [8,8,8], 1)
# spconv's default algorithm dies with a floating point exception on RTX 50-series; native works.
print('spconv gpu', sp.SubMConv3d(8,8,3,algo=ConvAlgo.Native).cuda()(x).features.shape)
EOF
echo "=== INSTALL DONE === $(date +%T)"
