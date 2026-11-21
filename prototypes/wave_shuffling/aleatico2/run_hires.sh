#!/usr/bin/env bash
# Wave-shuffling high-resolution runs on the GPU host, one job at a time, detached:
#   nohup bash run_hires.sh > run_hires.log 2>&1 &
# Needs: a Python env with CUDA torch, bartorch, blochsim, pypulseqpp, brainweb-dl (PYTHON),
# pulserver checkouts of claude/wave-shuffling-prototype (WSHFL, recon) and of the 7 T BrainWeb
# relaxation branch claude/project-thread-8o0ziu (SIM, virtual scanner).
set -euo pipefail
PYTHON=${PYTHON:-python}
WORK=${WORK:-$HOME/pp/wave-shuffling}
WSHFL=${WSHFL:-$WORK/pulserver-wshfl}
SIM=${SIM:-$WORK/pulserver-7t}
mkdir -p "$WORK" && cd "$WORK"
[ -d "$WSHFL" ] || git clone -q -b claude/wave-shuffling-prototype https://github.com/pulserver/pulserver "$WSHFL"
[ -d "$SIM" ] || git clone -q -b claude/project-thread-8o0ziu https://github.com/pulserver/pulserver "$SIM"
if nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -q .; then
  echo "GPU busy, refusing to start"; nvidia-smi; exit 1
fi
P=$WSHFL/prototypes/wave_shuffling
cp -r "$P"/*.py "$P"/plugins "$WORK"/
sed -i "s#/home/user/pulserver-7t#$SIM#g" "$WORK"/scan.py "$WORK"/plugins/*.py
export PYTHONPATH=$SIM/src DEVICE=cuda
# ~1 x 1 x 2 mm: 224 x 224 x 96 over 224 x 224 x 192 mm, so the whole head is inside the FOV.
GEOM="nx=224 ny=224 nslices=96 fov=224 phase_fov=224 slice_thickness=2"
export FOV=0.448,0.224,0.192 NY=224 NZ=96 MATRIX=96,224,224 FOVZYX=0.192,0.224,0.224
scan() { c=$1; shift; [ -f $c/dump.pkl ] && return 0; SCANARGS="--device cuda" $PYTHON scan.py $c "$@" > $c.log 2>&1
  DUMP=$PWD/$c/dump.pkl $PYTHON -c "import sys; from pulserver._cli import main; sys.exit(main(sys.argv[1:]))" replay $c/raw.h5 --plugin $PWD/dump.py --store $c/designs --output $c/dump-out > $c/dump.log 2>&1 || true; }
recon() { c=$1; r=$2; l=$3; b=$4; t=$5; PYTHONPATH=$WSHFL/src:$SIM/src $PYTHON -W ignore proto.py $c $r $l 100 $b -1 $t 2>&1 | grep -E "readouts|grid|resid|solve|peak|rror" | sed "s/^/$c $t /"; }
date
# FSE wave-shuffling R2x2 and shuffling R2x2 (rank 4), ME-GRE wave-CAIPI vs CAIPI at R3x3 (per echo).
scan H_shfl fse3d_shfl $GEOM Ry=2 Rz=2
scan H_wshfl fse3d_wshfl $GEOM Ry=2 Rz=2 user10_value=20 user11_value=3
for c in H_shfl H_wshfl; do $PYTHON -W ignore -c "import pickle,torch; from basis import fse_basis; torch.save(fse_basis(pickle.load(open('$c/dump.pkl.wf','rb'))).basis,'$c/basis_stream.pt')"; recon $c 4 0.02 $c/basis_stream.pt lam0.02; done
MATRIX=96,224,224 $PYTHON -W ignore truth.py H_shfl fse | grep saved
G="$GEOM TR=30000 flip=15 num_echoes=6"
scan G_cart3 gre_multiecho3d_cart $G Ry=3 Rz=3
scan G_wave3 gre_multiecho3d_wshfl $G Ry=3 Rz=3 user10_value=16 user11_value=9
$PYTHON -c "import torch; torch.save(torch.eye(6,dtype=torch.complex64),'g_eye.pt')"
for c in G_cart3 G_wave3; do recon $c 6 0.005 g_eye.pt eye; done
$PYTHON -W ignore truth.py G_cart3 megre | grep saved
date
