#!/usr/bin/env bash
# Wave-shuffling high-resolution runs on the GPU host, one job at a time, detached:
#   nohup bash run_hires.sh > run_hires.log 2>&1 &
# Needs: a Python env with CUDA torch, bartorch, blochsim, pypulseqpp, brainweb-dl (PYTHON),
# pulserver checkouts of claude/wave-shuffling-prototype (WSHFL, recon) and of the 7 T BrainWeb
# relaxation branch claude/project-thread-8o0ziu (SIM, virtual scanner).
# Each recon is a replay through the subspace plugin; measure.py prints wall time, peak RAM, peak VRAM.
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
# 1 x 1 x 2 mm: 224 x 224 x 96 over 224 x 224 x 192 mm, so the whole head is inside the FOV.
GEOM="nx=224 ny=224 nslices=96 fov=224 phase_fov=224 slice_thickness=2"
export MATRIX=96,224,224 FOVZYX=0.192,0.224,0.224
scan() { c=$1; shift; [ -f $c/raw.h5 ] && return 0; SCANARGS="${SCANARGS:-} --device cuda" $PYTHON scan.py $c "$@" > $c.log 2>&1; echo "$c scan $? $(date +%T)"; }
dump() { c=$1; DUMP=$PWD/$c/dump.pkl $PYTHON -c "import sys; from pulserver._cli import main; sys.exit(main(sys.argv[1:]))" replay $c/raw.h5 --plugin $PWD/dump.py --store $c/designs --output $c/dump-out > $c/dump.log 2>&1 || true; }
plug() { c=$1; p=$2; rm -rf $c/plug-out; PYTHONPATH=$WSHFL/src:$SIM/src $PYTHON -W ignore measure.py $c/raw.h5 $PWD/$p $c/designs $c/plug-out 2>&1 | grep MEASURE | sed "s/^/$c /"; $PYTHON -W ignore plug2npz.py $c > /dev/null; }
date; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
# ME-GRE R3x3: CAIPI, wave-CAIPI, and wave-CAIPI with a 6 us + 20 us GIRF error (calibrated in the plugin).
# Wave about 1.8x the readout gradient so the spread stays inside the 2x oversampled readout.
G="$GEOM TR=44000 flip=15 num_echoes=6 Ry=3 Rz=3 bandwidth=140000"
W="user10_value=13 user11_value=5"
scan G_cart gre_multiecho3d_cart $G
scan G_wave gre_multiecho3d_wshfl $G $W
SCANARGS="--girf 6 20" scan G_girf gre_multiecho3d_wshfl $G $W
for c in G_cart G_wave G_girf; do plug $c megre_subspace.py; done
dump G_wave; $PYTHON -W ignore truth.py G_wave megre | grep saved
TRUTH=G_wave $PYTHON score.py G_cart/recon_plugin.npz G_wave/recon_plugin.npz G_girf/recon_plugin.npz
# FSE R2x2: T2 shuffling and wave-shuffling (rank 4).
scan H_shfl fse3d_shfl $GEOM Ry=2 Rz=2
scan H_wshfl fse3d_wshfl $GEOM Ry=2 Rz=2 user10_value=20 user11_value=3
for c in H_shfl H_wshfl; do plug $c fse_subspace.py; done
dump H_shfl; $PYTHON -W ignore truth.py H_shfl fse | grep saved
TRUTH=H_shfl $PYTHON score.py H_shfl/recon_plugin.npz H_wshfl/recon_plugin.npz
date
