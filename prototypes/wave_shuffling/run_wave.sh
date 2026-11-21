#!/usr/bin/env bash
# ME-GRE R3x3, 64x64x48, 40 kHz readout so an 8 mT/m 5-cycle wave is not slew-limited.
cd /tmp/claude-0/-home-user/4141ee9b-67e4-5f24-ada4-e68b8f2f7904/scratchpad/wshfl
D=/tmp/claude-0/-home-user/4141ee9b-67e4-5f24-ada4-e68b8f2f7904/scratchpad/devsite
SIM=/home/user/pulserver-7t; WSHFL=/home/user/pulserver-wshfl
export PYTHONPATH=$D:$SIM/src FOV=0.48,0.24,0.192 NY=64 NZ=48
G="nx=64 ny=64 nslices=48 fov=240 phase_fov=240 slice_thickness=4 TR=32000 flip=15 num_echoes=6 Ry=3 Rz=3 bandwidth=40000"
W="user10_value=8 user11_value=5"
scan() { c=$1; shift; [ -f $c/dump.pkl ] && return 0; python scan.py $c "$@" > $c.log 2>&1; echo "$c scan $?"
  DUMP=$PWD/$c/dump.pkl python -c "import sys; from pulserver._cli import main; sys.exit(main(sys.argv[1:]))" replay $c/raw.h5 --plugin $PWD/dump.py --store $c/designs --output $c/dump-out > $c/dump.log 2>&1; echo "$c dump $?"; }
recon() { c=$1; t=$2; shift 2; env "$@" PYTHONPATH=$D:$WSHFL/src:$SIM/src python -W ignore proto.py $c 6 0.005 100 g_eye.pt -1 $t 2>&1 | grep -E "readouts|calib|resid|Maximum resident|Elapsed" | sed "s/^/$c $t /"; }
date
scan w_cart gre_multiecho3d_cart $G
scan w_wave gre_multiecho3d_wshfl $G $W
SCANARGS="--girf 6 20" scan w_girf gre_multiecho3d_wshfl $G $W
date
recon w_cart eye
recon w_wave eye
recon w_girf eye
recon w_girf cal CALIBRATE=1
recon w_wave cal CALIBRATE=1
for c in w_cart w_wave; do MATRIX=48,64,64 FOVZYX=0.192,0.24,0.24 python -W ignore truth.py $c megre | grep saved; done
TRUTH=w_cart python score.py w_cart/recon_eye.npz
TRUTH=w_wave python score.py w_wave/recon_eye.npz w_wave/recon_cal.npz w_girf/recon_eye.npz w_girf/recon_cal.npz
date
