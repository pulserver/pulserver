# run_gr.sh CASE KIND RANK LAM ITERS TAG [extra env via caller]
cd /tmp/claude-0/-home-user/4141ee9b-67e4-5f24-ada4-e68b8f2f7904/scratchpad/wshfl
D=/tmp/claude-0/-home-user/4141ee9b-67e4-5f24-ada4-e68b8f2f7904/scratchpad/devsite
SIM=/home/user/pulserver-7t; WSHFL=/home/user/pulserver-wshfl
c=$1; kind=$2; rank=$3; lam=$4; it=$5; tag=$6
export PYTHONPATH=$D:$SIM/src
[ -f $c/dump.pkl ] || DUMP=$PWD/$c/dump.pkl python -c "import sys; from pulserver._cli import main; sys.exit(main(sys.argv[1:]))" replay $c/raw.h5 --plugin $PWD/dump.py --store $c/designs --output $c/dump-out > $c/dump.log 2>&1; echo "dump $?"
[ -f $c/basis_stream.pt ] || python -W ignore -c "import pickle,torch; from basis import ${kind}_basis; torch.save(${kind}_basis(pickle.load(open('$c/dump.pkl.wf','rb'))).basis,'$c/basis_stream.pt')"; echo basis
[ -f $c/truth.npz ] || MATRIX=120,140,128 FOVZYX=0.24,0.28,0.256 python -W ignore truth.py $c $kind 2>&1 | grep -E "saved|rror"
KEEP=$(python -c "import numpy as np; print(','.join(map(str,np.load('$c/truth.npz')['keep'])))")
export PYTHONPATH=$D:$WSHFL/src:$SIM/src FOV=0.512,0.28,0.24 NY=140 NZ=120 KEEP
/usr/bin/env python -W ignore proto.py $c $rank $lam $it $c/basis_stream.pt -1 $tag 2>&1 | grep -E "readouts|grid|maps|solve|peak|resid|rror"
TRUTH=$c python score.py $c/recon_$tag.npz
