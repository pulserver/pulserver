cd /tmp/claude-0/-home-user/4141ee9b-67e4-5f24-ada4-e68b8f2f7904/scratchpad/wshfl
D=/tmp/claude-0/-home-user/4141ee9b-67e4-5f24-ada4-e68b8f2f7904/scratchpad/devsite
export PYTHONPATH=$D:/home/user/pulserver-7t/src PULSERVER_AHEAD_SAMPLES=262144
c=$1; shift
( while sleep 10; do ps -eo rss,args | grep "[_]cli import main" | awk '{s+=$1} END {print strftime("%T"), s/1e6, "GB"}'; done ) > $c.mem &
M=$!
date; python scan.py $c "$@" > $c.log 2>&1; echo "scan $?"; date
kill $M; sort -k2 -n $c.mem | tail -1
