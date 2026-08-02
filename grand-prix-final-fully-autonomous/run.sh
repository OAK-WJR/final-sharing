#!/bin/bash
# Grand Prix race run.   bash run.sh            drive at VMAX (default 4.0)
#                        CHECK=1 bash run.sh    static sensor check, no driving
#                        VMAX=2.0 bash run.sh   speed ladder
#                        bash run.sh -s         simulator
cd "$(dirname "$0")/src"
PARAMS=../config/wf_lite_finalloop_params.json \
CTRL=wf_lite.py \
CHECK="${CHECK:-0}" \
VMAX="${VMAX:-4.0}" \
exec python3 race_deploy.py "$@"
