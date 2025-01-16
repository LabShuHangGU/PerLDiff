#!/usr/bin/env bash

CONFIG=$1
BATCH_SIZE=$2
NAME=$3
FLAG=$4
GPUS=$5
PORT=${PORT:-29503}

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
# python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
#     $(dirname "$0")/test.py $CONFIG $CHECKPOINT --launcher pytorch ${@:4} --eval bbox

python -m torch.distributed.launch --nproc_per_node=$GPUS main.py --master_port=$PORT --yaml_file $CONFIG   --batch_size $ BATCH_SIZE --name $NAME --nuscenes_val_gen $FLAG --launcher pytorch ${@:6}