#!/bin/bash

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}/.."

if [ "$#" -lt 1 ]; then
    echo "Usage: bash scripts/fap_train.sh DATASET [CFG] [NCTX] [SHOTS] [EPS] [ALPHA] [STEPS] [LAMBDA_1] [SEED]"
    exit 1
fi

DATA=${DATA:-/data/dataset/}
DATASET=$1
CFG=${2:-vit_b32_ep10}
NCTX=${3:-2}
SHOTS=${4:-16}
EPS=${5:-4}
ALPHA=${6:-2.67}
STEPS=${7:-3}
LAMBDA_1=${8:-1.5}
SEED=${9:-1}
ADV_TERM=cos
TRAINER=FAP
ROB_TEST=False

DIR=output_${TRAINER}/${CFG}_${SHOTS}shots/nctx${NCTX}/eps${EPS}_alpha${ALPHA}_step${STEPS}/lambda${LAMBDA_1}/${DATASET}/seed${SEED}

if [ -d "$DIR" ]; then
    echo "Oops! The results exist at ${DIR} (so skip this job)"
else
    python train.py \
    --root ${DATA} \
    --seed ${SEED} \
    --trainer ${TRAINER} \
    --dataset-config-file configs/datasets/${DATASET}.yaml \
    --config-file configs/trainers/${TRAINER}/${CFG}.yaml \
    --output-dir ${DIR} \
    --eps ${EPS} \
    --alpha ${ALPHA} \
    --steps ${STEPS} \
    TRAINER.FAP.N_CTX ${NCTX} \
    TRAINER.DEFEAT.ATK_TEST ${ROB_TEST} \
    ATTACK.PGD.ADV_TERM ${ADV_TERM} \
    ATTACK.PGD.LAMBDA_1 ${LAMBDA_1} \
    DATASET.NUM_SHOTS ${SHOTS}
fi
