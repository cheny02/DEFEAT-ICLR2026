#!/bin/bash

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}/.."

if [ "$#" -lt 1 ]; then
    echo "Usage: bash scripts/fap_test.sh DATASET [CFG] [LOADEP] [NCTX] [SHOTS] [EPS] [ALPHA] [STEPS] [LAMBDA_1] [ATTACK] [FP] [SEED]"
    exit 1
fi

DATA=${DATA:-/data/dataset/}
DATASET=$1
CFG=${2:-vit_b32_ep10}
LOADEP=${3:-10}
NCTX=${4:-2}
SHOTS=${5:-16}
EPS=${6:-4}
ALPHA=${7:-2.67}
STEPS=${8:-3}
LAMBDA_1=${9:-1.5}
ATTACK=${10:-pgd}
FP=${11:-fp16}
SEED=${12:-1}
ADV_TERM=cos
TRAINER=FAP
ROB_TEST=True

COMMON_DIR=${CFG}_${SHOTS}shots/nctx${NCTX}/eps${EPS}_alpha${ALPHA}_step${STEPS}/lambda${LAMBDA_1}/${DATASET}/seed${SEED}
MODEL_DIR=output_${TRAINER}/${COMMON_DIR}
DIR=output_${TRAINER}_${ATTACK}/${COMMON_DIR}/test

if [ -d "$DIR" ]; then
    echo "Results are available in ${DIR}. Skip this job"
else
    echo "Run this job and save the output to ${DIR}"
    python train.py \
    --root ${DATA} \
    --seed ${SEED} \
    --trainer ${TRAINER} \
    --dataset-config-file configs/datasets/${DATASET}.yaml \
    --config-file configs/trainers/${TRAINER}/${CFG}.yaml \
    --output-dir ${DIR} \
    --model-dir ${MODEL_DIR} \
    --load-epoch ${LOADEP} \
    --eps ${EPS} \
    --alpha ${ALPHA} \
    --steps ${STEPS} \
    --eval-only \
    TRAINER.FAP.N_CTX ${NCTX} \
    TRAINER.FAP.PREC ${FP} \
    TRAINER.DEFEAT.ATK_TEST ${ROB_TEST} \
    TRAINER.DEFEAT.ATK ${ATTACK} \
    ATTACK.PGD.ADV_TERM ${ADV_TERM} \
    ATTACK.PGD.LAMBDA_1 ${LAMBDA_1} \
    DATASET.NUM_SHOTS ${SHOTS}
fi
