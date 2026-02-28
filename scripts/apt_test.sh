#!/bin/bash

cd ..

# custom config
DATA=/data/dataset/
DATASET=$1
CFG=$2
LOADEP=$3
CTP=end  # class token position (end or middle)
NCTX=$4  # number of context tokens
SHOTS=$5  # number of shots (1, 2, 4, 8, 16)
CSC=False  # class-specific context (False or True)
EPS=$6
ALPHA=$7
STEPS=$8
SEED=0
ATP=onfly
PALPHA=0
TRAINER=APT
ROB_TEST=True

MODEL_DIR=output_${TRAINER}/${CFG}_${SHOTS}shots/nctx${NCTX}/eps${EPS}_alpha${ALPHA}_step${STEPS}/${DATASET}/seed${SEED}
DIR=output_${TRAINER}/${CFG}_${SHOTS}shots/nctx${NCTX}/eps${EPS}_alpha${ALPHA}_step${STEPS}/${DATASET}/seed${SEED}/test
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
    --adv-prompt ${ATP} \
    --prompt-alpha ${PALPHA} \
    --eval-only \
    TRAINER.COOP.N_CTX ${NCTX} \
    TRAINER.COOP.CSC ${CSC} \
    TRAINER.COOP.CLASS_TOKEN_POSITION ${CTP} \
    DATASET.NUM_SHOTS ${SHOTS} \
    TRAINER.DEFEAT.ATK_TEST ${ROB_TEST}\
    
fi

