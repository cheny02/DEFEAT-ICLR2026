#!/bin/bash

cd ..

# custom config
DATA=/data/dataset/
DATASET=$1
CFG=$2  # config file
CTP=end  # class token position (end or middle)
NCTX=$3  # number of context tokens
SHOTS=$4  # number of shots (1, 2, 4, 8, 16)
CSC=False  # class-specific context (False or True)
EPS=$5  # epsilon for AT
ALPHA=$6  # alpha or step size for AT
STEPS=$7  # number of steps for AT
ATP=onfly
PALPHA=0
TRAINER=APT
ROB_TEST=False
SEED=0

DIR=output_${TRAINER}/${CFG}_${SHOTS}shots/nctx${NCTX}/eps${EPS}_alpha${ALPHA}_step${STEPS}/${DATASET}/seed${SEED}

if [ -d "$DIR" ]; then
    echo "Oops! The results exist at ${DIR} (so skip this job)"
else
    python train.py \
    --root ${DATA} \
    --trainer ${TRAINER} \
    --dataset-config-file configs/datasets/${DATASET}.yaml \
    --config-file configs/trainers/${TRAINER}/${CFG}.yaml \
    --output-dir ${DIR} \
    --eps ${EPS} \
    --alpha ${ALPHA} \
    --steps ${STEPS} \
    --adv-prompt ${ATP} \
    --prompt-alpha ${PALPHA} \
    TRAINER.COOP.N_CTX ${NCTX} \
    TRAINER.COOP.CSC ${CSC} \
    TRAINER.COOP.CLASS_TOKEN_POSITION ${CTP} \
    TRAINER.DEFEAT.ATK_TEST ${ROB_TEST}\
    DATASET.NUM_SHOTS ${SHOTS}\
    
fi
