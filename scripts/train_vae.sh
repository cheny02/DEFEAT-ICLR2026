#!/bin/bash
cd ..
# custom config
DATA=/data/dataset/
TRAINER=DEFEAT
DATASET=$1
CFG=$2  # config file
CTP=end  # class token position (end or middle)
NCTX=$3  # number of context tokens
SHOTS=$4  # number of shots (1, 2, 4, 8, 16)
CSC=False  # class-specific context (False or True)
EPS=$5  # epsilon for AT
ALPHA=$6  # alpha or step size for AT
STEPS=$7  # number of steps for AT
SEED=0
W1=$8
W2=$9
W3=${10}
W4=${11}
A=${12}
ROB_TEST=False

DIR=output_${TRAINER}/${CFG}_${SHOTS}shots/nctx${NCTX}_csc${CSC}_ctp${CTP}/eps${EPS}_alpha${ALPHA}_step${STEPS}/W1_${W1}_W2_${W2}_W3_${W3}_W4_${W4}_A_${A}/${DATASET}/seed${SEED}

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
    TRAINER.COOP.N_CTX ${NCTX} \
    TRAINER.COOP.CSC ${CSC} \
    TRAINER.COOP.CLASS_TOKEN_POSITION ${CTP} \
    DATASET.NUM_SHOTS ${SHOTS}\
    TRAINER.DEFEAT.ATK_TEST ${ROB_TEST}\
    TRAINER.COOP.W1 ${W1}\
    TRAINER.COOP.W2 ${W2}\
    TRAINER.COOP.W3 ${W3}\
    TRAINER.COOP.W4 ${W4}\
    TRAINER.COOP.ALPHA ${A}\  
fi
