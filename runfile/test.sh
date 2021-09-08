#!/usr/bin/env bash
set -x
DATAPATH="/home/yanziwei/data/project-test/Stereo_exp/HITnet/datafile/sceneflow/"
LOGDIR="./worklog"
CUDA_VISIBLE_DEVICES=0 python test.py --dataset middlebury \
    --modelpath "/home/yanziwei/data/project-test/Stereo_exp/HITnet/worklog/experiment_5/bestEPE_checkpoint.ckpt"\
    --savefile "worklog/e5" \
    --datapath $DATAPATH \
    --testlist ./datasets/list/sceneflow_train.list \
    --logdir $LOGDIR \
    --test_batch_size 1 \
    --maxdisp 192
