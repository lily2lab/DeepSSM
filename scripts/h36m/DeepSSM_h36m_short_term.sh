#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=2
cd ../..
savepath='results/h36m/DeepSSM_short_term/v0'
modelpath='checkpoints/h36m/DeepSSM_short_term/v0'
pretrain_modelpath='checkpoints/h36m/DeepSSM_short_term/v0/model.ckpt-21240'
logname='logs/h36m/DeepSSM_short_term.log'
nohup python -u Train_DeepSSM_h36m.py \
    --is_training True \
    --dataset_name skeleton \
    --train_data_paths data/h36m20/h36m20_train_3d.npy \
    --valid_data_paths data/h36m20/h36m20_val_3d.npy \
    --test_data_paths data/h36m20/test20_npy \
    --save_dir ${modelpath} \
    --gen_dir ${savepath} \
    --input_length 10 \
    --seq_length 20 \
    --stacklength 2 \
    --stacklength 1 \
    --filter_size 3 \
    --lr 0.0001 \
    --batch_size 16 \
    --sampling_stop_iter 0 \
    --max_iterations 3000000 \
    --display_interval 10 \
    --test_interval 500 \
    --n_gpu 1 \
    --snapshot_interval 500 >>${logname}  2>&1 &

tail -f ${logname}

# --pretrained_model ${pretrain_modelpath}  \



