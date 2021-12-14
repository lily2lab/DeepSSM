#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=2
cd ../..
savepath='results/h36m/DeepSSM_long_term/v0'
modelpath='checkpoints/h36m/DeepSSM_long_term/v0'
pretrain_modelpath='checkpoints/h36m/DeepSSM_long_term/v0/model.ckpt-9020'
logname='logs/h36m/DeepSSM_long_term.log'
nohup python -u Train_DeepSSM_h36m.py \
    --is_training True \
    --dataset_name skeleton \
    --train_data_paths data/h36m/h36m_train_3d.npy \
    --valid_data_paths data/h36m/h36m_val_3d.npy  \
    --test_data_paths data/h36m/test_npy \
    --save_dir ${modelpath} \
    --gen_dir ${savepath} \
    --input_length 10 \
    --seq_length 35 \
    --stacklength1 2 \
    --stacklength2 1 \
    --filter_size 3 \
    --lr 0.0001 \
    --batch_size 16 \
    --sampling_stop_iter 0 \
    --max_iterations 300000 \
    --display_interval 10 \
    --test_interval 500 \
    --n_gpu 1 \
    --snapshot_interval 500  >>${logname}  2>&1 &

tail -f ${logname}

# --pretrained_model ${pretrain_modelpath}  \



