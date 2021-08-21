#!/bin/bash
python train_slow.py --dataset_name='JSTL_large_4' --gpu_ids='0,1' --optimizer='adam' --start_eval_epoch=250 --max_epochs=400 --lr=5e-5 --base_mae='64,15,110,100' --name='HRNet_relu_aspp' --net_name='hrnet_aspp_relu' --batch_size=32 --nThreads=32 --eval_per_epoch=2
