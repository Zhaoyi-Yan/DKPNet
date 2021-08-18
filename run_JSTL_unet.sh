#!/bin/bash
python train_fast.py --dataset_name='JSTL_large_4' --gpu_ids='2,3' --optimizer='adam' --start_eval_epoch=250 --max_epochs=400 --lr=5e-5 --base_mae='65,15,110,100' --name='res_unet_aspp' --net_name='res_unet_aspp' --batch_size=18 --nThreads=18 --eval_per_epoch=2
