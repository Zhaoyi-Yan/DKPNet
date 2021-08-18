# DKPNet
ICCV 2021
Variational Attention: Propagating Domain-Specific Knowledge for Multi-Domain Learning in Crowd Counting

Baseline of DKPNet is available.

**Currently, only code of DKPNet-baseline  is released.**

# Datasets Preparation
Download the datasets `ShanghaiTech A`, `ShanghaiTech B`, `UCF-QNRF` and `NWPU`
Then generate the density maps via `generate_density_map_perfect_names_SHAB_QNRF_NWPU_JHU.py`.
After that, create a folder named `JSTL_large_4_dataset`, and directly copy all the processed data in `JSTL_large_4_dataset`.

The tree of the folder should be:
```bash
`DATASET` is `SHA`, `SHB`, `QNRF_large` or `NWPU_large`.

-JSTL_large_dataset
   -den
       -test
            -Npy files with the name of DATASET_img_xxx.npy, which logs the info of density maps.
       -train
            -Npy files with the name of DATASET_img_xxx.npy, which logs the info of density maps.
   -ori
       -test_data
            -ground_truth
                 -MAT files with the name of DATASET_img_xxx.mat, which logs the original dot annotations.
            -images
                 -JPG files with the name of DATASET_img_xxx.mat, which logs the original image files.
       -train_data
            -ground_truth
                 -MAT files with the name of DATASET_img_xxx.mat, which logs the original dot annotations.
            -images
                 -JPG files with the name of DATASET_img_xxx.mat, which logs the original image files.
```

Download the pretrained hrnet model `HRNet-W40-C` from the link `https://github.com/HRNet/HRNet-Image-Classification` and put it directly in the root path of the repository.
%
After doing that, download the pretrained model via
```bash
bash download_models.sh
```
And put the model into folder './output', change the model name in `test.sh` or `test_fast.sh` scripts.

# Train
```bash
sh run_JSTL.sh
```

# Test
```bash
python test.py
```

