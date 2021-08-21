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

# Train
```bash
sh run_JSTL.sh
```

# Training notes
There are two types of training scripts: `train_fast` and `train_slow`.
The main differences between them exist in the evaluation procedure.
In `train_slow`, the test images are processed in the main GPU, making the whole training very slow.
As the sizes of test images vary largely with each other (the maximum size / the minimun size equals up to 5x !), making the batch size of evaluation can only be `1` on a single GPU.
From our observation, the bottleneck lies in the evaluation stage (Maybe 10x computation time longer than the training time), it is not meaningful enough if you train the whole dataset with more GPUs as long as the evaluation processing is still on a single GPU.
To this end, we manage to **evaluate two images on two GPUs at the same time**, as what `train_fast` does.
We think two GPUs are enough for training the whole dataset in the affordable time (~2 days).

It is notable that the batch size of training should be no smaller than `32`, or the performance may degrade to some extent.


# Test
Download the pretrained model via
```bash
bash download_models.sh
```

And put the model into folder `./output/HRNet_relu_aspp/JSTL_large_4/`

```bash
python test.py
```

# Citation
If you find our work useful or our work gives you any insights, please cite:
```
@inproceedings{yan2019perspective,
  title = {Variational Attention: Propagating Domain-Specific Knowledge for Multi-Domain Learning in Crowd Counting},
  author = {Chen, Binghui and Yan, Zhaoyi and Li, Ke and Li, Pengyu and Wang, Biao and Zuo, Wangmeng and Zhang, Lei}
  booktitle = {The IEEE International Conference on Computer Vision (ICCV)},
  month = {October},
  year = {2021}
}
```