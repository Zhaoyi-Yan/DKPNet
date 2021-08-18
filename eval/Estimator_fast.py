import random
import math
import copy
import numpy as np
import sys
from PIL import Image
from metrics import AEBatch, SEBatch
import time
import torch
import scipy.io as scio

class Estimator(object):
    def __init__(self, opt, setting, eval_loader, criterion=torch.nn.MSELoss(reduction="sum")):
        self.datasets_com = ['SHA', 'SHB', 'QNRF', 'NWPU']
        self.setting = setting
        self.ae_batch = AEBatch().to(self.setting.device)
        self.se_batch = SEBatch().to(self.setting.device)
        self.criterion = criterion
        self.opt = opt
        self.criterion2 = copy.deepcopy(self.criterion).to(self.opt.gpu_ids[1])
        self.eval_loader = eval_loader

    def evaluate(self, net, is_show=False):
        net.eval()
        net2 = copy.deepcopy(net)
        net2 = net2.module.to(self.opt.gpu_ids[1]) # another gpu
        MAE_, MSE_, loss_ = [], [], []

        # for eval each single dataset
        imgs_cnt = [0, 0, 0, 0] # logging for each dataset
        pred_mae = [0, 0, 0, 0]
        pred_mse = [0, 0, 0, 0]

        rand_number, cur, time_cost = random.randint(0, self.setting.eval_num - 1), 0, 0
        for eval_img_path, eval_img, eval_gt, class_id, ph_min, pw_min, idx_h, idx_w in self.eval_loader:

            class_id = class_id.to(self.setting.device)

            start = time.time()
            # remove extra dim
            eval_patchs_tmp = torch.squeeze(eval_img[0].to(self.opt.gpu_ids[0]))
            eval_patchs2_tmp = torch.squeeze(eval_img[1].to(self.opt.gpu_ids[1]))

            # crop the patch to get the real patch
            ph_min, pw_min, idx_h, idx_w = ph_min.item(), pw_min.item(), idx_h.item(), idx_w.item()
            if idx_h + idx_w == 0:
                eval_patchs = eval_patchs_tmp[:, :, :ph_min, :pw_min]
                eval_patchs2 = eval_patchs2_tmp
            elif idx_h + idx_w == 2:
                eval_patchs = eval_patchs_tmp
                eval_patchs2 = eval_patchs2_tmp[:, :, :ph_min, :pw_min]
            elif idx_h + idx_w == 1:
                if idx_h == 0:
                    eval_patchs = eval_patchs_tmp[:, :, :ph_min, :]
                    eval_patchs2 = eval_patchs2_tmp[:, :, :, :pw_min]
                else:
                    eval_patchs = eval_patchs_tmp[:, :, :, :pw_min]
                    eval_patchs2 = eval_patchs2_tmp[:, :, :ph_min, :]

            eval_gt_1 = eval_gt[0].to(self.opt.gpu_ids[0])
            eval_gt_2 = eval_gt[1].to(self.opt.gpu_ids[1])
            eval_gt_shape = eval_gt_1.shape
            eval_gt_shape2 = eval_gt_2.shape

            prediction_map = torch.zeros_like(eval_gt_1)
            prediction_map2 = torch.zeros_like(eval_gt_2)
            eval_img_path0 = eval_img_path[0]
            eval_img_path2 = eval_img_path[1]

            with torch.no_grad():
                eval_prediction = net(eval_patchs)
                eval_prediction2 = net2(eval_patchs2)
                eval_patchs_shape = eval_prediction.shape
                eval_patchs_shape2 = eval_prediction2.shape
                # test cropped patches
                self.test_crops(eval_patchs_shape, eval_prediction, prediction_map)
                self.test_crops(eval_patchs_shape2, eval_prediction2, prediction_map2)
                gt_counts = self.get_gt_num(eval_img_path0)
                gt_counts2 = self.get_gt_num(eval_img_path2)
                # calculate metrics
                batch_ae = self.ae_batch(prediction_map, gt_counts).data.cpu().numpy()
                batch_ae2 = self.ae_batch(prediction_map2, gt_counts2).data.cpu().numpy()
                batch_se = self.se_batch(prediction_map, gt_counts).data.cpu().numpy()
                batch_se2 = self.se_batch(prediction_map2, gt_counts2).data.cpu().numpy()

                loss = self.criterion(prediction_map, eval_gt_1)
                loss2 = self.criterion2(prediction_map2, eval_gt_2)
                loss_.append(loss.data.item())
                loss_.append(loss2.data.item())
                MAE_.append(batch_ae)
                MAE_.append(batch_ae2)
                MSE_.append(batch_se)
                MSE_.append(batch_se2)

                # bz=2
                imgs_cnt[class_id[0].item()] += 1
                pred_mae[class_id[0].item()] += batch_ae[0]
                pred_mse[class_id[0].item()] += batch_se[0]

                imgs_cnt[class_id[1].item()] += 1
                pred_mae[class_id[1].item()] += batch_ae2[0]
                pred_mse[class_id[1].item()] += batch_se2[0]



            cur += 1
            torch.cuda.synchronize()
            end = time.time()
            time_cost += (end - start)

        # cal mae, mse for each dataset
        pred_mae = np.array(pred_mae)
        pred_mse = np.array(pred_mse)
        imgs_cnt = np.array(imgs_cnt)

        pred_mae = pred_mae / imgs_cnt
        pred_mse = pred_mse / imgs_cnt
        pred_mse = np.sqrt(pred_mse)

        # return the validate loss, validate MAE and validate RMSE
        MAE_, MSE_, loss_ = np.reshape(MAE_, [-1]), np.reshape(MSE_, [-1]), np.reshape(loss_, [-1])
        return np.mean(MAE_), np.sqrt(np.mean(MSE_)), np.mean(loss_), time_cost, pred_mae, pred_mse

    def get_cur_dataset(self, img_name):
        check_list = [img_name.find(da) for da in self.datasets_com]
        check_list = np.array(check_list)
        cur_idx = np.where(check_list != -1)[0][0]
        return self.datasets_com[cur_idx]


    # New Function
    def get_gt_num(self, eval_img_path):
        mat_name = eval_img_path.replace('images', 'ground_truth')[:-4] + ".mat"
        gt_counts = len(scio.loadmat(mat_name)['annPoints'])

        return gt_counts

    # infer the gt mat names from img names
    # Very specific for this repo
    # SHA/SHB: SHA_IMG_85.jpg --> SHA_GT_IMG_85.mat
    # QNRF: QNRF_img_0001.jpg --> QNRF_img_0001_ann.mat
    def get_gt_num_old(self, eval_img_path):
        mat_name = eval_img_path.replace('.jpg', '.mat').replace('images', 'ground_truth')
        cur_dataset = self.get_cur_dataset(mat_name)

        if cur_dataset.find("QNRF", -1):
            mat_name = mat_name.replace('.mat', '_ann.mat')
            gt_counts = len(scio.loadmat(mat_name)['annPoints'])
        elif cur_dataset == "SHA" or cur_dataset == "SHB":
            mat_name = mat_name.replace('IMG', 'GT_IMG')
            gt_counts = len(scio.loadmat(mat_name)['image_info'][0][0][0][0][0])
        else:
            raise NameError("No such dataset, only support SHA, SHB, QNRF")
        return gt_counts


    def test_crops(self, eval_shape, eval_p, pred_m):
        for i in range(3):
            for j in range(3):
                start_h, start_w = math.floor(eval_shape[2] / 4), math.floor(eval_shape[3] / 4)
                valid_h, valid_w = eval_shape[2] // 2, eval_shape[3] // 2
                pred_h = math.floor(3 * eval_shape[2] / 4) + (eval_shape[2] // 2) * (i - 1)
                pred_w = math.floor(3 * eval_shape[3] / 4) + (eval_shape[3] // 2) * (j - 1)
                if i == 0:
                    valid_h = math.floor(3 * eval_shape[2] / 4)
                    start_h = 0
                    pred_h = 0
                elif i == 2:
                    valid_h = math.ceil(3 * eval_shape[2] / 4)

                if j == 0:
                    valid_w = math.floor(3 * eval_shape[3] / 4)
                    start_w = 0
                    pred_w = 0
                elif j == 2:
                    valid_w = math.ceil(3 * eval_shape[3] / 4)
                pred_m[:, :, pred_h:pred_h + valid_h, pred_w:pred_w + valid_w] += eval_p[i * 3 + j:i * 3 + j + 1, :,start_h:start_h + valid_h, start_w:start_w + valid_w]
