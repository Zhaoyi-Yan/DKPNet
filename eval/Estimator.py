import random
import math
import os
import numpy as np
import sys
from PIL import Image
from metrics import AEBatch, SEBatch
import time
import torch
import scipy.io as scio
import util.utils as util

class Estimator(object):
    def __init__(self, setting, eval_loader, criterion=torch.nn.MSELoss(reduction="sum")):
        self.datasets_com = ['SHA', 'SHB', 'QNRF', 'NWPU']
        self.setting = setting
        self.ae_batch = AEBatch().to(self.setting.device)
        self.se_batch = SEBatch().to(self.setting.device)
        self.criterion = criterion
        self.eval_loader = eval_loader

    def evaluate(self, net, is_save=False):
        net.eval()
        MAE_, MSE_, loss_ = [], [], []

        # for eval each single dataset
        imgs_cnt = [0, 0, 0, 0] # logging for each dataset
        pred_mae = [0, 0, 0, 0]
        pred_mse = [0, 0, 0, 0]

        rand_number, cur, time_cost = random.randint(0, self.setting.eval_num - 1), 0, 0
        for eval_img_path, eval_img, eval_gt, class_id, _, _, _, _ in self.eval_loader:
            eval_img = eval_img.to(self.setting.device)
            eval_gt = eval_gt.to(self.setting.device)
            class_id = class_id.to(self.setting.device)

            start = time.time()
            eval_patchs = torch.squeeze(eval_img)
            eval_gt_shape = eval_gt.shape
            prediction_map = torch.zeros_like(eval_gt)
            eval_img_path = eval_img_path[0]
      #      img_index = eval_img_index.cpu().numpy()[0]

            with torch.no_grad():
                eval_prediction = net(eval_patchs)
                eval_patchs_shape = eval_prediction.shape
                # test cropped patches
                self.test_crops(eval_patchs_shape, eval_prediction, prediction_map)
                gt_counts = self.get_gt_num(eval_img_path)
                # calculate metrics
                batch_ae = self.ae_batch(prediction_map, gt_counts).data.cpu().numpy()
                batch_se = self.se_batch(prediction_map, gt_counts).data.cpu().numpy()
                loss = self.criterion(prediction_map, eval_gt)
                loss_.append(loss.data.item())
                MAE_.append(batch_ae)
                MSE_.append(batch_se)

                # bz=1
                imgs_cnt[class_id[0].item()] += 1
                pred_mae[class_id[0].item()] += batch_ae[0]
                pred_mse[class_id[0].item()] += batch_se[0]

                if is_save:
                    validate_pred_map = np.squeeze(prediction_map.permute(0, 2, 3, 1).data.cpu().numpy())
                    validate_gt_map = np.squeeze(eval_gt.permute(0, 2, 3, 1).data.cpu().numpy())
                    pred_counts = np.sum(validate_pred_map)
                    util.save_image(util.tensor2im(prediction_map), os.path.join('out_imgs', os.path.splitext(eval_img_path.split('/')[-1])[0]+'_pred.png'))
                    util.save_image(util.tensor2im(eval_gt), os.path.join('out_imgs', os.path.splitext(eval_img_path.split('/')[-1])[0]+'_gt.png'))
                    with open(os.path.join('out_imgs', os.path.splitext(eval_img_path.split('/')[-1])[0]+'.txt'), "w") as f:
                        f.write(str(gt_counts))
                        f.write('\n')
                        f.write(str(pred_counts.item()))

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

    # New Function
    def get_gt_num(self, eval_img_path):
        mat_name = eval_img_path.replace('images', 'ground_truth')[:-4] + ".mat"
        gt_counts = len(scio.loadmat(mat_name)['annPoints'])

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
