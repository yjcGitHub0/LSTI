from data_provider.data_factory_my import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping_my, adjust_learning_rate, visual
from utils.metrics import metric
import torch
import torch.nn as nn
from torch import optim
import os
import time
import warnings
import numpy as np
import copy
import random
import torch.nn.functional as F
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from layers.MLP import Projector

warnings.filterwarnings('ignore')


class Exp_LSTI_onlyLong(Exp_Basic):
    def __init__(self, args):
        super(Exp_LSTI_onlyLong, self).__init__(args)
        self.model_back = self._build_model().to(self.device)

    def _build_model(self):
        args_pred = copy.deepcopy(self.args)
        args_pred.task_name = "long_term_forecast"
        args_pred.pred_len = self.args.seq_len + self.args.pred_len
        self.args_pred = args_pred
        model = self.model_dict[self.args.model].Model(args_pred).float()
        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self, model):
        model_optim = optim.Adam(model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        self.model.eval()
        self.model_back.eval()

        mask_tot = np.isnan(vali_data.data_miss)
        true = vali_data.data_x[mask_tot].copy()
        true = np.nan_to_num(true)

        pred_len = self.args.pred_len
        with torch.no_grad():
            data_miss = vali_data.data_miss.copy()
            seq_len = self.args.seq_len
            i=seq_len
            while(i < len(data_miss)):
                mask_tmp = np.isnan(data_miss[i, :])
                if np.any(mask_tmp): 
                    mask = np.isnan(data_miss[i : i + pred_len, :])
                    series = data_miss[i - seq_len : i, :]
                    series = torch.tensor(series).float().to(self.device).unsqueeze(0)
                    series_mark = vali_data.data_stamp[i - seq_len : i, :]
                    series_mark = torch.tensor(series_mark).float().to(self.device).unsqueeze(0)
                    
                    outputs = self.model(series, series_mark, None, None)[:, -pred_len:, :]
                    outputs = outputs.squeeze(0).cpu().numpy()
                    outputs[~mask] = (data_miss[i : i + pred_len, :])[~mask]

                    data_miss[i : i + pred_len, :] = outputs
                    i = i + pred_len
                else:
                    i = i + 1
            
            pred_front = data_miss.copy()
            
        with torch.no_grad():
            data_miss_b = vali_data.data_miss
            data_miss_b = data_miss_b[::-1, :].copy()
            data_stamp = vali_data.data_stamp
            data_stamp = data_stamp[::-1, :].copy()

            seq_len = self.args.seq_len
            i=seq_len
            while(i < len(data_miss_b)):
                mask_tmp = np.isnan(data_miss_b[i, :])
                if np.any(mask_tmp): 
                    mask_b = np.isnan(data_miss_b[i : i + pred_len, :])
                    series_b = data_miss_b[i - seq_len : i, :]
                    series_b = torch.tensor(series_b).float().to(self.device).unsqueeze(0)
                    series_mark_b = data_stamp[i - seq_len : i, :]
                    series_mark_b = torch.tensor(series_mark_b).float().to(self.device).unsqueeze(0)
                    outputs = self.model_back(series_b, series_mark_b, None, None)[:, -pred_len:, :]
                    outputs = outputs.squeeze(0).cpu().numpy()
                    outputs[~mask_b] = (data_miss_b[i : i + pred_len, :])[~mask_b]

                    data_miss_b[i : i + pred_len, :] = outputs
                    i=i + pred_len
                else:
                    i=i + 1
            
            pred_back = data_miss_b[::-1, :]
            pred_back = pred_back.copy()

        l=r=0
        pred = (pred_front + pred_back) / 2
        while(l < len(pred_front) and r < len(pred_front)):
            mask = np.isnan(vali_data.data_miss[r, :])
            if np.any(mask): 
                r = r + 1
            else:
                if l == r:
                    l = l + 1
                    r = r + 1
                else:
                    ml = r - l
                    if ml != 1 :
                        weight_f = np.arange(ml - 1, -1, -1).astype(float) / (ml - 1)
                        weight_b = np.arange(ml).astype(float) / (ml - 1)
                        pred[l:r, :] = (pred_front[l:r, :] * weight_f.reshape(ml, 1) + pred_back[l:r, :] * weight_b.reshape(ml, 1))
                    r = r + 1
                    l = r

        impute_result = pred[mask_tot]
        data_impute = pred[mask_tot]
        pred = pred[mask_tot]
        pred_front = pred_front[mask_tot]
        pred_back = pred_back[mask_tot]

        loss = mean_squared_error(impute_result, true)
        loss_pred = mean_squared_error(pred, true)
        loss_impute = mean_squared_error(data_impute, true)
        loss_front = mean_squared_error(pred_front, true)
        loss_back = mean_squared_error(pred_back, true)
        print('loss: {}  loss_pred: {}  loss_impute: {}  loss_front: {}  loss_back: {}'.format(loss, loss_pred, loss_impute, loss_front, loss_back))
                        
        total_loss = torch.tensor(loss)
        self.model.train()
        self.model_back.train()
        return total_loss


    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        path = os.path.join(self.args.checkpoints, setting)
        path_back = os.path.join(self.args.checkpoints, setting, 'back')
        if not os.path.exists(path):
            os.makedirs(path)
        if not os.path.exists(path_back):
            os.makedirs(path_back)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping_my(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer(self.model)
        model_optim_back = self._select_optimizer(self.model_back)
        criterion = self._select_criterion()

        torch.autograd.set_detect_anomaly(True)
        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []
            train_loss_f = []
            train_loss_b = []

            self.model.train()
            self.model_back.train()
            epoch_time = time.time()
            for i, (batch_x, batch_x_mark, batch_y, batch_y_mark, batch_z, batch_z_mark) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()
                model_optim_back.zero_grad()

                batch_x = batch_x.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)
                batch_z = batch_z.float().to(self.device)
                batch_z_mark = batch_z_mark.float().to(self.device)

                seq_len = self.args.seq_len
                pred_len = self.args.pred_len
                inputs_front = batch_x
                stamp_front = batch_x_mark
                inputs_back = torch.flip(batch_z, dims=[1]).clone()
                stamp_back = torch.flip(batch_z_mark, dims=[1])

                outputs_f = self.model(inputs_front, stamp_front, None, None)
                outputs_b = self.model_back(inputs_back, stamp_back, None, None)
                outputs_f_t = outputs_f[:, -pred_len:, :]
                outputs_b_t = outputs_b[:, -pred_len:, :]

                outputs_b = torch.flip(outputs_b, dims=[1])
                outputs_b_t = torch.flip(outputs_b_t, dims=[1])

                weight_f = torch.arange(pred_len - 1, -1, -1).to(self.device).float() / (pred_len - 1)
                weight_f = weight_f.view(1, pred_len, 1)
                weight_b = torch.arange(0, pred_len, 1).to(self.device).float() / (pred_len - 1)
                weight_b = weight_b.view(1, pred_len, 1)
                pred = (outputs_f_t * weight_f + outputs_b_t * weight_b)
                impute_result = pred

                loss_f = criterion(outputs_f, torch.cat((batch_x, batch_y), dim=1))
                loss_b = criterion(outputs_b, torch.cat((batch_y, batch_z), dim=1))
                loss_c = criterion(outputs_f_t, outputs_b_t)
                loss_pred = criterion(pred, batch_y)
                loss = loss_f + loss_b + loss_c + loss_pred

                loss.backward()
                model_optim.step()
                model_optim_back.step()
                train_loss.append(loss.item())
                train_loss_f.append(loss_f.item())
                train_loss_b.append(loss_b.item())
                
                if (iter_count + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}  loss_f:{3:.7f}  loss_b:{4:.7f}  loss_i:{5:.7f} loss_pred:{6:.7f} loss_res:{7:.7f}".format(iter_count + 1, epoch + 1, loss.item(), loss_f.item(), loss_b.item(), loss_pred.item(), loss_pred.item(), loss_pred.item()))
                    # print(torch.mean(impute_ratio.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    time_now = time.time()  

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss = self.vali(test_data, test_loader, criterion)
            # self.test("test")

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f}  Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            early_stopping(vali_loss, self.model, self.model_back, path, path_back)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            adjust_learning_rate(model_optim, epoch + 1, self.args)
            adjust_learning_rate(model_optim_back, epoch + 1, self.args)

        best_model_path = path + '/' + 'checkpoint.pth'
        best_model_path_back = path_back + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))
        self.model_back.load_state_dict(torch.load(best_model_path_back))

        return self.model, self.model_back

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')
        # if test:
        #     print('loading model')
        #     self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))

        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        self.model_back.eval()
        tp_len = self.args.pred_len - self.args.seq_len

        mask_tot = np.isnan(test_data.data_miss)
        true = test_data.data_x[mask_tot].copy()
        true = np.nan_to_num(true)

        pred_len = self.args.pred_len
        with torch.no_grad():
            data_miss = test_data.data_miss.copy()
            seq_len = self.args.seq_len
            i=seq_len
            while(i < len(data_miss)):
                mask_tmp = np.isnan(data_miss[i, :])
                if np.any(mask_tmp): 
                    mask = np.isnan(data_miss[i : i + pred_len, :])
                    series = data_miss[i - seq_len : i, :]
                    series = torch.tensor(series).float().to(self.device).unsqueeze(0)
                    series_mark = test_data.data_stamp[i - seq_len : i, :]
                    series_mark = torch.tensor(series_mark).float().to(self.device).unsqueeze(0)
                    
                    outputs = self.model(series, series_mark, None, None)[:, -pred_len:, :]
                    outputs = outputs.squeeze(0).cpu().numpy()
                    outputs[~mask] = (data_miss[i : i + pred_len, :])[~mask]

                    data_miss[i : i + pred_len, :] = outputs
                    i = i + pred_len
                else:
                    i = i + 1
            
            pred_front = data_miss.copy()
            
        with torch.no_grad():
            data_miss_b = test_data.data_miss
            data_miss_b = data_miss_b[::-1, :].copy()
            data_stamp = test_data.data_stamp
            data_stamp = data_stamp[::-1, :].copy()

            seq_len = self.args.seq_len
            i=seq_len
            while(i < len(data_miss_b)):
                mask_tmp = np.isnan(data_miss_b[i, :])
                if np.any(mask_tmp): 
                    mask_b = np.isnan(data_miss_b[i : i + pred_len, :])
                    series_b = data_miss_b[i - seq_len : i, :]
                    series_b = torch.tensor(series_b).float().to(self.device).unsqueeze(0)
                    series_mark_b = data_stamp[i - seq_len : i, :]
                    series_mark_b = torch.tensor(series_mark_b).float().to(self.device).unsqueeze(0)
                    outputs = self.model_back(series_b, series_mark_b, None, None)[:, -pred_len:, :]
                    outputs = outputs.squeeze(0).cpu().numpy()
                    outputs[~mask_b] = (data_miss_b[i : i + pred_len, :])[~mask_b]

                    data_miss_b[i : i + pred_len, :] = outputs
                    i=i + pred_len
                else:
                    i=i + 1
            
            pred_back = data_miss_b[::-1, :]
            pred_back = pred_back.copy()

        l=r=0
        pred = (pred_front + pred_back) / 2
        while(l < len(pred_front) and r < len(pred_front)):
            mask = np.isnan(test_data.data_miss[r, :])
            if np.any(mask): 
                r = r + 1
            else:
                if l == r:
                    l = l + 1
                    r = r + 1
                else:
                    ml = r - l
                    if ml != 1 :
                        weight_f = np.arange(ml - 1, -1, -1).astype(float) / (ml - 1)
                        weight_b = np.arange(ml).astype(float) / (ml - 1)
                        pred[l:r, :] = (pred_front[l:r, :] * weight_f.reshape(ml, 1) + pred_back[l:r, :] * weight_b.reshape(ml, 1))
                    r = r + 1
                    l = r

        impute_result = pred[mask_tot]
        data_impute = pred[mask_tot]
        pred = pred[mask_tot]
        pred_front = pred_front[mask_tot]
        pred_back = pred_back[mask_tot]

        mae_f = mean_absolute_error(pred_front, true)
        mae_b = mean_absolute_error(pred_back, true)
        mse_f = mean_squared_error(pred_front, true)
        mse_b = mean_squared_error(pred_back, true)
        mae = mean_absolute_error(impute_result, true)
        mse = mean_squared_error(impute_result, true)

        print("test")
        print("mse_f_b:", mse_f, mse_b)
        print("mae_f_b:", mae_f, mae_b)

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        print('mse:{}, mae:{}'.format(mse, mae))
        f = open("result_forecastImputation_3M_auto.txt", 'a')
        f.write(setting + "  \n")
        f.write("rate:{}, seed:{}, cnt:{} \n".format(self.args.mask_rate,self.args.seed,self.args.miss_len))
        f.write('mae:, mse:\n')
        f.write('{}\t{}'.format(mae, mse))
        f.write('\n')
        f.write('\n')
        f.close()
        return mse
