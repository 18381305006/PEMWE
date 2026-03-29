from utils.tools import EarlyStopping, adjust_learning_rate, visual
from utils.metrics import metric
import torch
import torch.nn as nn
from torch import optim
import os
import time
import warnings
import numpy as np
from dataload_func import data_provider
from model import Transformer, Informer, Reformer, Flowformer, Flashformer, iTransformer, iInformer, iReformer, iFlowformer, iFlashformer,LSTM,LR,CNN,XGBoost, RandomForest, iFlashformer_zhudian,iFlashformer_SE,iFlashformer_SA,iFlashformer_SE_SA_zhudian,iFlowformer_extend
import tqdm
import pandas as pd
class Exp_Basic(object):
    def __init__(self, args):
        self.args = args
        self.model_dict = {
            'Transformer': Transformer,
            'Informer': Informer,
            'Reformer': Reformer,
            'Flowformer': Flowformer,
            'Flashformer': Flashformer,
            'iTransformer': iTransformer,
            'iInformer': iInformer,
            'iReformer': iReformer,
            'iFlowformer': iFlowformer,
            'iFlashformer': iFlashformer,
            'LSTM'      : LSTM,
            'LR':LR,
            "CNN":CNN,
            "XGBoost":XGBoost,
            "RandomForest":RandomForest,
            "iFlashformer_zhudian":iFlashformer_zhudian,
            'iFlashformer_SE':iFlashformer_SE,
            'iFlashformer_SA':iFlashformer_SA,
            'iFlashformer_SE_SA_zhudian':iFlashformer_SE_SA_zhudian,
            'iFlowformer_extend':iFlowformer_extend,
            'iFlowformer_zhudian':iFlowformer_extend,
            'iFlowformer_SE':iFlowformer_extend,
            'iFlowformer_SA':iFlowformer_extend,
        }
        self.device = self._acquire_device()
        self.model = self._build_model().to(self.device)

    def _build_model(self):
        raise NotImplementedError
        return None

    def _acquire_device(self):
        device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # device='cpu'
        return device

    def _get_data(self):
        pass

    def vali(self):
        pass

    def train(self):
        pass

    def test(self):
        pass
        
class Exp_Long_Term_Forecast(Exp_Basic):
    def __init__(self, args):
        super(Exp_Long_Term_Forecast, self).__init__(args)

    def _build_model(self):
        model = self.model_dict[self.args.model].Model(self.args).float()
        return model

    def _get_data(self, flag,batch_size_=None):
        data_set, data_loader = data_provider(self.args, flag,batch_size_=batch_size_)
        return data_set, data_loader

    def _select_optimizer(self):
        # print(list(self.model.parameters()))
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim
        
    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion

    def train(self, setting,data_num=''):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test',batch_size_=64)

        path = os.path.join(f'./results/results{data_num}/', setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        self.epochs_loss,self.train_results_dict,self.vali_results_dict,self.test_results_dict={},{},{},{}
        iter_count=0
        train_loss,preds,trues = [],[],[]
        stop_flag=False
        epoch_time = time.time()
        for epoch in range(self.args.train_epochs):
            self.model.train()
            # for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(tqdm.tqdm(train_loader)):
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                if 'PEMS' in self.args.data or 'Solar' in self.args.data:
                    batch_x_mark = None
                    batch_y_mark = None
                else:
                    batch_x_mark = batch_x_mark.float().to(self.device)
                    batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                        
                        # f_dim = -1 if self.args.features == 'MS' else 0
                        f_dim = -1
                        outputs = outputs[:, -self.args.pred_len:, f_dim:]
                        batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                        loss = criterion(outputs, batch_y)
                        train_loss.append(loss.item())
                else:
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                    f_dim = -1 if self.args.features == 'MS' else 0
                    outputs = outputs[:, -self.args.pred_len:, f_dim:]
                    batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                    loss = criterion(outputs, batch_y)
                    train_loss.append(loss.item())

                pred = outputs.detach().cpu()
                true = batch_y.detach().cpu()
                
                preds.append(pred)
                trues.append(true)

                # if (i + 1) % 100 == 0:
                #     print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                #     speed = (time.time() - time_now) / iter_count
                #     left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                #     print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                #     iter_count = 0
                #     time_now = time.time()

                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()

                #达到一定的迭代次数就计算评估指标，并保存最好的模型
                item_num=len(train_loader)//5#一个epoch评估5次，保留最优的那次模型
                if iter_count % item_num==0 and iter_count>0:
                    train_results = metric(np.concatenate(preds), np.concatenate(trues))#计算评估指标
                    train_loss = np.average(train_loss)#计算损失值
                    vali_loss,vali_results = self.vali(vali_data, vali_loader, criterion)
                    test_loss,test_results = self.vali(test_data, test_loader, criterion)
            
                    self.epochs_loss[len(self.epochs_loss)+1]=[train_loss,vali_loss,test_loss]
                    self.vali_results_dict[len(self.vali_results_dict)+1]=vali_results
                    self.test_results_dict[len(self.test_results_dict)+1]=test_results
                    self.train_results_dict[len(self.train_results_dict)+1]=train_results

                    print("epoch: {} Item: {} cost time: {}".format(epoch,iter_count // item_num, time.time() - epoch_time))
                    print("Train Loss: {0:.7f} Vali Loss: {1:.7f} Test Loss: {2:.7f}".format(train_loss, vali_loss, test_loss))
                    print("Test metrics: ",test_results)
                    train_loss,preds,trues = [],[],[]
                    
                    early_stopping(vali_loss, self.model, path)
                    if early_stopping.early_stop:
                        stop_flag=True
                        print("Early stopping")
                        break
            
            adjust_learning_rate(model_optim, epoch + 1, self.args)
            if stop_flag:
                break

        #======整理结果======
        loss_df=pd.DataFrame(self.epochs_loss,index=['train loss','valid loss','test loss']).T
        valid_result_df=pd.DataFrame(self.vali_results_dict,index=['mae', 'mse', 'rmse', 'mape', 'mspe','speed']).T
        test_result_df=pd.DataFrame(self.test_results_dict,index=['mae', 'mse', 'rmse', 'mape', 'mspe','speed']).T
        train_result_df=pd.DataFrame(self.train_results_dict,index=['mae', 'mse', 'rmse', 'mape', 'mspe']).T
        loss_df.to_csv(path+'/loss_df.csv',encoding='utf_8_sig')
        valid_result_df.to_csv(path+'/valid_result_df.csv',encoding='utf_8_sig')
        test_result_df.to_csv(path+'/test_result_df.csv',encoding='utf_8_sig')
        train_result_df.to_csv(path+'/train_result_df.csv',encoding='utf_8_sig')
        
        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))
        

        return self.model

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        preds = []
        trues = []
        self.model.eval()

        time_list=[]
        start_time=time.time()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(tqdm.tqdm(vali_loader)):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                if 'PEMS' in self.args.data or 'Solar' in self.args.data:
                    batch_x_mark = None
                    batch_y_mark = None
                else:
                    batch_x_mark = batch_x_mark.float().to(self.device)
                    batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                # f_dim = -1 if self.args.features == 'MS' else 0
                f_dim = -1
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                pred = outputs.detach().cpu()
                true = batch_y.detach().cpu()

                loss = criterion(pred, true)

                total_loss.append(loss)
                preds.append(pred)
                trues.append(true)

        end_time=time.time()
        total_loss = np.average(total_loss)
        preds=np.concatenate(preds)
        trues=np.concatenate(trues)
        mae, mse, rmse, mape, mspe = metric(preds, trues)#评估
        self.model.train()
        return total_loss,[mae, mse, rmse, mape, mspe,(start_time-end_time)/len(vali_loader)]

    def test(self, setting, test=0,batch_size_=None,root_path='',data_num=''):
        test_data, test_loader = self._get_data(flag='test',batch_size_=batch_size_)
        if test:
            print('loading model')
            print()
            self.model.load_state_dict(torch.load(os.path.join(f'./results/results{data_num}/' + setting, 'checkpoint.pth')))

        preds = []
        trues = []
        folder_path = f'./results/test_results{data_num}/' + setting + '/'
        if not os.path.exists(f'./results/test_results{data_num}/'):
            os.makedirs(f'./results/test_results{data_num}/')

        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(tqdm.tqdm(test_loader)):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                if 'PEMS' in self.args.data or 'Solar' in self.args.data:
                    batch_x_mark = None
                    batch_y_mark = None
                else:
                    batch_x_mark = batch_x_mark.float().to(self.device)
                    batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]

                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                # f_dim = -1 if self.args.features == 'MS' else 0
                f_dim = -1
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()
                if test_data.scale and self.args.inverse:
                    shape = outputs.shape
                    outputs = test_data.inverse_transform(outputs.squeeze(0)).reshape(shape)
                    batch_y = test_data.inverse_transform(batch_y.squeeze(0)).reshape(shape)

                pred = outputs
                true = batch_y

                preds.append(pred)
                trues.append(true)

                num_=len(test_loader)//20#保存20个样本的预测结果
                if i % num_ == 0:
                    input = batch_x.detach().cpu().numpy()
                    if test_data.scale and self.args.inverse:
                        shape = input.shape
                        input = test_data.inverse_transform(input.squeeze(0)).reshape(shape)
                    gt = np.concatenate((input[0, :, -1], true[0, :, -1]), axis=0)
                    pd = np.concatenate((input[0, :, -1], pred[0, :, -1]), axis=0)
                    visual(gt, pd, os.path.join(folder_path, str(i) + '.pdf'))

        preds = np.array(preds)
        trues = np.array(trues)
        print('test shape:', preds.shape, trues.shape)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        print('test shape:', preds.shape, trues.shape)

        # result save
        if not os.path.exists(f'./results/results{data_num}/'):
            os.makedirs(f'./results/results{data_num}/')
            
        folder_path = f'./results/results{data_num}/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        mae, mse, rmse, mape, mspe = metric(preds, trues)
        print('mse:{}, mae:{}'.format(mse, mae))
        f = open(f"result{data_num}_long_term_forecast.txt", 'a')
        f.write(setting + "  \n")
        f.write('mse:{}, mae:{}'.format(mse, mae))
        f.write('\n')
        f.write('\n')
        f.close()

        np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe]))
        np.save(folder_path + 'pred.npy', preds)
        np.save(folder_path + 'true.npy', trues)

        return