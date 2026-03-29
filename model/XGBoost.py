import torch
import torch.nn as nn
import torch.nn.functional as F
import xgboost as xgb
from xgboost import XGBRegressor
import numpy as np
import pandas as pd
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()

        self.colsample_bytree = configs.colsample_bytree
        self.max_depth = configs.max_depth
        self.n_estimators = configs.n_estimators
        self.d_model = configs.d_model
        self.hidden_size = configs.hidden_size
        self.c_out = configs.c_out
        self.model = XGBRegressor(random_state = 42, max_depth = self.max_depth, n_estimators = self.n_estimators,
                                  colsample_bytree=self.colsample_bytree,
                                  enable_categorical = True)
        self.pred_len = configs.pred_len
        self.fc1 = nn.Linear(1,self.hidden_size)
        self.fc2 = nn.Linear(self.hidden_size,self.hidden_size)
        self.fc3 = nn.Linear(self.hidden_size,self.c_out)


    def long_term_forecast(self,x_enc, x_mark_enc, x_dec, x_mark_dec):
        x_enc = torch.cat([x_enc, x_dec[:, -self.pred_len:, :]], dim=1)
        if x_mark_enc is not None:
            x_mark_enc = torch.cat(
                [x_mark_enc, x_mark_dec[:, -self.pred_len:, :]], dim=1)

        x_enc_flattened = x_enc.reshape(x_enc.shape[0], -1)
       
        x_dec_flattened = x_dec.reshape(x_dec.shape[0], -1)
        x_mark_dec_flattened = x_mark_dec.reshape(x_mark_dec.shape[0],-1)

        Xtrain = x_enc_flattened.cpu().numpy()
        ytrain = x_dec_flattened.cpu().numpy()
        dtest = x_mark_dec_flattened.cpu().numpy()
        Xtrain = np.mean(Xtrain,axis=1)
    
        Xtrain = Xtrain.reshape(-1,1)
    
        ytrain = np.mean(ytrain,axis =1 )
        ytrain = ytrain.reshape(-1,1)
        dtest = np.mean(dtest,axis =1)
        dtest = dtest.reshape(-1,1)
        self.model.fit(Xtrain,ytrain)

        # Predict using the trained model
        pred = self.model.predict(dtest)
        pred = torch.tensor(pred, dtype=torch.float32,device = device,requires_grad = True)
        extended_pred = pred.unsqueeze(1).repeat(1, 240)

        # 添加一个通道维度
        final_pred = extended_pred.unsqueeze(-1)

        # 现在 final_pred 的形状应该是 [128, 240, 1]
        pred = self.fc1(final_pred)
        pred = self.fc2(pred)
        pred = self.fc3(pred)
    
        return pred

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        last_hidden_state = self.long_term_forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
        return last_hidden_state[:,-self.pred_len:,:]

