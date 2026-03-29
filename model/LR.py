import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Embed import DataEmbedding
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.pred_len = configs.pred_len
        self.hidden_size = configs.hidden_size
        self.d_model = configs.d_model
        
        if configs.channel_independence:
            self.enc_in = 1
            self.dec_in = 1
            self.c_out = 1
        else:
            self.enc_in = configs.enc_in
            self.dec_in = configs.dec_in
            self.c_out = configs.c_out
        self.linear1 = nn.Linear(self.d_model,self.hidden_size)
        self.linear2 = nn.Linear(self.hidden_size,self.hidden_size)
   
        self.linear3 = nn.Linear(self.hidden_size,self.c_out)
        self.enc_embedding = DataEmbedding(
            self.enc_in, configs.d_model, configs.embed, configs.freq, configs.dropout
        )
    
    def long_term_forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):

        # 数据嵌入
     
        x_enc = torch.cat([x_enc, x_dec[:, -self.pred_len:, :]], dim=1)
        if x_mark_enc is not None:
            x_mark_enc = torch.cat([x_mark_enc, x_mark_dec[:, -self.pred_len:, :]], dim=1)
       
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
      
        x_enc=self.linear1(enc_out)
        x_enc = self.linear2(x_enc)
        out = self.linear3(x_enc)
      
        return out
    
    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        last_hidden_state = self.long_term_forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
        return last_hidden_state[:,-self.pred_len:,:]
