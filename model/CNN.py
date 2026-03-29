import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Embed import DataEmbedding


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.pred_len = configs.pred_len
        self.d_model = configs.d_model
        self.c_out = configs.c_out
        self.hidden_size = configs.hidden_size
        self.num_layers = configs.num_layers
        if configs.channel_independence:
            self.enc_in = 1
            self.dec_in = 1
            self.c_out = 1
        else:
            self.enc_in = configs.enc_in
            self.dec_in = configs.dec_in
            self.c_out = configs.c_out
        
        self.enc_embedding = DataEmbedding(
            self.enc_in, configs.d_model, configs.embed, configs.freq, configs.dropout
        )
        self.conv1 = nn.Conv1d(in_channels=240, out_channels=240, kernel_size=1, padding=0)
       
        self.fc1 = nn.Linear(128,self.hidden_size)
        self.fc2 = nn.Linear(self.hidden_size,self.c_out)
        self.relu = nn.ReLU()
        self.lstm = nn.LSTM(configs.d_model, configs.hidden_size, configs.num_layers, batch_first=True)
        self.dropout = nn.Dropout(0.2)

      
    
    
    def long_term_forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        x_enc = torch.cat([x_enc, x_dec[:, -self.pred_len:, :]], dim=1)
        if x_mark_enc is not None:
            x_mark_enc = torch.cat([x_mark_enc, x_mark_dec[:, -self.pred_len:, :]], dim=1)
       
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
     
        
        x = F.relu(self.conv1(enc_out))


    
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)

        lstm_out, _ = self.lstm(x)

        out = self.fc1(lstm_out)
        out = self.fc2(out)
       
        return out
    
    def forward(self,x_enc,x_mark_enc,x_dec,x_mark_dec,mask=None):
        last_hidden_state = self.long_term_forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
        return last_hidden_state[:, -self.pred_len:, :]

