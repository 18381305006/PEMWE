import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.SelfAttention_Family import FlowAttention, AttentionLayer
from layers.Embed import DataEmbedding

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.pred_len = configs.pred_len
        self.output_attention = configs.output_attention
        self.num_layers = configs.num_layers
        self.hidden_size = configs.hidden_size

        if configs.channel_independence:
            self.enc_in = 1
            self.dec_in = 1
            self.c_out = 1
        else:
            self.enc_in = configs.enc_in
            self.dec_in = configs.dec_in
            self.c_out = configs.c_out

        # 数据嵌入
        self.enc_embedding = DataEmbedding(
            self.enc_in, configs.d_model, configs.embed, configs.freq, configs.dropout,time_type=configs.time_type
        )

        # LSTM
        self.lstm = nn.LSTM(configs.d_model, configs.hidden_size, configs.num_layers, batch_first=True)

        # 替换为 FlowAttention 注意力层
        self.attention = AttentionLayer(
            FlowAttention(attention_dropout=configs.dropout),  # 修正参数
            configs.hidden_size,
            configs.hidden_size,
        )

        # 全连接层
        self.fc = nn.Linear(configs.hidden_size, configs.output_size)

    def generate_attention_mask(self, seq_length, batch_size):
        """
        根据需求生成适当的注意力掩码。
        示例：全为 1 的掩码（不阻挡任何注意力）。
        """
        return torch.ones(batch_size, seq_length, seq_length).to(device)

    def long_term_forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        # 合并编码器和解码器的输入
        x_enc = torch.cat([x_enc, x_dec[:, -self.pred_len:, :]], dim=1)
        if x_mark_enc is not None:
            x_mark_enc = torch.cat([x_mark_enc, x_mark_dec[:, -self.pred_len:, :]], dim=1)

        # 数据嵌入
        enc_out = self.enc_embedding(x_enc, x_mark_enc)  # [B, T, C]

        # 初始化 LSTM 隐状态
        h0 = torch.zeros(self.num_layers, enc_out.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, enc_out.size(0), self.hidden_size).to(device)

        # LSTM 前向传播
        lstm_out, _ = self.lstm(enc_out, (h0, c0))

        # 生成注意力掩码
        attn_mask = self.generate_attention_mask(lstm_out.size(1), lstm_out.size(0))

        # 应用注意力层
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out, attn_mask=attn_mask)

        # 全连接层
        out = self.fc(attn_out)
        # print(out.shape)
        # print(type(out))
        return out

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        # 调用 long_term_forecast 获取预测结果
        last_hidden_state = self.long_term_forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
        return last_hidden_state[:, -self.pred_len:, :]


'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.SelfAttention_Family import FullAttention, AttentionLayer, LinearAttention, Attention
from layers.Embed import DataEmbedding

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.pred_len = configs.pred_len
        self.output_attention = configs.output_attention
        self.num_layers = configs.num_layers
        self.hidden_size = configs.hidden_size

        if configs.channel_independence:
            self.enc_in = 1
            self.dec_in = 1
            self.c_out = 1
        else:
            self.enc_in = configs.enc_in
            self.dec_in = configs.dec_in
            self.c_out = configs.c_out

        self.enc_embedding = DataEmbedding(self.enc_in, configs.d_model, configs.embed, configs.freq,
                                           configs.dropout)

        self.lstm = nn.LSTM(configs.d_model, configs.hidden_size, configs.num_layers, batch_first=True)
        self.attention = AttentionLayer(
            FullAttention(mask_flag=False, attention_dropout=configs.dropout,
                          output_attention=configs.output_attention),
            configs.hidden_size, configs.hidden_size
        )
        self.fc = nn.Linear(configs.hidden_size, configs.output_size)

    def generate_attention_mask(self, seq_length, batch_size):
        # 根据需求生成适当的注意力掩码
        return torch.ones(batch_size, seq_length, seq_length).to(device)

    def long_term_forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        x_enc = torch.cat([x_enc, x_dec[:, -self.pred_len:, :]], dim=1)
        if x_mark_enc is not None:
            x_mark_enc = torch.cat(
                [x_mark_enc, x_mark_dec[:, -self.pred_len:, :]], dim=1)

        enc_out = self.enc_embedding(x_enc, x_mark_enc)  # [B,T,C]

        h0 = torch.zeros(self.num_layers, enc_out.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, enc_out.size(0), self.hidden_size).to(device)

        lstm_out, _ = self.lstm(enc_out, (h0, c0))
        # 生成注意力掩码
        attn_mask = self.generate_attention_mask(lstm_out.size(1), lstm_out.size(0))

        # 应用注意力层
        attn_out, _ = self.attention(
            lstm_out, lstm_out, lstm_out, attn_mask=attn_mask
        )
        out = self.fc(attn_out)

        return out

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        last_hidden_state = self.long_term_forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
        return last_hidden_state[:, -self.pred_len:, :]


'''
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.SelfAttention_Family import FullAttention,AttentionLayer,LinearAttention,Attention
from layers.Embed import DataEmbedding
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.pred_len = configs.pred_len
        self.output_attention = configs.output_attention
        self.num_layers = configs.num_layers
        self.hidden_size = configs.hidden_size

        if configs.channel_independence:
            self.enc_in = 1
            self.dec_in = 1
            self.c_out = 1
        else:
            self.enc_in = configs.enc_in
            self.dec_in = configs.dec_in
            self.c_out = configs.c_out
        self.enc_embedding = DataEmbedding(self.enc_in, configs.d_model, configs.embed, configs.freq,
                                           configs.dropout)
        
        self.lstm = nn.LSTM(configs.d_model, configs.hidden_size, configs.num_layers, batch_first=True)
        self.fc   = nn.Linear(configs.hidden_size,configs.output_size)
        self.attention = Attention(configs.hidden_size)
    def long_term_forecast(self,x_enc, x_mark_enc, x_dec, x_mark_dec):
        x_enc = torch.cat([x_enc, x_dec[:, -self.pred_len:, :]], dim=1)
        if x_mark_enc is not None:
            x_mark_enc = torch.cat(
                [x_mark_enc, x_mark_dec[:, -self.pred_len:, :]], dim=1)

        enc_out = self.enc_embedding(x_enc, x_mark_enc)  # [B,T,C]
        
        h0 = torch.zeros(self.num_layers, enc_out.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, enc_out.size(0), self.hidden_size).to(device)

        lstm_out,_ = self.lstm(enc_out,(h0,c0))
        
        out = self.fc(lstm_out)
        
        return out

    def forward(self,x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):

        last_hidden_state = self.long_term_forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
        return last_hidden_state[:, -self.pred_len:, :]

'''
