import torch
import torch.nn as nn
import numpy as np
from math import sqrt
from utils.masking import TriangularCausalMask, ProbMask
from reformer_pytorch import LSHSelfAttention
from einops import rearrange


# Code implementation from https://github.com/thuml/Flowformer
class FlowAttention(nn.Module):
    def __init__(self, attention_dropout=0.1,nolinear='None'):
        super(FlowAttention, self).__init__()
        self.dropout = nn.Dropout(attention_dropout)
        if nolinear=='ReLU':
            self.relu = nn.ReLU()
        elif nolinear=='GELU':
            self.relu = nn.GELU()
        elif nolinear=='ReLU6':
            self.relu = nn.ReLU6()
        else:
            self.relu = lambda x: x

    def kernel_method(self, x):
        return torch.sigmoid(x)

    def forward(self, queries, keys, values, attn_mask, tau=None, delta=None):
        queries = queries.transpose(1, 2)
        keys = keys.transpose(1, 2)
        values = values.transpose(1, 2)
        # kernel
        queries = self.kernel_method(queries)
        keys = self.kernel_method(keys)
        # incoming and outgoing
        normalizer_row = 1.0 / (torch.einsum("nhld,nhd->nhl", queries + 1e-6, keys.sum(dim=2) + 1e-6))
        normalizer_col = 1.0 / (torch.einsum("nhsd,nhd->nhs", keys + 1e-6, queries.sum(dim=2) + 1e-6))
        # reweighting
        normalizer_row_refine = (torch.einsum("nhld,nhd->nhl", queries + 1e-6, (keys * normalizer_col[:, :, :, None]).sum(dim=2) + 1e-6))
        normalizer_col_refine = (torch.einsum("nhsd,nhd->nhs", keys + 1e-6, (queries * normalizer_row[:, :, :, None]).sum(dim=2) + 1e-6))
        # competition and allocation
        normalizer_row_refine = torch.sigmoid(normalizer_row_refine * (float(queries.shape[2]) / float(keys.shape[2])))
        normalizer_col_refine = torch.softmax(normalizer_col_refine, dim=-1) * keys.shape[2]  # B h L vis
        # multiply
        kv = self.relu(keys.transpose(-2, -1) @ (values * normalizer_col_refine[:, :, :, None]))
        x = (((queries @ kv) * normalizer_row[:, :, :, None]) * normalizer_row_refine[:, :, :, None]).transpose(1,2).contiguous()
        return x, None


# Code implementation from https://github.com/shreyansh26/FlashAttention-PyTorch
class FlashAttention(nn.Module):
    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=False,nolinear='None'):
        super(FlashAttention, self).__init__()
        self.scale = scale
        self.nolinear = nolinear
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)
        if nolinear=='ReLU':
            self.relu = nn.ReLU()#nn.GELU()/nn.ReLU6(0.1)
        elif nolinear=='GELU':
            self.relu = nn.GELU()#nn.GELU()/nn.ReLU6(0.1)
        elif nolinear=='ReLU6':
            self.relu = nn.ReLU6()#nn.GELU()/nn.ReLU6(0.1)
        else:
            self.relu = lambda x: x

    def flash_attention_forward(self, Q, K, V, mask=None):
        BLOCK_SIZE = 32
        NEG_INF = -1e10  # -infinity
        EPSILON = 1e-10
        # mask = torch.randint(0, 2, (128, 8)).to(device='cuda')
        O = torch.zeros_like(Q, requires_grad=True)
        l = torch.zeros(Q.shape[:-1])[..., None]
        m = torch.ones(Q.shape[:-1])[..., None] * NEG_INF

        O = O.to(device='cuda')
        l = l.to(device='cuda')
        m = m.to(device='cuda')

        Q_BLOCK_SIZE = min(BLOCK_SIZE, Q.shape[-1])
        KV_BLOCK_SIZE = BLOCK_SIZE

        Q_BLOCKS = torch.split(Q, Q_BLOCK_SIZE, dim=2)
        K_BLOCKS = torch.split(K, KV_BLOCK_SIZE, dim=2)
        V_BLOCKS = torch.split(V, KV_BLOCK_SIZE, dim=2)
        if mask is not None:
            mask_BLOCKS = list(torch.split(mask, KV_BLOCK_SIZE, dim=1))

        Tr = len(Q_BLOCKS)
        Tc = len(K_BLOCKS)

        O_BLOCKS = list(torch.split(O, Q_BLOCK_SIZE, dim=2))
        l_BLOCKS = list(torch.split(l, Q_BLOCK_SIZE, dim=2))
        m_BLOCKS = list(torch.split(m, Q_BLOCK_SIZE, dim=2))

        for j in range(Tc):
            Kj = K_BLOCKS[j]
            Vj = V_BLOCKS[j]
            if mask is not None:
                maskj = mask_BLOCKS[j]

            for i in range(Tr):
                Qi = Q_BLOCKS[i]
                Oi = O_BLOCKS[i]
                li = l_BLOCKS[i]
                mi = m_BLOCKS[i]

                scale = 1 / np.sqrt(Q.shape[-1])
                Qi_scaled = Qi * scale

                S_ij = torch.einsum('... i d, ... j d -> ... i j', Qi_scaled, Kj)
                if mask is not None:
                    # Masking
                    maskj_temp = rearrange(maskj, 'b j -> b 1 1 j')
                    S_ij = torch.where(maskj_temp > 0, S_ij, NEG_INF)

                m_block_ij, _ = torch.max(S_ij, dim=-1, keepdims=True)
                P_ij = torch.exp(S_ij - m_block_ij)
                if mask is not None:
                    # Masking
                    P_ij = torch.where(maskj_temp > 0, P_ij, 0.)

                l_block_ij = torch.sum(P_ij, dim=-1, keepdims=True) + EPSILON

                P_ij_Vj = torch.einsum('... i j, ... j d -> ... i d', P_ij, Vj)

                mi_new = torch.maximum(m_block_ij, mi)
                li_new = torch.exp(mi - mi_new) * li + torch.exp(m_block_ij - mi_new) * l_block_ij

                O_BLOCKS[i] = self.relu((li / li_new) * torch.exp(mi - mi_new) * Oi + (
                        torch.exp(m_block_ij - mi_new) / li_new) * P_ij_Vj)
   
                l_BLOCKS[i] = li_new
                m_BLOCKS[i] = mi_new

        O = torch.cat(O_BLOCKS, dim=2)
        l = torch.cat(l_BLOCKS, dim=2)
        m = torch.cat(m_BLOCKS, dim=2)
        return O, l, m

    def forward(self, queries, keys, values, attn_mask, tau=None, delta=None):
        res = \
        self.flash_attention_forward(queries.permute(0, 2, 1, 3), keys.permute(0, 2, 1, 3), values.permute(0, 2, 1, 3),
                                     attn_mask)[0]
        return res.permute(0, 2, 1, 3).contiguous(), None


class FullAttention(nn.Module):
    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=False):
        super(FullAttention, self).__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, queries, keys, values, attn_mask, tau=None, delta=None):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        scale = self.scale or 1. / sqrt(E)

        scores = torch.einsum("blhe,bshe->bhls", queries, keys)

        if self.mask_flag:
            if attn_mask is None:
                attn_mask = TriangularCausalMask(B, L, device=queries.device)

            scores.masked_fill_(attn_mask.mask, -np.inf)

        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        V = torch.einsum("bhls,bshd->blhd", A, values)

        if self.output_attention:
            return (V.contiguous(), A)
        else:
            return (V.contiguous(), None)


# Code implementation from https://github.com/zhouhaoyi/Informer2020
class ProbAttention(nn.Module):
    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=False):
        super(ProbAttention, self).__init__()
        self.factor = factor
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def _prob_QK(self, Q, K, sample_k, n_top):  # n_top: c*ln(L_q)
        # Q [B, H, L, D]
        B, H, L_K, E = K.shape
        _, _, L_Q, _ = Q.shape

        # calculate the sampled Q_K
        K_expand = K.unsqueeze(-3).expand(B, H, L_Q, L_K, E)
        # real U = U_part(factor*ln(L_k))*L_q
        index_sample = torch.randint(L_K, (L_Q, sample_k))
        K_sample = K_expand[:, :, torch.arange(
            L_Q).unsqueeze(1), index_sample, :]
        Q_K_sample = torch.matmul(
            Q.unsqueeze(-2), K_sample.transpose(-2, -1)).squeeze()

        # find the Top_k query with sparisty measurement
        M = Q_K_sample.max(-1)[0] - torch.div(Q_K_sample.sum(-1), L_K)
        M_top = M.topk(n_top, sorted=False)[1]

        # use the reduced Q to calculate Q_K
        Q_reduce = Q[torch.arange(B)[:, None, None],
                   torch.arange(H)[None, :, None],
                   M_top, :]  # factor*ln(L_q)
        Q_K = torch.matmul(Q_reduce, K.transpose(-2, -1))  # factor*ln(L_q)*L_k

        return Q_K, M_top

    def _get_initial_context(self, V, L_Q):
        B, H, L_V, D = V.shape
        if not self.mask_flag:
            # V_sum = V.sum(dim=-2)
            V_sum = V.mean(dim=-2)
            contex = V_sum.unsqueeze(-2).expand(B, H,
                                                L_Q, V_sum.shape[-1]).clone()
        else:  # use mask
            # requires that L_Q == L_V, i.e. for self-attention only
            assert (L_Q == L_V)
            contex = V.cumsum(dim=-2)
        return contex

    def _update_context(self, context_in, V, scores, index, L_Q, attn_mask):
        B, H, L_V, D = V.shape

        if self.mask_flag:
            attn_mask = ProbMask(B, H, L_Q, index, scores, device=V.device)
            scores.masked_fill_(attn_mask.mask, -np.inf)

        attn = torch.softmax(scores, dim=-1)  # nn.Softmax(dim=-1)(scores)

        context_in[torch.arange(B)[:, None, None],
        torch.arange(H)[None, :, None],
        index, :] = torch.matmul(attn, V).type_as(context_in)
        if self.output_attention:
            attns = (torch.ones([B, H, L_V, L_V]) /
                     L_V).type_as(attn).to(attn.device)
            attns[torch.arange(B)[:, None, None], torch.arange(H)[
                                                  None, :, None], index, :] = attn
            return (context_in, attns)
        else:
            return (context_in, None)

    def forward(self, queries, keys, values, attn_mask, tau=None, delta=None):
        B, L_Q, H, D = queries.shape
        _, L_K, _, _ = keys.shape

        queries = queries.transpose(2, 1)
        keys = keys.transpose(2, 1)
        values = values.transpose(2, 1)

        U_part = self.factor * \
                 np.ceil(np.log(L_K)).astype('int').item()  # c*ln(L_k)
        u = self.factor * \
            np.ceil(np.log(L_Q)).astype('int').item()  # c*ln(L_q)

        U_part = U_part if U_part < L_K else L_K
        u = u if u < L_Q else L_Q

        scores_top, index = self._prob_QK(
            queries, keys, sample_k=U_part, n_top=u)

        # add scale factor
        scale = self.scale or 1. / sqrt(D)
        if scale is not None:
            scores_top = scores_top * scale
        # get the context
        context = self._get_initial_context(values, L_Q)
        # update the context with selected top_k queries
        context, attn = self._update_context(
            context, values, scores_top, index, L_Q, attn_mask)

        return context.contiguous(), attn


class AttentionLayer(nn.Module):
    def __init__(self, attention, d_model, n_heads, d_keys=None,
                 d_values=None):
        super(AttentionLayer, self).__init__()

        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)

        self.inner_attention = attention
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads

    def forward(self, queries, keys, values, attn_mask, tau=None, delta=None):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads

        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)

        out, attn = self.inner_attention(
            queries,
            keys,
            values,
            attn_mask,
            tau=tau,
            delta=delta
        )
        out = out.view(B, L, -1)

        return self.out_projection(out), attn


class ReformerLayer(nn.Module):
    def __init__(self, attention, d_model, n_heads, d_keys=None,
                 d_values=None, causal=False, bucket_size=4, n_hashes=4):
        super().__init__()
        self.bucket_size = bucket_size
        self.attn = LSHSelfAttention(
            dim=d_model,
            heads=n_heads,
            bucket_size=bucket_size,
            n_hashes=n_hashes,
            causal=causal
        )

    def fit_length(self, queries):
        # inside reformer: assert N % (bucket_size * 2) == 0
        B, N, C = queries.shape
        if N % (self.bucket_size * 2) == 0:
            return queries
        else:
            # fill the time series
            fill_len = (self.bucket_size * 2) - (N % (self.bucket_size * 2))
            return torch.cat([queries, torch.zeros([B, fill_len, C]).to(queries.device)], dim=1)

    def forward(self, queries, keys, values, attn_mask, tau, delta):
        # in Reformer: defalut queries=keys
        B, N, C = queries.shape
        queries = self.attn(self.fit_length(queries))[:, :N, :]
        return queries, None


class AttentionLayer_zhudiancnn(nn.Module):
    def __init__(self, attention, d_model, n_heads, d_keys=None,
                 d_values=None,out_channels=16):
        super(AttentionLayer_zhudiancnn, self).__init__()
        self.out_channels=out_channels

        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)

        self.inner_attention = attention
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * out_channels, d_model)#修改了输入维度
        self.n_heads = n_heads

        #------逐点卷积实现通道间的信息交互-------
        self.conv_zhudian = nn.Conv2d(in_channels=n_heads, out_channels=out_channels, kernel_size=1, 
                                      stride=1, padding=0,dilation=1, groups=1)

    def forward(self, queries, keys, values, attn_mask, tau=None, delta=None):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads

        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)

        out, attn = self.inner_attention(
            queries,
            keys,
            values,
            attn_mask,
            tau=tau,
            delta=delta
        )

        out = self.conv_zhudian(out.permute(0, 2, 1, 3)).permute(0, 2, 1, 3)
        out = out.contiguous().view(B, L, -1)

        return self.out_projection(out), attn

'''-------------一、SE模块-----------------------------'''
#全局平均池化+1*1卷积核+ReLu+1*1卷积核+Sigmoid
class SE_Block(nn.Module):
    def __init__(self, inchannel, ratio=16):
        super(SE_Block, self).__init__()
        # 全局平均池化(Fsq操作)
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        # 两个全连接层(Fex操作)
        self.fc = nn.Sequential(
            nn.Linear(inchannel, int(inchannel // ratio), bias=False),  # 从 c -> c/r
            nn.ReLU(),
            nn.Linear(int(inchannel // ratio), inchannel, bias=False),  # 从 c/r -> c
            nn.Sigmoid()
        )
 
    def forward(self, x):
            # 读取批数据图片数量及通道数
            b, c, h, w = x.size()
            # Fsq操作：经池化后输出b*c的矩阵
            y = self.gap(x).view(b, c)
            # Fex操作：经全连接层输出（b，c，1，1）矩阵
            y = self.fc(y).view(b, c, 1, 1)
            # Fscale操作：将得到的权重乘以原来的特征图x
            return x * y.expand_as(x)

class AttentionLayer_SE(nn.Module):
    def __init__(self, attention, d_model, n_heads, d_keys=None,
                 d_values=None,ratio=16):
        super(AttentionLayer_SE, self).__init__()
        self.ratio=ratio

        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)

        self.inner_attention = attention
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)#修改了输入维度
        self.n_heads = n_heads

        #------逐点卷积实现通道间的信息交互-------
        self.conv_SE = SE_Block(n_heads,ratio=ratio)

    def forward(self, queries, keys, values, attn_mask, tau=None, delta=None):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads

        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)

        out, attn = self.inner_attention(
            queries,
            keys,
            values,
            attn_mask,
            tau=tau,
            delta=delta
        )

        out = self.conv_SE(out.permute(0, 2, 1, 3)).permute(0, 2, 1, 3)
        out = out.contiguous().view(B, L, -1)

        return self.out_projection(out), attn


#=============使用注意力机制实现多头信息交互==============
class AttentionLayer_SA(nn.Module):
    def __init__(self, attention, d_model, n_heads, d_keys=None,
                 d_values=None):
        super(AttentionLayer_SA, self).__init__()

        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)

        self.inner_attention = attention
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)#修改了输入维度
        self.n_heads = n_heads


    def forward(self, queries, keys, values, attn_mask, tau=None, delta=None):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads

        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)

        out, attn = self.inner_attention(
            queries,
            keys,
            values,
            attn_mask,
            tau=tau,
            delta=delta
        )

        out = self.get_interaction(out)
        out = out.contiguous().view(B, L, -1)

        return self.out_projection(out), attn

    def get_interaction(slef,out):
        """ 使用注意力机制实现多头信息交互 """
        B,L,H,HE=out.shape#求数据形状
        out_1=out.permute(0, 2, 1, 3).contiguous().view(B, H, -1)#转置
        attention_scores = torch.matmul(out_1, out_1.transpose(-1, -2))#头与头之间的注意力分数
        attention_scores = torch.softmax(attention_scores, dim=-1)#分数归一化
        out_2 = torch.matmul(attention_scores, out_1)#加权求和
        out_2 = out_2.view([B,H,L,-1]).permute(0, 2, 1, 3)
        return out_2

#================将多种注意力头交互的代码合并在一起===============
class AttentionLayer_SE_SA_zhudian(nn.Module):
    def __init__(self, attention, d_model, n_heads, d_keys=None,
                 d_values=None,ratio=16,out_channels=16,mode='SE'):
        super(AttentionLayer_SE_SA_zhudian, self).__init__()
        self.mode=mode
        self.ratio=ratio
        self.out_channels=out_channels

        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)

        self.inner_attention = attention
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        
        self.n_heads = n_heads

        if mode=='zhudian':
            #------逐点卷积实现通道间的信息交互-------
            self.inter_layer = nn.Conv2d(in_channels=n_heads, out_channels=out_channels, kernel_size=1, 
                                         stride=1, padding=0,dilation=1, groups=1)
            self.out_projection = nn.Linear(d_values * out_channels, d_model)#修改了输入维度
            
        elif mode=='SE':
             #------SE实现通道间的信息交互-------
            self.inter_layer = SE_Block(n_heads,ratio=ratio)  
            self.out_projection = nn.Linear(d_values * n_heads, d_model)#修改了输入维度
        elif mode=='SA':
            self.out_projection = nn.Linear(d_values * n_heads, d_model)#修改了输入维度

    def forward(self, queries, keys, values, attn_mask, tau=None, delta=None):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads

        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)

        out, attn = self.inner_attention(
            queries,
            keys,
            values,
            attn_mask,
            tau=tau,
            delta=delta
        )

        if self.mode=='zhudian':
            out = self.inter_layer(out.permute(0, 2, 1, 3)).permute(0, 2, 1, 3)
        elif self.mode=='SE':
            out = self.inter_layer(out.permute(0, 2, 1, 3)).permute(0, 2, 1, 3)
        elif self.mode=='SA':
            out = self.get_interaction(out)
        out = out.contiguous().view(B, L, -1)

        return self.out_projection(out), attn

    def get_interaction(slef,out):
        """ 使用注意力机制实现多头信息交互 """
        B,L,H,HE=out.shape#求数据形状
        out_1=out.permute(0, 2, 1, 3).contiguous().view(B, H, -1)#转置
        attention_scores = torch.matmul(out_1, out_1.transpose(-1, -2))#头与头之间的注意力分数
        attention_scores = torch.softmax(attention_scores, dim=-1)#分数归一化
        out_2 = torch.matmul(attention_scores, out_1)#加权求和
        out_2 = out_2.view([B,H,L,-1]).permute(0, 2, 1, 3)
        return out_2





class LinearAttention(nn.Module):
    def __init__(self, dim_q, dim_k, dim_v, dim_out=None):
        super(LinearAttention, self).__init__()
        self.dim_q = dim_q
        self.dim_k = dim_k
        self.dim_v = dim_v
        self.dim_out = dim_out or dim_v

        # 假设使用简单的线性变换来映射查询、键和值
        self.query_proj = nn.Linear(dim_q, dim_k)
        self.key_proj = nn.Linear(dim_k, dim_k)
        self.value_proj = nn.Linear(dim_v, dim_v) if dim_v != dim_k else None
        self.out_proj = nn.Linear(dim_v, dim_out) if dim_out != dim_v else None

        # 对于线性注意力，我们可能需要一个额外的特征映射或核函数
        # 这里我们简化处理，不使用额外的核函数

    def forward(self, query, key, value, mask=None):
        # query, key, value: [batch_size, seq_len, dim]

        # 映射查询和键
        Q = self.query_proj(query)  # [batch_size, seq_len_q, dim_k]
        K = self.key_proj(key)  # [batch_size, seq_len_k, dim_k]

        # 如果值向量需要映射，则进行映射
        if self.value_proj:
            V = self.value_proj(value)  # [batch_size, seq_len_v, dim_v]
        else:
            V = value

        # 计算注意力得分（这里使用点积作为相似度度量）
        # 注意：为了简化，我们没有使用softmax进行归一化，这在实际应用中可能是必要的
        # 也没有使用mask来处理序列中的填充部分
        attn_scores = torch.bmm(Q, K.transpose(1, 2))  # [batch_size, seq_len_q, seq_len_k]

        # 聚合值向量（这里直接使用注意力得分进行加权求和）
        # 在实际应用中，可能需要使用softmax或其他归一化方法来稳定训练
        attn_output = torch.bmm(attn_scores, V)  # [batch_size, seq_len_q, dim_v]

        # 如果需要，对输出进行映射
        if self.out_proj:
            attn_output = self.out_proj(attn_output)  # [batch_size, seq_len_q, dim_out]

        return attn_output


class Attention(nn.Module):
    def __init__(self, hidden_size,attention_dropout = 0.1, mask_flag=True, factor=5, scale=None,):
        super(Attention, self).__init__()
        self.q_linear = nn.Linear(hidden_size, hidden_size)
        self.k_linear = nn.Linear(hidden_size, hidden_size)
        self.v_linear = nn.Linear(hidden_size, hidden_size)
        self.attention_score = nn.Linear(hidden_size, 1)
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, lstm_output):
        q = self.q_linear(lstm_output)
        k = self.k_linear(lstm_output)
        v = self.v_linear(lstm_output)

        # 计算注意力得分
        scores = torch.tanh(q + k)
        scores = self.attention_score(scores)
        attention_weights = torch.softmax(scores, dim=-1)


        # 应用注意力权重
        attended_output = torch.sum(attention_weights * v, dim=-1)
        A = self.dropout(attended_output)
        V = torch.einsum("bhls,bshd->blhd", A, v)

        return  (V.contiguous(), None)
