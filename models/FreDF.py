import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Embed import DataEmbedding
import torch.fft


class ConcatenationWithLearnableCoefficient(nn.Module):
    def __init__(self, T):
        super(ConcatenationWithLearnableCoefficient, self).__init__()
        self.weights = nn.Parameter(torch.randn(T))

    def forward(self, input):
        self.weights.data = F.softmax(self.weights, dim=0).data
        output = torch.einsum('btlc, t -> blc', input, self.weights)
        return output


class FFT_for_Decomp(nn.Module):
    def __init__(self, four_l, d_model):
        super(FFT_for_Decomp, self).__init__()
        self.linear_layers = nn.ModuleList(
            [nn.Linear(d_model, d_model, dtype=torch.complex64, bias=False) for _ in range(four_l)])
        self.concat = ConcatenationWithLearnableCoefficient(four_l)

    def forward(self, x):
        # [B, T, C]
        xf = torch.fft.rfft(x, dim=1, norm="ortho")  # [B,T/2+1,C]
        B, t, C = xf.shape

        xf = xf.unsqueeze(1)
        xf = xf.repeat(1, xf.shape[2], 1, 1)

        mask = torch.eye(t).byte().to(xf.device)  # 生成对角矩阵，并在最后一个维度上添加一个维度
        fre = torch.einsum('btlc, tl -> btlc', xf, mask)

        fre_tmp = fre.clone()
        for i in range(t):
            fre_tmp[:, i, i, :] = (self.linear_layers[i](fre[:, i, i, :]))
        fre = fre_tmp

        enc_out = torch.fft.irfft(fre, dim=2, norm="ortho")
        enc_out = self.concat(enc_out)
        return enc_out


class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.output_attention = configs.output_attention
        self.is_embed = configs.is_embed

        self.embedding = DataEmbedding(configs.c_out, configs.d_model, configs.embed, configs.freq,
                                       configs.dropout)
        self.projection = nn.Sequential(
            nn.Linear(configs.d_model, configs.c_out),
            # nn.ReLU()
        )
        self.linear1 = nn.Sequential(
            nn.Linear(self.seq_len, self.seq_len),
            nn.ReLU(),
            nn.Dropout(configs.dropout)
        )
        self.linear2 = nn.Sequential(
            nn.Linear(self.pred_len, self.pred_len),
            nn.ReLU(),
            nn.Dropout(configs.dropout)
        )
        self.layer = configs.e_layers
        self.dropout = nn.Dropout(configs.dropout)
        self.fft = FFT_for_Decomp((self.pred_len + self.seq_len) // 2 + 1, configs.d_model)

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        # Normalization from Non-stationary Transformer
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev
        x_enc = self.linear1(x_enc.permute(0, 2, 1)).permute(0, 2, 1)
        zeros = torch.zeros(x_enc.shape[0], self.pred_len, x_enc.shape[2], device=x_enc.device)
        x_enc = torch.cat([x_enc, zeros], dim=1)
        if self.is_embed:
            x_enc = self.embedding(x_enc, torch.cat((x_mark_enc, x_mark_dec[:, -self.pred_len:, :]), dim=1))
        x_enc = self.dropout(x_enc)

        for _ in range(self.layer):
            x = x_enc
            x_enc = self.fft(x_enc)
            x_enc = x + self.dropout(x_enc)
        dec_out = x_enc[:, -self.pred_len:, :]
        dec_out = self.linear2(dec_out.permute(0, 2, 1)).permute(0, 2, 1)
        if self.is_embed:
            dec_out = self.projection(dec_out)

        # De-Normalization from Non-stationary Transformer
        dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        return dec_out

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
        return dec_out[:, -self.pred_len:, :]  # [B, L, D]
